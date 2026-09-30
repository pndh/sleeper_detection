"""AgentDojo runner for a local vLLM gpt-oss-20b endpoint.

Differences from `python -m agentdojo.scripts.benchmark`:
  * temperature is sent explicitly (agentdojo treats 0.0 as "unset" -> vLLM default 1.0)
  * HTTP 500s from vLLM (harmony parse errors on malformed tool-call headers) are retried with a
    temperature ramp; if all retries fail the episode ends gracefully instead of killing the run
  * the model's reasoning (`reasoning_content`) is stored on each assistant message in the run log
  * failures are appended to <logdir>/server_errors.jsonl
"""
import json, logging, time
from pathlib import Path
from typing import Sequence
import click, openai
from openai import NOT_GIVEN
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.llms import openai_llm as OL
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.benchmark import benchmark_suite_with_injections, benchmark_suite_without_injections
from agentdojo.logging import OutputLogger
from agentdojo.models import MODEL_NAMES
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import ChatAssistantMessage, text_content_block_from_string


def _lenient_args(raw: str | None) -> tuple[dict, str | None]:
    """gpt-oss sometimes emits non-JSON tool arguments; agentdojo's strict json.loads would crash the run.
    Returns (args, problem). Bad args are passed through as {} so the tool errors back to the model."""
    if raw is None or not raw.strip():
        return {}, None
    for kw in ({}, {"strict": False}):
        try:
            v = json.loads(raw, **kw)
            return (v, None) if isinstance(v, dict) else ({"value": v}, "non-object json")
        except json.JSONDecodeError:
            pass
    return {}, f"invalid json arguments: {raw[:300]!r}"


def _to_assistant_message(choice) -> ChatAssistantMessage:
    tool_calls, problems = None, []
    if choice.tool_calls:
        tool_calls = []
        for tc in choice.tool_calls:
            args, problem = _lenient_args(tc.function.arguments)
            if problem:
                problems.append({"tool": tc.function.name, "problem": problem})
            tool_calls.append(FunctionCall(function=tc.function.name, args=args, id=tc.id))
    content = [text_content_block_from_string(choice.content)] if choice.content is not None else None
    msg = ChatAssistantMessage(role="assistant", content=content, tool_calls=tool_calls)
    if problems:
        msg["tool_call_problems"] = problems  # type: ignore[typeddict-unknown-key]
    return msg

log = logging.getLogger("run_dojo")


class RobustOpenAILLM(OL.OpenAILLM):
    def __init__(self, client, model, temperature, retry_temps, errlog: Path, max_tokens):
        super().__init__(client, model, temperature=temperature)
        self.name = model
        self.retry_temps, self.errlog, self.max_tokens = retry_temps, errlog, max_tokens

    def _create(self, msgs, tools, temperature):
        return self.client.chat.completions.create(
            model=self.model, messages=msgs, tools=tools or NOT_GIVEN,
            tool_choice="auto" if tools else NOT_GIVEN, temperature=temperature,
            max_tokens=self.max_tokens)

    def query(self, query, runtime, env=OL.EmptyEnv(), messages: Sequence = [], extra_args: dict = {}):
        msgs = [OL._message_to_openai(m, self.model) for m in messages]
        tools = [OL._function_to_openai(t) for t in runtime.functions.values()]
        errors = []
        for attempt, temp in enumerate([self.temperature, *self.retry_temps]):
            try:
                completion = self._create(msgs, tools, temp)
                break
            except (openai.InternalServerError, openai.APIConnectionError, openai.APITimeoutError) as e:
                errors.append(f"{type(e).__name__}: {str(e)[:300]}")
                log.warning(f"server error (attempt {attempt}, T={temp}): {errors[-1][:120]}")
                time.sleep(1.0)
        else:
            with self.errlog.open("a") as f:
                f.write(json.dumps({"ts": time.time(), "n_messages": len(messages), "errors": errors}) + "\n")
            output = ChatAssistantMessage(role="assistant",
                content=[text_content_block_from_string("[SERVER_ERROR: model output could not be parsed by vLLM]")],
                tool_calls=None)
            output["server_errors"] = errors  # type: ignore[typeddict-unknown-key]
            return query, runtime, env, [*messages, output], extra_args
        choice = completion.choices[0].message
        output = _to_assistant_message(choice)
        reasoning = getattr(choice, "reasoning_content", None) or getattr(choice, "reasoning", None)
        if reasoning:
            output["reasoning"] = reasoning  # type: ignore[typeddict-unknown-key]
        if attempt > 0:
            output["retry_attempt"], output["retry_temperature"] = attempt, temp  # type: ignore
        return query, runtime, env, [*messages, output], extra_args


@click.command()
@click.option("--suite", required=True, type=click.Choice(["workspace", "slack", "travel", "banking"]))
@click.option("--attack", default=None, help=f"one of {sorted(ATTACKS)}; omit for clean utility run")
@click.option("--defense", default=None)
@click.option("--user-task", "user_tasks", multiple=True)
@click.option("--injection-task", "injection_tasks", multiple=True)
@click.option("--logdir", default="./runs", type=Path)
@click.option("--benchmark-version", default="v1.2.2")
@click.option("--base-url", default="http://localhost:8000/v1")
@click.option("--model", default="gpt-oss-20b")
@click.option("--temperature", default=0.0, type=float)
@click.option("--retry-temps", default="0.0,0.3,0.7,1.0", help="temperatures for retries after a server error")
@click.option("--max-tokens", default=4096, type=int)
@click.option("--force-rerun", is_flag=True)
@click.option("--quiet", is_flag=True, help="don't stream per-message logs")
def main(suite, attack, defense, user_tasks, injection_tasks, logdir, benchmark_version, base_url, model,
         temperature, retry_temps, max_tokens, force_rerun, quiet):
    logging.basicConfig(level=logging.WARNING if quiet else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = openai.OpenAI(api_key="EMPTY", base_url=base_url, max_retries=2, timeout=600)
    llm = RobustOpenAILLM(client, model, temperature, [float(t) for t in retry_temps.split(",") if t],
                          logdir / "server_errors.jsonl", max_tokens)
    MODEL_NAMES.setdefault(model, "AI assistant")  # how attacks address the model in injections
    pipeline = AgentPipeline.from_config(PipelineConfig(llm=llm, model_id=model, defense=defense, system_message_name=None, system_message=None))
    task_suite = get_suite(benchmark_version, suite)
    logdir.mkdir(parents=True, exist_ok=True)
    with OutputLogger(str(logdir), live=None):
        if attack is None:
            res = benchmark_suite_without_injections(pipeline, task_suite, logdir=logdir, force_rerun=force_rerun,
                                                     user_tasks=user_tasks or None, benchmark_version=benchmark_version)
        else:
            res = benchmark_suite_with_injections(pipeline, task_suite, load_attack(attack, task_suite, pipeline),
                                                  logdir=logdir, force_rerun=force_rerun, user_tasks=user_tasks or None,
                                                  injection_tasks=injection_tasks or None, benchmark_version=benchmark_version)
    u = list(res["utility_results"].values()); s = list(res["security_results"].values())
    print(f"RESULT suite={suite} attack={attack} episodes={len(u)} utility={sum(u)/max(len(u),1):.3f}"
          + (f" attack_success_rate={sum(s)/max(len(s),1):.3f}" if attack else ""))


if __name__ == "__main__":
    main()
