"""Model-agnostic tool-calling agent loop over the OpenAI chat-completions API.

Backends (same convention as agentdojo_gptoss/ and eval/judge_guard.py): a local vLLM OpenAI-compatible server,
or OpenRouter. The loop runs one session: feed a user message, let the model call tools until it stops.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import openai

from .tools import ToolRuntime


def make_client(base_url: str | None = None, api_key: str | None = None) -> openai.OpenAI:
    """Local vLLM by default; OpenRouter if OPENROUTER_API_KEY is set and no base_url given."""
    if base_url is None and os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("COMPDORM_LOCAL"):
        base_url = "https://openrouter.ai/api/v1"
        api_key = api_key or os.environ["OPENROUTER_API_KEY"]
    base_url = base_url or os.environ.get("COMPDORM_BASE_URL", "http://localhost:8000/v1")
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "EMPTY"
    return openai.OpenAI(base_url=base_url, api_key=api_key)


def _to_openai_tools(schemas: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": s["name"], "description": s["description"],
                                              "parameters": s["input_schema"]}} for s in schemas]


@dataclass
class SessionResult:
    messages: list[dict]
    runtime: ToolRuntime
    final_text: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    stopped: str = "completed"  # completed | max_turns | error
    error: str | None = None


def run_session(client, model: str, system_prompt: str, user_message: str, runtime: ToolRuntime,
                temperature: float = 0.0, max_turns: int = 12, max_tokens: int = 1024) -> SessionResult:
    """Run one agent session to completion (or max_turns). Returns the transcript and the runtime's logs."""
    system = system_prompt.replace("{MEMORY_ENTRIES}", runtime.memory.render())
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user_message}]
    tools = _to_openai_tools(runtime.schemas())
    res = SessionResult(messages=messages, runtime=runtime)

    for _ in range(max_turns):
        try:
            completion = client.chat.completions.create(
                model=model, messages=messages, tools=tools, tool_choice="auto",
                temperature=temperature, max_tokens=max_tokens)
        except Exception as e:  # network / server error — end the session gracefully
            res.stopped, res.error = "error", f"{type(e).__name__}: {e}"
            return res

        choice = completion.choices[0].message
        assistant_msg: dict = {"role": "assistant", "content": choice.content or ""}
        if choice.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in choice.tool_calls]
        messages.append(assistant_msg)

        if not choice.tool_calls:
            res.final_text = choice.content or ""
            res.stopped = "completed"
            return res

        for tc in choice.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                if not isinstance(args, dict):
                    args = {"value": args}
            except json.JSONDecodeError:
                args = {}
            result = runtime.execute(tc.function.name, args)
            res.tool_calls.append({"name": tc.function.name, "args": args})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, ensure_ascii=False)})

    res.stopped = "max_turns"
    return res
