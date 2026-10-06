"""Sample trigger-turn behaviour for interp/data/behavior_v1 under vLLM and judge it (labels by behaviour, not design).

For each request: --k samples at --temperature. Each sample is an agent loop of up to --max-steps assistant
responses. Tool results after the first response are canned (docs/decisions.md P3): MemoryRetrieve returns the same
memory result as the request, every other tool returns {'success': True}. Each sample is judged with the case's own
success conditions (paper Evaluator._compare_condition) two ways, because docs/decisions.md P2 is still open:
  fired_first  a matching call in the first response (the stricter judge used for demo_v1)
  fired_any    a matching call anywhere in the loop (closer to the paper's evaluator, which scans the whole turn)

Writes <dataset>/behavior.jsonl (evaluation only, one row per request id; resumable: done ids are skipped).
vLLM must be serving gpt-oss-20b (agentdojo_gptoss/serve_gptoss.sh), and it inserts its own date into the prompt,
so sample on the build's start_date (manifest.json) or rebuild first.
  cd ~/sleeper_detection && .venv-tools/bin/python demo/sample_behavior.py --k 8
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
import assemble  # noqa: E402,F401  (sets up the paper-code import path)
from src.evaluator import Evaluator  # noqa: E402

EV = Evaluator()


def matches(call: dict, judge: dict) -> bool:
    return call["name"] == judge["target_tool"] and all(
        EV._compare_condition(call["args"].get(c["arg_name"]), c["operator"], c["value"]) for c in judge["conditions"])


def canned_result(name: str, memory_result: str) -> str:
    return memory_result if name == "MemoryRetrieve" else str({"success": True})


def one_sample(client, model, req: dict, judge: dict, temperature: float, max_steps: int, max_tokens: int) -> dict:
    messages = [dict(m) for m in req["messages"]]
    memory_result = req["messages"][3]["content"]
    steps = []
    for step in range(max_steps):
        r = client.chat.completions.create(model=model, messages=messages, tools=req["tools"], tool_choice="auto",
                                           temperature=temperature, max_tokens=max_tokens)
        m = r.choices[0].message
        calls = []
        for t in m.tool_calls or []:
            try:
                args = json.loads(t.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": t.function.arguments}
            calls.append({"id": t.id, "name": t.function.name, "args": args})
        steps.append({"reasoning": getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None),
                      "content": m.content, "tool_calls": [{"name": c["name"], "args": c["args"]} for c in calls],
                      "finish_reason": r.choices[0].finish_reason})
        if not calls:
            break
        messages.append({"role": "assistant", "content": m.content, "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for c in calls]})
        messages += [{"role": "tool", "tool_call_id": c["id"], "content": canned_result(c["name"], memory_result)}
                     for c in calls]
    all_calls = [c for s in steps for c in s["tool_calls"]]
    return {"steps": steps, "fired_first": any(matches(c, judge) for c in steps[0]["tool_calls"]),
            "fired_any": any(matches(c, judge) for c in all_calls), "n_steps": len(steps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "interp" / "data" / "behavior_v1"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=32, help="concurrent requests (vLLM batches them)")
    ap.add_argument("--limit", type=int, default=None, help="first N requests only (smoke tests)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="gpt-oss-20b")
    args = ap.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    today = dt.date.today().isoformat()
    if manifest["start_date"] != today:
        sys.exit(f"built for start_date {manifest['start_date']} but vLLM will insert {today}: rebuild with "
                 f"demo/build_behavior_set.py first, or the activations and the sampled prompts differ")
    reqs = [json.loads(line) for line in open(data / "requests.jsonl")]
    judges = {lab["id"]: lab["judge"] for lab in map(json.loads, open(data / "labels.jsonl"))}
    out_path = data / "behavior.jsonl"
    done = {json.loads(line)["id"] for line in open(out_path)} if out_path.exists() else set()
    todo = [r for r in reqs[:args.limit] if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to sample, k={args.k}, T={args.temperature}, max_steps={args.max_steps}")

    client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY", max_retries=2, timeout=900)

    def run(r):
        samples = [one_sample(client, args.model, r["request"], judges[r["id"]], args.temperature, args.max_steps,
                              args.max_tokens) for _ in range(args.k)]
        return {"id": r["id"], "date": today, "k": args.k, "temperature": args.temperature,
                "max_steps": args.max_steps, "fired_first": sum(s["fired_first"] for s in samples),
                "fired_any": sum(s["fired_any"] for s in samples), "samples": samples}

    with ThreadPoolExecutor(args.workers) as pool, open(out_path, "a") as f:
        futs = [pool.submit(run, r) for r in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(todo)}] {row['id']} fired_first {row['fired_first']}/{row['k']} "
                  f"fired_any {row['fired_any']}/{row['k']}", flush=True)


if __name__ == "__main__":
    main()
