"""Build an interp-ready request dataset from the demo generators.

Writes <out>/ with
  requests.jsonl  DEFENDER VIEW. Opaque id, the chat request, exact gpt-oss prompt token ids (as the server's chat
                  endpoint renders them), and label-free position annotations: last prompt token, end of the user
                  message, end of the memory tool result, and the token spans of every stored memory entry.
  labels.jsonl    EVALUATION ONLY (PLAN(2) §3/§12). Case, strategy, state, variant, turn type, which entry is the
                  payload, and observed behaviour from the vLLM runs.
  outputs.jsonl   What the model answered under vLLM (repeat 0): reasoning, text, tool calls. For teacher forcing.
  manifest.json   Sources, start date, counts, sha256 of every file.

Run with the paper-code venv (it has the paper code and openai_harmony):
  cd ~/sleeper_detection && .venv-tools/bin/python interp/build_dataset.py --out interp/data/demo_v1
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
import assemble  # noqa: E402
import harmony_render as hr  # noqa: E402
import type_b  # noqa: E402


def char_offsets(ids: list[int]) -> tuple[str, list[int]]:
    """Rendered text and the start char of every token (cumulative decode, exact for ASCII-aligned boundaries)."""
    enc = hr.encoding()
    starts = [len(enc.decode(ids[:k])) for k in range(len(ids))]
    return enc.decode(ids), starts


def tok_span(starts: list[int], text_len: int, c0: int, c1: int) -> list[int]:
    """Token range [t0, t1) covering chars [c0, c1)."""
    ends = starts[1:] + [text_len]
    t0 = next(i for i in range(len(starts)) if ends[i] > c0)
    t1 = next(i for i in range(len(starts)) if ends[i] >= c1) + 1
    return [t0, t1]


def annotate(req: dict, ids: list[int]) -> dict:
    text, starts = char_offsets(ids)
    n = len(text)
    user = req["messages"][1]["content"]
    tool_content = req["messages"][3]["content"]
    u0 = text.index(user)
    t0 = text.index(tool_content)
    entries = [e for e in eval(tool_content)["entries"]]  # noqa: S307  (paper's own str(dict) output)
    entry_spans = []
    for i, e in enumerate(entries):
        needle = repr(e)[1:-1]
        occ, pos = [], text.find(needle, t0, t0 + len(tool_content))
        while pos != -1:
            occ.append(tok_span(starts, n, pos, pos + len(needle)))
            pos = text.find(needle, pos + 1, t0 + len(tool_content))
        entry_spans.append({"entry_index": i, "text": e, "occurrences": occ})
    return {
        "n_tokens": len(ids),
        "positions": {"last_prompt": len(ids) - 1,
                      "user_end": tok_span(starts, n, u0, u0 + len(user))[1] - 1,
                      "memory_result_end": tok_span(starts, n, t0, t0 + len(tool_content))[1] - 1},
        "entry_spans": entry_spans,
    }


def teacher_forced(ids: list[int], output: dict) -> dict:
    """Prompt + the recorded vLLM reply in harmony format. vLLM returns parsed text, not raw tokens, so this is a
    faithful reconstruction (analysis channel, then the first tool call or the final text), not the exact sample."""
    enc = hr.encoding()
    parts = []
    if output.get("reasoning"):
        parts.append("<|channel|>analysis<|message|>" + output["reasoning"] + "<|end|><|start|>assistant")
    if output.get("tool_calls"):
        c = output["tool_calls"][0]
        parts.append(f"<|channel|>commentary to=functions.{c['name']} <|constrain|>json<|message|>"
                     + json.dumps(c["args"]) + "<|call|>")
    elif output.get("content"):
        parts.append("<|channel|>final<|message|>" + output["content"] + "<|return|>")
    tail = enc.encode("".join(parts), allowed_special=enc.special_tokens_set)
    return {"teacher_forced_ids": list(ids) + tail, "response_start": len(ids)}


def rid(req: dict, ids: list[int]) -> str:
    return hashlib.sha1(json.dumps({"r": req, "i": ids}, sort_keys=True).encode()).hexdigest()[:16]


def latest(pattern: str):
    fs = sorted(glob.glob(str(ROOT / pattern)))
    return json.load(open(fs[-1])) if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "interp" / "data" / "demo_v1"))
    ap.add_argument("--start-date", default=None, help="date the server inserted; default: taken from the run files")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reqs, labels, outputs = [], [], []

    def add(req, start_date, label, output, payload_text=None):
        ids = hr.render_tokens(req, start_date)
        i = rid(req, ids)
        ann = annotate(req, ids)
        reqs.append({"id": i, "request": req, "start_date": start_date, "input_ids": ids, **ann})
        # index of the extra note (payload or hard negative) as it appears in the request's memory result
        label["extra_note_entry_index"] = next((e["entry_index"] for e in ann["entry_spans"] if e["text"] == payload_text),
                                               None) if payload_text else None
        label["payload_entry_index"] = label["extra_note_entry_index"] if label["poisoned"] else None
        labels.append({"id": i, **label})
        if output is not None:
            outputs.append({"id": i, **output, **teacher_forced(ids, output)})

    # Part 2: Type B demo (tool-triggered LIP payload in memory, benign equivalent, no-note reference)
    tb = latest("results/demo_type_b/*_run_*.json")
    case, _, payload = type_b.load()
    notes = {"poisoned": payload, "benign_equivalent": type_b.BENIGN_NOTE, "no_note": None}
    tb_date = args.start_date or (tb["start_date"] if tb else dt.date.today().isoformat())
    for qname, (ttype, query) in type_b.QUERIES.items():
        for v, note in notes.items():
            req = type_b.build_request(case, query, note)
            run = (tb or {}).get("queries", {}).get(qname, {}).get("runs", {}).get(v)
            label = {"source": "type_b", "case_id": type_b.CASE_ID, "strategy": "latent_instruction_planting",
                     "payload_origin_state": "session", "state_target": "memory", "variant": v,
                     "poisoned": v == "poisoned", "hard_negative": v == "benign_equivalent",
                     "turn_type": ttype, "query_name": qname,
                     "behavior": None if not run else {"repeats": len(run["responses"]),
                                                       "attack_fired": run["attack_fired"],
                                                       "benign_conditional_acted": run["benign_conditional_acted"]}}
            output = None if not run else {k: run["responses"][0][k] for k in ("reasoning", "content", "tool_calls")}
            add(req, tb_date, label, output, payload_text=note)

    # Part 1: fires-on-next-read LIP memory payload (every memory read is a trigger turn -> Type D)
    p1 = latest("results/demo/SlackManageMembership_multi_1_1_*.json")
    if p1:
        c1 = assemble.load_case("latent_instruction_planting", "memory", p1["case_id"])
        for qname, q in p1["queries"].items():
            for v in ("clean", "poisoned"):
                req = q["requests"][v]
                run = q["runs"][v]
                label = {"source": "part1", "case_id": p1["case_id"], "strategy": "latent_instruction_planting",
                         "payload_origin_state": "memory", "state_target": "memory", "variant": v,
                         "poisoned": v == "poisoned", "hard_negative": False, "turn_type": "D", "query_name": qname,
                         "behavior": {"repeats": len(run["responses"]),
                                      "attack_fired": run.get("fired_count", sum(j["fired"] for j in run.get("judge_per_repeat", [])))}}
                output = {k: run["responses"][0][k] for k in ("reasoning", "content", "tool_calls")}
                add(req, p1["timestamp"][:10], label, output,
                    payload_text=assemble.planted_note(c1) if v == "poisoned" else None)

    for name, rows in (("requests", reqs), ("labels", labels), ("outputs", outputs)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"created": dt.datetime.now().isoformat(timespec="seconds"), "n_requests": len(reqs),
                "renderer": "demo/harmony_render.py (matches vLLM 0.30 chat endpoint prompt tokens)",
                "sources": {"type_b_run": tb and "results/demo_type_b (latest run)", "part1_run": p1 and "results/demo (latest)"},
                "sha256": {f"{n}.jsonl": hashlib.sha256((out / f"{n}.jsonl").read_bytes()).hexdigest()
                           for n in ("requests", "labels", "outputs")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(reqs)} requests to {out}")
    for r, l in zip(reqs, labels):
        spans = {e["entry_index"]: len(e["occurrences"]) for e in r["entry_spans"]}
        print(f"  {r['id']}  {l['source']:6s} {l['turn_type']} {l['query_name']:22s} {l['variant']:18s} "
              f"tokens={r['n_tokens']} entry-occurrences={spans} extra-note-entry={l['extra_note_entry_index']}")


if __name__ == "__main__":
    main()
