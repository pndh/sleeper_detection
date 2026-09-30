"""Check the Hugging Face bf16 model against vLLM's recorded replies, then cache activations and run a first lens.

    cd ~/sleeper_detection/interp && ./run_guarded.sh ~/.conda/envs/gpt-oss-env/bin/python validate_and_capture.py

1. Fidelity: teacher-force each recorded vLLM reply and report how often HF's argmax equals the recorded token.
2. Cache: residual stream + router logits at every named position -> cache/<dataset>/ (resumable).
3. First look: logit lens at the last prompt token and per-layer cosine distance, poisoned vs benign equivalent,
   on the Type B no-tool request.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import gptoss as G

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def agreement(ids: list[int], start: int) -> tuple[float, int]:
    m, _ = G.model()
    x = torch.tensor([ids], device=m.device)
    pred = m(x).logits[0, start - 1:-1].argmax(-1).cpu()
    gold = torch.tensor(ids[start:])
    return float((pred == gold).float().mean()), len(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "data" / "demo_v1"))
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    ds = Path(args.dataset)
    cache = Path(args.cache or HERE / "cache" / ds.name)
    requests, labels, outputs = G.load_dataset(ds)
    report = {"dataset": str(ds), "fidelity": {}, "lens": {}, "cosine_distance": {}}

    print("== 1. HF bf16 vs recorded vLLM replies (teacher-forced argmax agreement)")
    for r in requests:
        if r["id"] not in outputs:
            continue
        ids, start = G.teacher_force_ids(outputs[r["id"]])
        acc, n = agreement(ids, start)
        lab = labels[r["id"]]
        report["fidelity"][r["id"]] = {"agreement": acc, "reply_tokens": n}
        print(f"  {lab['turn_type']} {lab['query_name']:22s} {lab['variant']:18s} agreement {acc:.3f} over {n} tokens")

    print("\n== 2. caching activations ->", cache)
    G.capture(ds, cache)

    print("\n== 3. Type B no-tool request: poisoned vs benign equivalent at the last prompt token")
    pick = {labels[r["id"]]["variant"]: r["id"] for r in requests
            if labels[r["id"]]["query_name"] == "B_no_tool"}
    c = G.load_cache(cache, "last_prompt", [pick["poisoned"], pick["benign_equivalent"]])
    rp, rb = torch.tensor(c["resid"][0]).float(), torch.tensor(c["resid"][1]).float()
    cos = torch.nn.functional.cosine_similarity(rp, rb, dim=-1)
    report["cosine_distance"] = [float(1 - v) for v in cos]
    lp, lb = G.logit_lens(rp, 3), G.logit_lens(rb, 3)
    for L in range(0, rp.shape[0], 4):
        print(f"  layer {L:2d}  1-cos {1 - cos[L]:.4f}   poisoned top {lp[L][0][0]!r:14} {lp[L][0][1]:.2f}   "
              f"benign top {lb[L][0][0]!r:14} {lb[L][0][1]:.2f}")
    L = rp.shape[0] - 1
    print(f"  layer {L:2d}  1-cos {1 - cos[L]:.4f}   poisoned {lp[L]}   benign {lb[L]}")
    report["lens"] = {"poisoned": lp, "benign_equivalent": lb}
    (cache / "validate_report.json").write_text(json.dumps(report, indent=1))
    print("\nreport ->", cache / "validate_report.json")


if __name__ == "__main__":
    main()
