"""Data for the v2 site figures (docs/website_plan.md, section 7). CPU only, a few seconds.

Usage:
    cd /home/huynp2/sleeper_detection && ~/.conda/envs/gpt-oss-env/bin/python site/build_figures.py

Outputs (site/public/data/):
  twins.json         Hero figure. One Sleeper Attack case (ShopifyDeleteProduct_multi_2_2, demo_v1 type_b), three
                     memory variants x three turns. Outputs are the model's recorded vLLM replies (repeat 0). Probe
                     scores come from the detector_demo probe trained on all 40 sound_2x2 requests; none of these 9
                     requests were in its training data.
  distribution.json  Results figure. All 40 sound_2x2 out-of-fold (leave-one-domain-out) scores for the probe and
                     the TF-IDF text baseline, with cell, domain, and each score's percentile among non-target scores.

Thresholds are the 90th percentile of non-target out-of-fold scores (10% FPR), as in build_static.py but over all
domains. The twin scores come from a probe trained on all 40 requests, not from LODO folds, so their percentiles
against the out-of-fold reference are approximate. twins.json says so in its caveats.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.detector_demo import WINDOW, get_oof_scores, load, probe, window  # noqa: E402

OUT = ROOT / "site" / "public" / "data"
V1 = ROOT / "interp" / "data" / "demo_v1"
V1_CACHE = ROOT / "interp" / "cache" / "demo_v1"
TURNS = [("B_no_tool", "Dormant turn 1"), ("B_other_tool", "Dormant turn 2"), ("D_trigger", "Trigger turn")]
LANES = [("no_note", "Clean agent", "No extra note in memory."),
         ("benign_equivalent", "Harmless planted note",
          "Same conditional frame, same trigger tool, harmless action (set a reminder)."),
         ("poisoned", "Attack planted note", "The Sleeper Attack paper's payload: delete product PROD-789.")]


def pct(score: float, ref: np.ndarray) -> float:
    return float(100.0 * np.mean(ref < score))


def output_summary(o: dict) -> dict:
    if o.get("tool_calls"):
        c = o["tool_calls"][0]
        return {"kind": "tool_call", "text": f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})",
                "tool": c["name"]}
    return {"kind": "text", "text": o.get("content") or ""}


def main():
    d = load("sound_2x2", "case_id")
    oof = get_oof_scores(d)
    nontarget = ~d["target"]
    ref = {k: oof[k][nontarget] for k in ("probe", "tfidf")}
    thr = {k: float(np.percentile(ref[k], 90)) for k in ref}

    labels = [json.loads(line) for line in open(ROOT / "interp" / "data" / "sound_2x2" / "labels.jsonl")]
    dist = {"source": "interp/data/sound_2x2 + interp/detector_demo.py (leave-one-domain-out, out-of-fold)",
            "window": list(WINDOW), "thresholds_10pct_fpr": thr, "points": []}
    for i, lab in enumerate(labels):
        dist["points"].append({
            "domain": lab["domain_name"], "cell": lab["cell"], "framing": lab["framing_type"],
            "payload": lab["payload_type"], "target": bool(d["target"][i]),
            **{f"{k}_score": round(float(oof[k][i]), 3) for k in ("probe", "tfidf")},
            **{f"{k}_pct_vs_nontarget": round(pct(oof[k][i], ref[k]), 1) for k in ("probe", "tfidf")},
        })

    m = probe().fit(window(d), d["target"])
    v1_labels = [json.loads(line) for line in open(V1 / "labels.jsonl")]
    v1_out = {o["id"]: o for o in map(json.loads, open(V1 / "outputs.jsonl"))}
    v1_idx = json.load(open(V1_CACHE / "index.json"))
    v1_req = {r["id"]: r for r in map(json.loads, open(V1 / "requests.jsonl"))}
    by_key = {(lab["query_name"], lab["variant"]): lab for lab in v1_labels if lab["source"] == "type_b"}

    twins = {"case_id": "ShopifyDeleteProduct_multi_2_2", "threshold_10pct_fpr": thr["probe"],
             "lanes": [{"key": k, "title": t, "description": desc} for k, t, desc in LANES], "turns": [],
             "caveats": [
                 "One case and one payload. Illustrates the idea; it is not an evaluation.",
                 "The harmless planted note also scores high: the probe responds to a live planted instruction, "
                 "and the harmful one scores higher still.",
                 "Outputs are one recorded vLLM sample per request (temperature 0, which is not fully deterministic "
                 "on vLLM; see docs/decisions.md P5).",
                 "Percentiles compare a probe trained on all 40 sound_2x2 requests against out-of-fold scores, so "
                 "they are approximate."],
             "source": "interp/data/demo_v1 (type_b), interp/cache/demo_v1, probe from interp/detector_demo.py"}
    for qname, ttitle in TURNS:
        turn = {"query_name": qname, "title": ttitle, "cells": {}}
        for variant, _, _ in LANES:
            lab = by_key[(qname, variant)]
            turn["user_query"] = v1_req[lab["id"]]["request"]["messages"][1]["content"]
            k = v1_idx[lab["id"]]["position_names"].index("memory_result_end")
            r = np.load(V1_CACHE / f"{lab['id']}.npz")["resid"][k].astype(np.float32)
            s = float(m.decision_function(r[WINDOW[0]:WINDOW[1] + 1].mean(0)[None])[0])
            note = None
            if lab.get("extra_note_entry_index") is not None:
                note = v1_req[lab["id"]]["entry_spans"][lab["extra_note_entry_index"]]["text"]
            turn["cells"][variant] = {"probe_score": round(s, 3), "probe_pct_vs_nontarget": round(pct(s, ref["probe"]), 1),
                                      "flagged": s >= thr["probe"], "output": output_summary(v1_out[lab["id"]]),
                                      "planted_note": note}
        twins["turns"].append(turn)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "twins.json").write_text(json.dumps(twins, indent=1, ensure_ascii=False))
    (OUT / "distribution.json").write_text(json.dumps(dist, indent=1))
    print(f"threshold probe {thr['probe']:.2f}  tfidf {thr['tfidf']:.2f}")
    for t in twins["turns"]:
        print(f"{t['title']:15s}", "  ".join(f"{v}: {c['probe_score']:+6.2f} ({c['probe_pct_vs_nontarget']:.0f}%) "
                                             f"{c['output']['text'][:28]!r}" for v, c in t["cells"].items()))


if __name__ == "__main__":
    main()
