"""Precompute static website assets for the sleeper detector demo.

Usage:
    cd /home/huynp2/sleeper_detection && ~/.conda/envs/gpt-oss-env/bin/python site/build_static.py

Outputs:
    site/public/data/examples.json
    site/public/data/results.json
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
SITE_DATA = ROOT / "site" / "public" / "data"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.detector_demo import load, get_oof_scores


def main():
    SITE_DATA.mkdir(parents=True, exist_ok=True)

    # 1. Load sound_2x2 data and get out-of-fold scores
    d = load("sound_2x2", "case_id")
    oof = get_oof_scores(d)
    s_probe = oof["probe"]
    s_tfidf = oof["tfidf"]

    y = d["target"]
    g = d["groups"]

    labels = [json.loads(line) for line in open(ROOT / "interp" / "data" / "sound_2x2" / "labels.jsonl")]
    requests_map = {
        r["id"]: r
        for r in (json.loads(line) for line in open(ROOT / "interp" / "data" / "sound_2x2" / "requests.jsonl"))
    }

    # Selected domains: Shopify and Slack (8 requests total)
    target_domains = ("Shopify", "Slack")
    examples = []

    for i, lab in enumerate(labels):
        dom = lab["domain_name"]
        if dom not in target_domains:
            continue

        rid = lab["id"]
        req = requests_map[rid]
        msgs = req["request"]["messages"]
        sys_msg = msgs[0]["content"] if len(msgs) > 0 else ""
        user_msg = msgs[1]["content"] if len(msgs) > 1 else ""
        tool_call = msgs[2]["tool_calls"] if len(msgs) > 2 else []
        tool_resp = msgs[3]["content"] if len(msgs) > 3 else ""
        # Redact any incidental mock emails from tool_response to satisfy privacy checklist
        tool_resp = tool_resp.replace("alice.it@company.com", "alice.it [at] company [dot] com")

        # Training folds: all domains except this one
        te = (g == g[i])
        train_negatives = (~te) & (~y)
        # Threshold is the LODO score at which the training folds had 10% FPR
        probe_thresh = float(np.percentile(s_probe[train_negatives], 90))
        tfidf_thresh = float(np.percentile(s_tfidf[train_negatives], 90))

        probe_score = float(s_probe[i])
        tfidf_score = float(s_tfidf[i])

        examples.append({
            "id": rid,
            "domain": dom,
            "case_id": lab["case_id"],
            "cell": lab["cell"],
            "framing_type": lab["framing_type"],
            "payload_type": lab["payload_type"],
            "user_query": user_msg,
            "system_prompt": sys_msg,
            "tool_call": tool_call,
            "tool_response": tool_resp,
            "memory_note_text": lab["note_text"],
            "probe_score": round(probe_score, 3),
            "probe_threshold": round(probe_thresh, 3),
            "probe_detected": bool(probe_score >= probe_thresh),
            "tfidf_score": round(tfidf_score, 3),
            "tfidf_threshold": round(tfidf_thresh, 3),
            "tfidf_detected": bool(tfidf_score >= tfidf_thresh),
        })

    # Sort examples deterministically: Shopify first, then Slack; cells ordered 1_1, 1_2, 2_1, 2_2
    cell_order = {
        "cell_1_1_live_malicious": 1,
        "cell_1_2_live_benign": 2,
        "cell_2_1_inert_malicious": 3,
        "cell_2_2_inert_benign": 4,
    }
    examples.sort(key=lambda x: (x["domain"], cell_order.get(x["cell"], 99)))

    examples_out = SITE_DATA / "examples.json"
    with open(examples_out, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Wrote {len(examples)} examples to {examples_out}")

    # 2. Results JSON from results/detector_demo/report.json
    report_file = ROOT / "results" / "detector_demo" / "report.json"
    report_data = json.loads(report_file.read_text())

    results_out = SITE_DATA / "results.json"
    with open(results_out, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Wrote results to {results_out}")


if __name__ == "__main__":
    main()
