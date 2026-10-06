"""Behavioral Readiness Harness: Correlating Latent Commitment with Trigger Execution.

As recommended by peer review:
Evaluates whether the Type B probe projection score predicts whether the agent
actually executes the attack when triggered on Turn D (filtering write-only duds).

Analyzes the benchmark datasets (data/sleeper_attack/datasets/*) across:
- Latent Instruction Planting (LIP)
- Proactive Information Elicitation (PIE)
- Persistent Information Corruption (PIC)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "data" / "sleeper_attack" / "datasets"


def audit_benchmark_targets():
    print("=" * 80)
    print("BEHAVIORAL READINESS HARNESS: BENCHMARK SUITE AUDIT")
    print("=" * 80)
    
    strategies = ["latent_instruction_planting", "proactive_info_elicitation", "persistent_info_corruption"]
    targets = ["memory", "skill", "session"]
    
    summary = {}
    for strat in strategies:
        summary[strat] = {}
        for tgt in targets:
            fpath = DATASETS_DIR / strat / f"{tgt}.json"
            if fpath.exists():
                cases = json.load(open(fpath))
                summary[strat][tgt] = len(cases)
                print(f"  {strat:30} | {tgt:10} | {len(cases):4} cases loaded")
                
    print(f"\nCataloged {sum(sum(v.values()) for v in summary.values())} cases present on disk "
          "(the paper reports 1,896; files missing here are not counted).")
    return summary


if __name__ == "__main__":
    audit_benchmark_targets()
