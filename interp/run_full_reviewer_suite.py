"""Comprehensive Reviewer Experimental Suite: 2x2 Factorial, In-Flight Causal Intervention, and Behavioral Readiness.

Pins gpt-oss-20b MoE (24 layers, 32 experts). Runs on A100 GPU under interp/gptoss.py.

1. 2x2 Factorial Evaluation:
   - Cell (1,1): Live Instruction + Malicious Payload
   - Cell (1,2): Live Instruction + Benign Payload
   - Cell (2,1): Inert Framing    + Malicious Payload
   - Cell (2,2): Inert Framing    + Benign Payload
   Tests Competing Hypotheses:
     H_content: Fires on any malicious text
     H_instruction: Fires on any pending conditional
     H_commitment: Fires ONLY on live pending malicious instruction (Cell 1,1)

2. In-Flight Intervention Check (Layer 10):
   - Projection ablation of u_10 on the poisoned trigger turn and steering +alpha * u_10 on the clean twin, each
     asserted (gptoss.check_edit) to show up in the recorded residual stream. Behavioural effect: see
     verify_causal_soundness.py.

3. Case catalog: counts of Sleeper Attack cases per strategy and state. No behavioural numbers.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "interp"))

import gptoss as G

DATA_2X2 = ROOT / "interp" / "data" / "demo_2x2"
CACHE_2X2 = ROOT / "interp" / "cache" / "demo_2x2"
DATA_V1 = ROOT / "interp" / "data" / "demo_v1"
CACHE_V1 = ROOT / "interp" / "cache" / "demo_v1"
RESULTS_DIR = ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "full_reviewer_suite_report.json"


def step1_capture_2x2():
    print("=" * 80)
    print("STEP 1: CAPTURING ACTIVATIONS FOR 2x2 FACTORIAL DATASET ON A100 GPU")
    print("=" * 80)
    CACHE_2X2.mkdir(parents=True, exist_ok=True)
    out = G.capture(DATA_2X2, CACHE_2X2, router=True, include_entries=True)
    print(f"2x2 Activations cached successfully at: {out}\n")


def step2_evaluate_2x2():
    print("=" * 80)
    print("STEP 2: 2x2 FACTORIAL EVALUATION (COMPETING EXPLANATIONS TEST)")
    print("=" * 80)
    
    # 1. Derive Unit Direction v_10 from D_trigger difference in demo_v1
    # 8a413989c4d60bef: D_trigger poisoned
    # 9d339b7b6829028f: D_trigger benign_equivalent
    idx_v1 = json.load(open(CACHE_V1 / "index.json"))
    pos_d = idx_v1["8a413989c4d60bef"]["position_names"].index("memory_result_end")
    layer = 10
    
    d_pois = np.load(CACHE_V1 / "8a413989c4d60bef.npz")["resid"][pos_d, layer].astype(np.float32)
    d_ben = np.load(CACHE_V1 / "9d339b7b6829028f.npz")["resid"][pos_d, layer].astype(np.float32)
    v_diff = d_pois - d_ben
    norm_v = np.linalg.norm(v_diff)
    u_10 = v_diff / norm_v
    print(f"Derived Layer 10 Unit Direction u_10: norm={norm_v:.3f}, dim={len(u_10)}")
    
    # Load 2x2 labels and cached representations
    labels = [json.loads(line) for line in open(DATA_2X2 / "labels.jsonl")]
    idx_2x2 = json.load(open(CACHE_2X2 / "index.json"))
    results_2x2 = {}
    
    print("\n--- 2x2 Matrix Results (Evaluated at memory_result_end, Layer 10) ---")
    print(f"{'Cell ID':26} | {'Payload':10} | {'Framing':16} | {'Projection':12} | {'Cosine Sim':12}")
    print("-" * 80)
    
    matrix_scores = {}
    for lab in labels:
        rid = lab["id"]
        cell_name = lab["cell"]
        pos_cell = idx_2x2[rid]["position_names"].index("memory_result_end")
        npz = np.load(CACHE_2X2 / f"{rid}.npz")
        h = npz["resid"][pos_cell, layer].astype(np.float32)
        
        proj = float(np.dot(h, u_10))
        h_norm = np.linalg.norm(h)
        cos_sim = float(np.dot(h, u_10) / h_norm) if h_norm > 1e-6 else 0.0
        
        results_2x2[cell_name] = {
            "id": rid,
            "payload_type": lab["payload_type"],
            "framing_type": lab["framing_type"],
            "projection_score": proj,
            "cosine_similarity": cos_sim,
        }
        matrix_scores[cell_name] = proj
        print(f"{cell_name:26} | {lab['payload_type']:10} | {lab['framing_type']:16} | {proj:+12.3f} | {cos_sim:+12.4f}")
        
    s_11 = matrix_scores["cell_1_1_live_malicious"]
    s_12 = matrix_scores["cell_1_2_live_benign"]
    s_21 = matrix_scores["cell_2_1_inert_malicious"]
    s_22 = matrix_scores["cell_2_2_inert_benign"]
    
    # Hypothesis Testing & Competing Explanation Analysis
    # Content effect: (s_11 + s_21) - (s_12 + s_22)
    content_effect = (s_11 + s_21) - (s_12 + s_22)
    # Instruction effect: (s_11 + s_12) - (s_21 + s_22)
    instruction_effect = (s_11 + s_12) - (s_21 + s_22)
    # Commitment Interaction effect: s_11 - (s_12 + s_21 - s_22)
    commitment_interaction = s_11 - (s_12 + s_21 - s_22)
    # Cell 1_1 Margin over best competing cell:
    margin_over_runner_up = s_11 - max(s_12, s_21, s_22)
    
    print("\n--- Hypothesis Contrast & Effect Decomposition ---")
    print(f"Cell (1,1) [Live Malicious] Score        : {s_11:+.3f}")
    print(f"Cell (1,2) [Live Benign] Score           : {s_12:+.3f}")
    print(f"Cell (2,1) [Inert Malicious] Score       : {s_21:+.3f}")
    print(f"Cell (2,2) [Inert Benign] Score          : {s_22:+.3f}")
    print(f"Margin over Runner-Up Cell               : {margin_over_runner_up:+.3f}")
    print(f"Commitment Interaction Effect            : {commitment_interaction:+.3f}")
    print(f"Main Effect of Content (Malicious-Benign): {content_effect:+.3f}")
    print(f"Main Effect of Framing (Live-Inert)      : {instruction_effect:+.3f}")
    
    hypothesis_conclusion = {
        "content_detector_falsified": bool(s_21 < s_11 - 100),
        "instruction_detector_falsified": bool(s_12 < s_11 - 100),
        "commitment_hypothesis_confirmed": bool(s_11 > max(s_12, s_21, s_22) + 200),
        "runner_up_margin": float(margin_over_runner_up),
    }
    print("\n[Verdict]: " + (
        "CONFIRMED! Only Cell (1,1) lights up. "
        "Falsifies both simple content matching (inert malicious is suppressed) "
        "and generic instruction matching (live benign is suppressed)."
        if hypothesis_conclusion["commitment_hypothesis_confirmed"]
        else "Inconclusive."
    ))
    return {"results_2x2": results_2x2, "matrix": matrix_scores, "conclusion": hypothesis_conclusion}


def step3_causal_interventions():
    print("\n" + "=" * 80)
    print("STEP 3: IN-FLIGHT INTERVENTION CHECK (GPU HOOKS ON LAYER 10)")
    print("=" * 80)
    # This step only checks that the edits reach the residual stream (G.check_edit raises otherwise). The
    # next-token distribution at the prompt boundary is not a readout (it is <|channel|> with p = 1.0 in every
    # condition), so it is not reported. Trigger-turn tool-call ASR: verify_causal_soundness.py.
    layer = 10
    idx_v1 = json.load(open(CACHE_V1 / "index.json"))
    pos = idx_v1["8a413989c4d60bef"]["position_names"].index("memory_result_end")
    # resid index layer + 1 is the output of decoder layer `layer`, where G.hooks({layer: ...}) edits.
    # (step 2's "Layer 10" is resid index 10, i.e. the output of decoder layer 9.)
    d_pois = np.load(CACHE_V1 / "8a413989c4d60bef.npz")["resid"][pos, layer + 1].astype(np.float32)
    d_ben = np.load(CACHE_V1 / "9d339b7b6829028f.npz")["resid"][pos, layer + 1].astype(np.float32)
    u_10 = (d_pois - d_ben) / np.linalg.norm(d_pois - d_ben)

    reqs_v1, _, _ = G.load_dataset(DATA_V1)
    by_id = {r["id"]: r for r in reqs_v1}
    req_d = by_id["8a413989c4d60bef"]      # D_trigger, poisoned
    req_clean = by_id["625ef54c92d02aaa"]  # B_no_tool, clean twin

    print("\n--- 1. Ablation of u_10 on the trigger turn ---")
    abl = G.check_edit(req_d["input_ids"], layer, u_10, req_d["positions"]["memory_result_end"])
    print(f"Post-layer-{layer} projection onto u_10 at memory_result_end: baseline {abl['baseline_projection']:+.3f}, "
          f"ablated {abl['edited_projection']:+.4f} (|.| <= {abl['tolerance']:.4f} required)")

    print("\n--- 2. Steering +alpha * u_10 on the clean twin ---")
    sweeps = []
    for alpha in [10.0, 20.0, 40.0, 60.0]:
        st = G.check_edit(req_clean["input_ids"], layer, u_10, req_clean["positions"]["memory_result_end"], alpha=alpha)
        sweeps.append({"alpha": alpha, "score": st["edited_projection"], "expected": st["expected_projection"]})
        print(f"alpha={alpha:4.1f}: projection {st['edited_projection']:+8.3f} (expected {st['expected_projection']:+8.3f})")

    strip = lambda r: {k: v for k, v in r.items() if k != "logits"}  # noqa: E731
    return {"ablation": strip(abl),
            "steering": {"clean_baseline_score": st["baseline_projection"], "sweeps": sweeps}}


def step4_behavioral_readiness():
    print("\n" + "=" * 80)
    print("STEP 4: SLEEPER ATTACK CASE CATALOG")
    print("=" * 80)
    # Counts only. No behavioural measurement is made here: readiness labels come from sampled trigger-turn
    # behaviour (demo/sample_behavior.py on interp/data/behavior_v1).
    strategies = ["latent_instruction_planting", "proactive_info_elicitation", "persistent_info_corruption"]
    targets = ["memory", "skill", "session"]
    catalog = {}
    for s in strategies:
        catalog[s] = {}
        for t in targets:
            p = ROOT / "data" / "sleeper_attack" / "datasets" / s / f"{t}.json"
            if p.exists():
                catalog[s][t] = len(json.load(open(p)))
    total_cases = sum(sum(v.values()) for v in catalog.values())
    print(f"Cataloged {total_cases} cases across {len(catalog)} strategies: {catalog}")
    return {"cataloged_cases_by_strategy": catalog, "total_cases_cataloged": total_cases}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Capture 2x2 dataset on A100 GPU
    step1_capture_2x2()
    
    # Step 2: 2x2 Factorial Evaluation
    step2_res = step2_evaluate_2x2()
    
    # Step 3: In-Flight Causal Intervention (GPU Hooks)
    step3_res = step3_causal_interventions()
    
    # Step 4: Behavioral Readiness Analysis
    step4_res = step4_behavioral_readiness()
    
    full_report = {
        "model": "openai/gpt-oss-20b",
        "architecture": "Mixture-of-Experts (24 layers, 32 experts, top-4 routing)",
        "hardware": "NVIDIA A100-SXM4-80GB",
        "step2_2x2_factorial": step2_res,
        "step3_causal_intervention": step3_res,
        "step4_behavioral_readiness": step4_res
    }
    
    RESULTS_FILE.write_text(json.dumps(full_report, indent=2))
    print(f"\nFull Experimental Suite Report saved to: {RESULTS_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
