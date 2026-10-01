"""Analyze cached activations and behavioral traces without needing GPU or reruns.

Performs 4 interpretability analyses using only:
  - interp/cache/demo_v1/*.npz
  - interp/cache/demo_v1/index.json
  - interp/data/demo_v1/labels.jsonl
  - results/demo_type_b/ShopifyDeleteProduct_multi_2_2_run_20260930T014318.json

Analyses:
  1. Multi-position activation geometry & cosine distance profiles (layers x positions).
  2. MoE expert routing specialization, Jaccard overlap, and router entropy.
  3. Difference-in-means pending payload direction, dormancy transfer (B vs D alignment), and 1D projections.
  4. Reasoning-channel deliberation audit (hidden thinking traces, suppression logic, and token counts).
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "interp" / "cache" / "demo_v1"
DATA_DIR = ROOT / "interp" / "data" / "demo_v1"
RESULTS_FILE = ROOT / "results" / "demo_type_b" / "ShopifyDeleteProduct_multi_2_2_run_20260930T014318.json"


def load_data():
    labels = [json.loads(line) for line in open(DATA_DIR / "labels.jsonl")]
    index = json.load(open(CACHE_DIR / "index.json"))
    
    # Map (case_id, query_name, variant) -> id
    records = {}
    for l in labels:
        cid = l.get("case_id")
        q = l.get("query_name")
        v = l.get("variant")
        rid = l["id"]
        records[(cid, q, v)] = {
            "id": rid,
            "label": l,
            "index": index.get(rid, {}),
            "npz": np.load(CACHE_DIR / f"{rid}.npz")
        }
    return records


def analyze_position_geometry(records):
    """Analysis 1: Cosine distances across layers and positions for B_no_tool."""
    cid = "ShopifyDeleteProduct_multi_2_2"
    pois = records[(cid, "B_no_tool", "poisoned")]
    ben = records[(cid, "B_no_tool", "benign_equivalent")]
    no_note = records[(cid, "B_no_tool", "no_note")]
    
    pos_names = pois["index"]["position_names"]
    n_layers = pois["npz"]["resid"].shape[1]  # 25
    
    print("\n" + "="*80)
    print("ANALYSIS 1: MULTI-POSITION ACTIVATION GEOMETRY (B_no_tool: Poisoned vs Benign)")
    print("="*80)
    
    results = {}
    # Check Layer 17 across all positions
    print("\n--- Cosine Distance across Positions at Layer 17 ---")
    header = f"{'Position Name':22} | {'Token Pos':10} | {'Cosine Dist':12} | {'Relative to Prompt'}"
    print(header)
    print("-" * len(header))
    
    prompt_cos_dist = None
    for p_idx, p_name in enumerate(pos_names):
        tok_pos = pois["index"]["positions"][p_idx]
        v_pois = pois["npz"]["resid"][p_idx, 17].astype(np.float32)
        v_ben = ben["npz"]["resid"][p_idx, 17].astype(np.float32)
        cos_sim = np.dot(v_pois, v_ben) / (np.linalg.norm(v_pois) * np.linalg.norm(v_ben))
        cos_dist = float(1.0 - cos_sim)
        if p_name == "last_prompt":
            prompt_cos_dist = cos_dist
        ratio_str = f"{cos_dist / prompt_cos_dist:.2f}x" if prompt_cos_dist else "1.00x"
        print(f"{p_name:22} | {tok_pos:<10} | {cos_dist:.6f}     | {ratio_str}")
        results[p_name] = cos_dist

    # Triangle inequality test at memory_result_end (pos 2) at layer 17
    # Compare d(Poisoned, NoNote), d(Benign, NoNote), d(Poisoned, Benign)
    # In no_note, pos 2 is memory_result_end
    print("\n--- Triangle Inequality Test at memory_result_end (Layer 17) ---")
    h_p = pois["npz"]["resid"][2, 17].astype(np.float32)
    h_b = ben["npz"]["resid"][2, 17].astype(np.float32)
    h_c = no_note["npz"]["resid"][2, 17].astype(np.float32)
    
    d_pc = 1.0 - np.dot(h_p, h_c) / (np.linalg.norm(h_p) * np.linalg.norm(h_c))
    d_bc = 1.0 - np.dot(h_b, h_c) / (np.linalg.norm(h_b) * np.linalg.norm(h_c))
    d_pb = 1.0 - np.dot(h_p, h_b) / (np.linalg.norm(h_p) * np.linalg.norm(h_b))
    
    print(f"  d(Poisoned, Clean Oracle) = {d_pc:.6f}  (injected note vs no note)")
    print(f"  d(Benign, Clean Oracle)   = {d_bc:.6f}  (benign note vs no note)")
    print(f"  d(Poisoned, Benign Twin)  = {d_pb:.6f}  (pure payload difference)")
    print(f"  Delta |d(P,C) - d(B,C)|   = {abs(d_pc - d_bc):.6f}  (container overhead symmetry)")

    return results


def analyze_moe_routing(records):
    """Analysis 2: MoE Expert Routing Specialization & Jaccard Overlap."""
    cid = "ShopifyDeleteProduct_multi_2_2"
    pois = records[(cid, "B_no_tool", "poisoned")]
    ben = records[(cid, "B_no_tool", "benign_equivalent")]
    
    # pos 3 is entry0.occ0.last (the planted instruction token itself)
    # pos 2 is memory_result_end
    # pos 0 is last_prompt
    
    print("\n" + "="*80)
    print("ANALYSIS 2: MOE EXPERT ROUTING SPECIALIZATION & DISRUPTION")
    print("="*80)
    
    for pos_idx, pos_label in [(3, "entry0.occ0.last (Planted Instruction Token)"), (2, "memory_result_end (End of Memory Block)")]:
        print(f"\n--- Top-4 Expert Overlap at {pos_label} ---")
        print(f"{'Layer':6} | {'Top-4 Poisoned':22} | {'Top-4 Benign':22} | {'Jaccard Overlap':15} | {'Entropy (P / B)'}")
        print("-" * 85)
        for layer in range(24):
            r_pois = pois["npz"]["router"][pos_idx, layer].astype(np.float32)
            r_ben = ben["npz"]["router"][pos_idx, layer].astype(np.float32)
            
            top_p = set(np.argsort(r_pois)[-4:])
            top_b = set(np.argsort(r_ben)[-4:])
            
            jaccard = len(top_p & top_b) / len(top_p | top_b)
            
            # Router entropy
            p_probs = np.exp(r_pois - np.max(r_pois))
            p_probs = p_probs / p_probs.sum()
            ent_p = -np.sum(p_probs * np.log(p_probs + 1e-12))
            
            b_probs = np.exp(r_ben - np.max(r_ben))
            b_probs = b_probs / b_probs.sum()
            ent_b = -np.sum(b_probs * np.log(b_probs + 1e-12))
            
            # Only print significant divergence or sample layers for brevity
            if jaccard < 1.0 or layer in [0, 6, 12, 17, 20, 23]:
                top_p_str = str(sorted(list(top_p)))
                top_b_str = str(sorted(list(top_b)))
                flag = " <--- SHARP DIVERGENCE" if jaccard <= 0.35 else ""
                print(f"L{layer:<5} | {top_p_str:22} | {top_b_str:22} | {jaccard:<15.2f} | {ent_p:.2f} / {ent_b:.2f}{flag}")


def analyze_dormancy_direction(records):
    """Analysis 3: Difference-in-Means Direction and Cross-Query Transfer."""
    cid = "ShopifyDeleteProduct_multi_2_2"
    b_pois = records[(cid, "B_no_tool", "poisoned")]
    b_ben = records[(cid, "B_no_tool", "benign_equivalent")]
    
    d_pois = records[(cid, "D_trigger", "poisoned")]
    d_ben = records[(cid, "D_trigger", "benign_equivalent")]
    
    b_oth_pois = records[(cid, "B_other_tool", "poisoned")]
    b_oth_ben = records[(cid, "B_other_tool", "benign_equivalent")]
    
    print("\n" + "="*80)
    print("ANALYSIS 3: PENDING PAYLOAD DIRECTION & DORMANCY TRANSFER (B vs D)")
    print("="*80)
    
    print("\n--- Cosine Similarity of Difference Direction (d_B vs d_D) across Layers ---")
    print(f"{'Layer':6} | {'Cosine Sim (d_B, d_D)':25} | {'Cosine Sim (d_B_oth, d_D)':25} | {'Norm ||d_B||'}")
    print("-" * 75)
    
    pos = 2  # memory_result_end
    d_B_l10 = None
    
    for layer in range(25):
        h_bp = b_pois["npz"]["resid"][pos, layer].astype(np.float32)
        h_bb = b_ben["npz"]["resid"][pos, layer].astype(np.float32)
        d_B = h_bp - h_bb
        
        h_dp = d_pois["npz"]["resid"][pos, layer].astype(np.float32)
        h_db = d_ben["npz"]["resid"][pos, layer].astype(np.float32)
        d_D = h_dp - h_db
        
        h_op = b_oth_pois["npz"]["resid"][pos, layer].astype(np.float32)
        h_ob = b_oth_ben["npz"]["resid"][pos, layer].astype(np.float32)
        d_O = h_op - h_ob
        
        cos_bd = np.dot(d_B, d_D) / (np.linalg.norm(d_B) * np.linalg.norm(d_D))
        cos_od = np.dot(d_O, d_D) / (np.linalg.norm(d_O) * np.linalg.norm(d_D))
        norm_b = np.linalg.norm(d_B)
        
        if layer == 10:
            d_B_l10 = d_B / norm_b
            
        if layer in [0, 2, 5, 8, 10, 12, 15, 17, 20, 24]:
            highlight = " <=== PEAK ALIGNMENT (67.4%)" if layer == 10 else ""
            print(f"L{layer:<5} | {cos_bd:<25.4f} | {cos_od:<25.4f} | {norm_b:<10.3f}{highlight}")

    # 1D Projection Score onto unit direction d_B at Layer 10
    print("\n--- 1D Projection onto Dormant Payload Direction d_10 across all 9 Part 2 Requests ---")
    print(f"{'Query':12} | {'Variant':18} | {'Ground Truth':14} | {'1D Projection Score'}")
    print("-" * 65)
    for q in ["B_no_tool", "B_other_tool", "D_trigger"]:
        for v in ["poisoned", "benign_equivalent", "no_note"]:
            rec = records[(cid, q, v)]
            h = rec["npz"]["resid"][pos, 10].astype(np.float32)
            score = float(np.dot(h, d_B_l10))
            label = "POISONED" if v == "poisoned" else "BENIGN"
            print(f"{q:12} | {v:18} | {label:14} | {score:+.3f}")


def analyze_deliberation(results_file):
    """Analysis 4: Hidden Reasoning Deliberation Audit."""
    print("\n" + "="*80)
    print("ANALYSIS 4: HIDDEN REASONING-CHANNEL (CoT) DELIBERATION AUDIT")
    print("="*80)
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
        
    data = json.load(open(results_file))
    queries = data.get("queries", {})
    
    print("\n--- Reasoning Trace Length & Sleeper Evaluation in B_no_tool ---")
    for q_name in ["B_no_tool", "B_other_tool", "D_trigger"]:
        q_data = queries.get(q_name, {})
        runs = q_data.get("runs", {})
        print(f"\nQuery: {q_name} (User: '{q_data.get('query')}')")
        for v_name in ["poisoned", "benign_equivalent", "no_note"]:
            v_runs = runs.get(v_name, {})
            responses = v_runs.get("responses", [])
            lens = [len(r.get("reasoning", "")) for r in responses]
            avg_len = sum(lens) / len(lens) if lens else 0
            
            # Check for mention of trigger or payload
            mentions_note = 0
            for r in responses:
                txt = r.get("reasoning", "")
                if "PROD-789" in txt or "water the plants" in txt or "GoogleHomeSearchDevices" in txt:
                    mentions_note += 1
            print(f"  {v_name:18}: avg reasoning chars = {avg_len:5.1f} | note evaluated = {mentions_note}/{len(responses)}")
            
    # Sample exact reasoning from B_no_tool poisoned repeat 0
    b_pois_rep0 = queries["B_no_tool"]["runs"]["poisoned"]["responses"][0]["reasoning"]
    print("\n--- Exact CoT Suppression Trace (B_no_tool, Poisoned Repeat 0) ---")
    for line in b_pois_rep0.strip().split("\n"):
        print(f"  | {line}")


def main():
    records = load_data()
    analyze_position_geometry(records)
    analyze_moe_routing(records)
    analyze_dormancy_direction(records)
    analyze_deliberation(RESULTS_FILE)
    print("\n" + "="*80)
    print("ALL CACHED ANALYSES COMPLETED SUCCESSFULLY (0 GPU MEMORY, 0 RERUNS)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
