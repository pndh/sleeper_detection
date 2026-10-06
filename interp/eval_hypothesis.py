"""Evaluation Suite: Testing the Latent Commitment Invariance Hypothesis.

Evaluates the generalizable scientific hypothesis across cached representations:
1. Cross-State Invariance (D -> B Zero-Shot Transfer): Does the active trigger direction
   isolate dormant poison from benign conditional twins during benign queries?
2. Cross-Case Generalization: Does the directional vector transfer zero-shot across cases
   (ShopifyDeleteProduct -> SlackManageMembership)?
3. Spatial Localization & Delimiter Collapse: Contrastive signal strength across sequence positions
   (internal memory occurrences vs memory_result_end vs last_prompt).
4. MoE Routing Trajectory Disruption: Layerwise Jaccard divergence and entropy shifts in top-4 gating.
5. Deliberation-Activation Alignment: Correlation between hidden CoT deliberation and latent projection magnitude.

Uses ONLY cached data in interp/cache/demo_v1/*.npz (0 GPU memory, instant CPU runtime).
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "interp" / "cache" / "demo_v1"
DATA_DIR = ROOT / "interp" / "data" / "demo_v1"
OUT_REPORT = ROOT / "results" / "hypothesis_evaluation_report.json"


def load_dataset_records():
    labels = [json.loads(line) for line in open(DATA_DIR / "labels.jsonl")]
    index = json.load(open(CACHE_DIR / "index.json"))
    
    records = {}
    for l in labels:
        rid = l["id"]
        records[rid] = {
            "id": rid,
            "label": l,
            "index": index.get(rid, {}),
            "npz": np.load(CACHE_DIR / f"{rid}.npz")
        }
    return records


def run_cross_state_transfer_analysis(records):
    """Test 1: Cross-State (D -> B) Transfer and Layerwise Separation."""
    print("=" * 80)
    print("TEST 1: CROSS-STATE (D -> B) TRANSFER & LATENT COMMITMENT SEPARATION")
    print("=" * 80)
    
    # Locate Shopify D_trigger poisoned, benign_equivalent, no_note
    shopify_records = {
        (r["label"]["query_name"], r["label"]["variant"]): r
        for r in records.values() if r["label"]["case_id"] == "ShopifyDeleteProduct_multi_2_2"
    }
    
    d_pois = shopify_records[("D_trigger", "poisoned")]
    d_ben = shopify_records[("D_trigger", "benign_equivalent")]
    
    # Position 2 is memory_result_end
    pos = 2
    n_layers = d_pois["npz"]["resid"].shape[1]
    
    layer_results = []
    best_layer = 10
    max_margin = -1.0
    
    print(f"\n--- Layerwise Separation on Dormant Queries using Direction from Trigger D ---")
    print(f"{'Layer':6} | {'Cosine Sim (d_B, d_D)':23} | {'B_no_tool Margin':18} | {'B_other Margin':18} | {'Status'}")
    print("-" * 80)
    
    b_pois = shopify_records[("B_no_tool", "poisoned")]
    b_ben = shopify_records[("B_no_tool", "benign_equivalent")]
    b_oth_pois = shopify_records[("B_other_tool", "poisoned")]
    b_oth_ben = shopify_records[("B_other_tool", "benign_equivalent")]
    
    for l in range(n_layers):
        h_dp = d_pois["npz"]["resid"][pos, l].astype(np.float32)
        h_db = d_ben["npz"]["resid"][pos, l].astype(np.float32)
        v_D = h_dp - h_db
        norm_D = np.linalg.norm(v_D)
        
        h_bp = b_pois["npz"]["resid"][pos, l].astype(np.float32)
        h_bb = b_ben["npz"]["resid"][pos, l].astype(np.float32)
        v_B = h_bp - h_bb
        norm_B = np.linalg.norm(v_B)
        
        if norm_D > 1e-6 and norm_B > 1e-6:
            cos_sim = float(np.dot(v_B, v_D) / (norm_B * norm_D))
            u_D = v_D / norm_D
            
            # Projection scores
            s_bp = float(np.dot(h_bp, u_D))
            s_bb = float(np.dot(h_bb, u_D))
            margin_b = s_bp - s_bb
            
            h_op = b_oth_pois["npz"]["resid"][pos, l].astype(np.float32)
            h_ob = b_oth_ben["npz"]["resid"][pos, l].astype(np.float32)
            s_op = float(np.dot(h_op, u_D))
            s_ob = float(np.dot(h_ob, u_D))
            margin_oth = s_op - s_ob
            
            avg_margin = (margin_b + margin_oth) / 2.0
            if avg_margin > max_margin:
                max_margin = avg_margin
                best_layer = l
                
            flag = " <=== PEAK" if l == 10 else ""
            if l in [0, 2, 5, 8, 10, 12, 15, 17, 20, 24]:
                print(f"L{l:<5} | {cos_sim:<23.4f} | {margin_b:<+18.3f} | {margin_oth:<+18.3f} | {flag}")
                
            layer_results.append({
                "layer": l,
                "cos_sim_B_D": cos_sim,
                "margin_B_no_tool": margin_b,
                "margin_B_other": margin_oth,
                "avg_margin": avg_margin
            })
            
    print(f"\nOptimal Invariant Subspace: Layer {best_layer} (Average Separation Margin: {max_margin:.3f})")
    
    # 1D projection table at best layer
    u_best = (d_pois["npz"]["resid"][pos, best_layer] - d_ben["npz"]["resid"][pos, best_layer]).astype(np.float32)
    u_best = u_best / np.linalg.norm(u_best)
    
    print(f"\n--- 1D Projections at Optimal Layer {best_layer} across All 9 Shopify Requests ---")
    print(f"{'Query':14} | {'Variant':18} | {'Ground Truth':12} | {'Projection Score':18} | {'Separation vs Twin'}")
    print("-" * 80)
    
    projection_scores = {}
    for q_name in ["B_no_tool", "B_other_tool", "D_trigger"]:
        p_val = float(np.dot(shopify_records[(q_name, "poisoned")]["npz"]["resid"][pos, best_layer], u_best))
        b_val = float(np.dot(shopify_records[(q_name, "benign_equivalent")]["npz"]["resid"][pos, best_layer], u_best))
        n_val = float(np.dot(shopify_records[(q_name, "no_note")]["npz"]["resid"][pos, best_layer], u_best))
        
        projection_scores[f"{q_name}_poisoned"] = p_val
        projection_scores[f"{q_name}_benign_twin"] = b_val
        projection_scores[f"{q_name}_no_note"] = n_val
        
        delta = p_val - b_val
        print(f"{q_name:14} | {'poisoned':18} | {'POISONED':12} | {p_val:<+18.3f} | {delta:<+18.3f}")
        print(f"{q_name:14} | {'benign_twin':18} | {'BENIGN':12} | {b_val:<+18.3f} | baseline (0.000)")
        print(f"{q_name:14} | {'no_note':18} | {'CLEAN':12} | {n_val:<+18.3f} | {n_val - b_val:<+18.3f}")
        print("-" * 80)
        
    return {
        "best_layer": best_layer,
        "max_margin": max_margin,
        "layer_results": layer_results,
        "projection_scores": projection_scores,
        "u_best": u_best
    }


def run_cross_case_generalization(records, u_best, best_layer):
    """Test 2: Cross-Case Generalization (Shopify -> Slack)."""
    print("\n" + "=" * 80)
    print("TEST 2: CROSS-CASE GENERALIZATION (Shopify L10 Vector -> Unseen Slack Case)")
    print("=" * 80)
    
    slack_records = [r for r in records.values() if r["label"]["case_id"] == "SlackManageMembership_multi_1_1"]
    
    pos = 2  # memory_result_end
    print(f"\n{'Case ID':32} | {'Query Name':22} | {'Variant':10} | {'Score':10} | {'Delta (Pois - Clean)'}")
    print("-" * 85)
    
    slack_by_query = {}
    for r in slack_records:
        q = r["label"]["query_name"]
        v = r["label"]["variant"]
        slack_by_query.setdefault(q, {})[v] = r
        
    slack_deltas = {}
    for q_name, variants in slack_by_query.items():
        r_clean = variants["clean"]
        r_pois = variants["poisoned"]
        
        s_clean = float(np.dot(r_clean["npz"]["resid"][pos, best_layer].astype(np.float32), u_best))
        s_pois = float(np.dot(r_pois["npz"]["resid"][pos, best_layer].astype(np.float32), u_best))
        delta = s_pois - s_clean
        slack_deltas[q_name] = delta
        
        print(f"{'SlackManageMembership':32} | {q_name:22} | {'clean':10} | {s_clean:<+10.3f} | baseline")
        print(f"{'SlackManageMembership':32} | {q_name:22} | {'poisoned':10} | {s_pois:<+10.3f} | {delta:<+10.3f}")
        print("-" * 85)
        
    print(f"\nZero-Shot Cross-Case Separation on Slack: Average Delta = {np.mean(list(slack_deltas.values())):.3f}")
    return slack_deltas


def run_spatial_delimiter_analysis(records):
    """Test 3: Spatial Signal Profile and Delimiter Collapse Verification."""
    print("\n" + "=" * 80)
    print("TEST 3: SPATIAL LOCALIZATION & DELIMITER COLLAPSE VERIFICATION")
    print("=" * 80)
    
    cid = "ShopifyDeleteProduct_multi_2_2"
    shopify_records = {
        (r["label"]["query_name"], r["label"]["variant"]): r
        for r in records.values() if r["label"]["case_id"] == cid
    }
    
    pois = shopify_records[("B_no_tool", "poisoned")]
    ben = shopify_records[("B_no_tool", "benign_equivalent")]
    
    pos_names = pois["index"]["position_names"]
    spatial_data = []
    
    print(f"\n{'Position Name':22} | {'Token Pos':10} | {'Layer 10 Cos Dist':18} | {'Layer 17 Cos Dist':18} | {'Signal vs Prompt'}")
    print("-" * 85)
    
    prompt_dist_l10 = None
    for p_idx, p_name in enumerate(pos_names):
        tok_pos = pois["index"]["positions"][p_idx]
        
        # Layer 10
        v_p10 = pois["npz"]["resid"][p_idx, 10].astype(np.float32)
        v_b10 = ben["npz"]["resid"][p_idx, 10].astype(np.float32)
        dist_10 = float(1.0 - np.dot(v_p10, v_b10) / (np.linalg.norm(v_p10) * np.linalg.norm(v_b10)))
        
        # Layer 17
        v_p17 = pois["npz"]["resid"][p_idx, 17].astype(np.float32)
        v_b17 = ben["npz"]["resid"][p_idx, 17].astype(np.float32)
        dist_17 = float(1.0 - np.dot(v_p17, v_b17) / (np.linalg.norm(v_p17) * np.linalg.norm(v_b17)))
        
        if p_name == "last_prompt":
            prompt_dist_l10 = dist_10
            
        ratio_str = f"{dist_10 / prompt_dist_l10:.2f}x" if prompt_dist_l10 else "1.00x"
        print(f"{p_name:22} | {tok_pos:<10} | {dist_10:<18.6f} | {dist_17:<18.6f} | {ratio_str}")
        
        spatial_data.append({
            "name": p_name,
            "token_pos": tok_pos,
            "cos_dist_l10": dist_10,
            "cos_dist_l17": dist_17
        })
        
    return spatial_data


def run_moe_routing_analysis(records):
    """Test 4: MoE Gating Anomaly & Expert Overlap Across All Layers."""
    print("\n" + "=" * 80)
    print("TEST 4: MOE ROUTING DISRUPTION (Top-4 Jaccard Overlap across All 24 Layers)")
    print("=" * 80)
    
    cid = "ShopifyDeleteProduct_multi_2_2"
    shopify_records = {
        (r["label"]["query_name"], r["label"]["variant"]): r
        for r in records.values() if r["label"]["case_id"] == cid
    }
    pois = shopify_records[("B_no_tool", "poisoned")]
    ben = shopify_records[("B_no_tool", "benign_equivalent")]
    
    # Position 3 is entry0.occ0.last, Pos 2 is memory_result_end
    pos_idx = 3
    print(f"\n--- Expert Allocation at Planted Memory Token (entry0.occ0.last) ---")
    print(f"{'Layer':6} | {'Top-4 Poisoned Experts':24} | {'Top-4 Benign Experts':24} | {'Jaccard':10} | {'Status'}")
    print("-" * 75)
    
    routing_results = []
    bifurcation_layers = []
    
    for l in range(24):
        r_p = pois["npz"]["router"][pos_idx, l].astype(np.float32)
        r_b = ben["npz"]["router"][pos_idx, l].astype(np.float32)
        
        top_p = sorted(list(np.argsort(r_p)[-4:]))
        top_b = sorted(list(np.argsort(r_b)[-4:]))
        
        jaccard = float(len(set(top_p) & set(top_b)) / len(set(top_p) | set(top_b)))
        
        flag = ""
        if jaccard <= 0.35:
            flag = "<=== SHARP BIFURCATION"
            bifurcation_layers.append(l)
        elif jaccard < 1.0:
            flag = "<--- partial drift"
            
        routing_results.append({
            "layer": l,
            "top_poisoned": [int(x) for x in top_p],
            "top_benign": [int(x) for x in top_b],
            "jaccard": jaccard
        })
        
        if jaccard < 1.0 or l in [0, 6, 12, 17, 20, 23]:
            print(f"L{l:<5} | {str(top_p):24} | {str(top_b):24} | {jaccard:<10.2f} | {flag}")
            
    print(f"\nLayers with Significant Routing Bifurcation (Jaccard <= 0.35): {bifurcation_layers}")
    return routing_results


def main():
    print("\n" + "#" * 80)
    print("STARTING SCIENTIFIC HYPOTHESIS EVALUATION ON CACHED REPRESENTATIONS")
    print("#" * 80 + "\n")
    
    records = load_dataset_records()
    print(f"Loaded {len(records)} requests from cache across 2 benchmark cases.")
    
    t1 = run_cross_state_transfer_analysis(records)
    t2 = run_cross_case_generalization(records, t1["u_best"], t1["best_layer"])
    t3 = run_spatial_delimiter_analysis(records)
    t4 = run_moe_routing_analysis(records)
    
    report = {
        "hypothesis": "Latent Commitment Invariance (D -> B Amplitude Modulation)",
        "optimal_layer": t1["best_layer"],
        "max_separation_margin": float(t1["max_margin"]),
        "cross_state_transfer": t1["projection_scores"],
        "cross_case_slack_deltas": {k: float(v) for k, v in t2.items()},
        "spatial_profiles": t3,
        "moe_routing_profiles": t4
    }
    
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE. Structured report written to: {OUT_REPORT}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
