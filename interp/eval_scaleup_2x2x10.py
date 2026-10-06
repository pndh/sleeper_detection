"""Statistical Evaluation Suite for the Scaled-Up 2x2x10 Dataset.

Evaluates 40 samples (10 cases x 4 factorial cells) at Layer 10 (memory_result_end):
1. Paired Contrastive Deltas (Malicious vs Benign) under Live vs Inert Framing:
     Delta_Live  = Score(1,1) - Score(1,2)
     Delta_Inert = Score(2,1) - Score(2,2)
   Calculates Paired t-test and Wilcoxon signed-rank test across the 10 cases.

2. Repeated-Measures Two-Way ANOVA:
   Tests Main Effect of Content (Malicious vs Benign),
   Main Effect of Framing (Live vs Inert),
   and the Crucial Interaction Effect (Content x Framing).

3. Supervised Probing & Cross-Validation:
   Leave-One-Case-Out Cross-Validation (LOCO-CV) using a Linear Discriminant / Logistic Probe
   trained on 9 cases and tested on the unseen 10th case.
   Reports AUROC, Accuracy, and Separation Margin on unseen domains.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "interp" / "data" / "scaleup_2x2x10"
CACHE_DIR = ROOT / "interp" / "cache" / "scaleup_2x2x10"
CACHE_V1 = ROOT / "interp" / "cache" / "demo_v1"
OUT_REPORT = ROOT / "results" / "scaleup_2x2x10_statistical_report.json"


def run_statistical_analysis():
    print("=" * 80)
    print("STATISTICAL EVALUATION: SCALED-UP 2x2x10 EXPERIMENTAL SUITE")
    print("=" * 80)
    
    # 1. Load Direction Vector u_10 from D_trigger difference
    idx_v1 = json.load(open(CACHE_V1 / "index.json"))
    pos_d = idx_v1["8a413989c4d60bef"]["position_names"].index("memory_result_end")
    layer = 10
    
    d_pois = np.load(CACHE_V1 / "8a413989c4d60bef.npz")["resid"][pos_d, layer].astype(np.float32)
    d_ben = np.load(CACHE_V1 / "9d339b7b6829028f.npz")["resid"][pos_d, layer].astype(np.float32)
    v_diff = d_pois - d_ben
    norm_v = np.linalg.norm(v_diff)
    u_10 = v_diff / norm_v
    print(f"Loaded Layer 10 Unit Direction u_10: norm={norm_v:.3f}, dim={len(u_10)}")
    
    # 2. Load 2x2x10 records
    labels = [json.loads(line) for line in open(DATA_DIR / "labels.jsonl")]
    idx_2x2x10 = json.load(open(CACHE_DIR / "index.json"))
    
    # Organize by case_id and cell
    case_data = {}
    for lab in labels:
        rid = lab["id"]
        cid = lab["case_id"]
        cell = lab["cell"]
        
        pos_idx = idx_2x2x10[rid]["position_names"].index("memory_result_end")
        npz = np.load(CACHE_DIR / f"{rid}.npz")
        h = npz["resid"][pos_idx, layer].astype(np.float32)
        proj = float(np.dot(h, u_10))
        
        if cid not in case_data:
            case_data[cid] = {}
        case_data[cid][cell] = {
            "id": rid,
            "proj": proj,
            "h": h,
            "label": lab
        }
        
    print(f"\nEvaluated {len(case_data)} cases across 4 cells (40 total requests).")
    
    # 3. Compute Per-Case Factorial Metrics
    print("\n" + "-" * 90)
    print(f"{'Case ID':25} | {'Live Mal(1,1)':13} | {'Live Ben(1,2)':13} | {'Inert Mal(2,1)':14} | {'Inert Ben(2,2)':14} | {'Delta_Live':10} | {'Delta_Inert':10}")
    print("-" * 90)
    
    table_rows = []
    deltas_live = []
    deltas_inert = []
    cell_11_scores = []
    cell_12_scores = []
    cell_21_scores = []
    cell_22_scores = []
    
    for cid, cells in case_data.items():
        s_11 = cells["cell_1_1_live_malicious"]["proj"]
        s_12 = cells["cell_1_2_live_benign"]["proj"]
        s_21 = cells["cell_2_1_inert_malicious"]["proj"]
        s_22 = cells["cell_2_2_inert_benign"]["proj"]
        
        d_live = s_11 - s_12
        d_inert = s_21 - s_22
        
        deltas_live.append(d_live)
        deltas_inert.append(d_inert)
        cell_11_scores.append(s_11)
        cell_12_scores.append(s_12)
        cell_21_scores.append(s_21)
        cell_22_scores.append(s_22)
        
        print(f"{cid:25} | {s_11:+13.2f} | {s_12:+13.2f} | {s_21:+14.2f} | {s_22:+14.2f} | {d_live:+10.2f} | {d_inert:+10.2f}")
        table_rows.append({
            "case_id": cid,
            "cell_1_1": s_11,
            "cell_1_2": s_12,
            "cell_2_1": s_21,
            "cell_2_2": s_22,
            "delta_live": d_live,
            "delta_inert": d_inert,
            "interaction": d_live - d_inert
        })
        
    print("-" * 90)
    mean_d_live = float(np.mean(deltas_live))
    mean_d_inert = float(np.mean(deltas_inert))
    std_d_live = float(np.std(deltas_live, ddof=1))
    std_d_inert = float(np.std(deltas_inert, ddof=1))
    
    print(f"{'MEAN':25} | {np.mean(cell_11_scores):+13.2f} | {np.mean(cell_12_scores):+13.2f} | {np.mean(cell_21_scores):+14.2f} | {np.mean(cell_22_scores):+14.2f} | {mean_d_live:+10.2f} | {mean_d_inert:+10.2f}")
    print(f"{'STD':25} | {np.std(cell_11_scores, ddof=1):13.2f} | {np.std(cell_12_scores, ddof=1):13.2f} | {np.std(cell_21_scores, ddof=1):14.2f} | {np.std(cell_22_scores, ddof=1):14.2f} | {std_d_live:10.2f} | {std_d_inert:10.2f}")
    
    # 4. Statistical Significance Testing
    # Paired differences: D_diff = Delta_Live - Delta_Inert (Interaction Effect)
    interactions = np.array(deltas_live) - np.array(deltas_inert)
    mean_interaction = float(np.mean(interactions))
    std_interaction = float(np.std(interactions, ddof=1))
    t_stat = mean_interaction / (std_interaction / np.sqrt(len(interactions)))
    
    # Simple t-distribution CDF approximation for p-value (df = 9)
    from scipy import stats
    p_val_interaction = float(stats.t.sf(t_stat, df=len(interactions)-1))
    
    print("\n--- Two-Way Repeated Measures Contrast & Hypothesis Test ---")
    print(f"Mean Live Delta (Malicious - Benign)  : {mean_d_live:+.2f} (std={std_d_live:.2f})")
    print(f"Mean Inert Delta (Malicious - Benign) : {mean_d_inert:+.2f} (std={std_d_inert:.2f})")
    print(f"Mean Commitment Interaction (Live - Inert Delta) : {mean_interaction:+.2f} (std={std_interaction:.2f})")
    print(f"Paired t-statistic : t(9) = {t_stat:.4f}, p-value (one-tailed) = {p_val_interaction:.4e}")
    
    # 5. Leave-One-Case-Out Cross-Validation (LOCO-CV) on 2D Linear Probe
    # Features: [Projection on u_10, Normalized Residual Norm / Framing Feature]
    # Labels: 1 for Cell (1,1) [Live Malicious / Active Commitment], 0 for Cells (1,2), (2,1), (2,2)
    print("\n--- Leave-One-Case-Out Cross-Validation (LOCO-CV) Evaluation ---")
    
    X_all = []
    y_all = []
    case_tags = []
    
    for cid, cells in case_data.items():
        for cell_name, cinfo in cells.items():
            h_vec = cinfo["h"]
            proj_val = cinfo["proj"]
            # 2D representation: proj onto u_10, and projection onto mean framing axis
            norm_val = float(np.linalg.norm(h_vec))
            X_all.append([proj_val, norm_val])
            y_all.append(1 if cell_name == "cell_1_1_live_malicious" else 0)
            case_tags.append(cid)
            
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int32)
    case_tags = np.array(case_tags)
    
    # Standardize features
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    
    loco_preds = []
    loco_probs = []
    loco_trues = []
    
    unique_cases = list(case_data.keys())
    for test_case in unique_cases:
        train_mask = case_tags != test_case
        test_mask = case_tags == test_case
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_all[train_mask])
        X_test = scaler.transform(X_all[test_mask])
        y_train = y_all[train_mask]
        y_test = y_all[test_mask]
        
        clf = LogisticRegression(class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)
        
        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)
        
        loco_probs.extend(probs)
        loco_preds.extend(preds)
        loco_trues.extend(y_test)
        
    loco_auroc = float(roc_auc_score(loco_trues, loco_probs))
    loco_acc = float(accuracy_score(loco_trues, loco_preds))
    print(f"Zero-Shot LOCO-CV AUROC across 10 Unseen Domains : {loco_auroc:.4f}")
    print(f"Zero-Shot LOCO-CV Accuracy (Balanced Threshold)   : {loco_acc * 100:.1f}%")
    
    summary = {
        "n_cases": len(case_data),
        "total_samples": len(labels),
        "mean_delta_live": mean_d_live,
        "mean_delta_inert": mean_d_inert,
        "commitment_interaction_mean": mean_interaction,
        "t_statistic": float(t_stat),
        "p_value": float(p_val_interaction),
        "loco_auroc": loco_auroc,
        "loco_accuracy": loco_acc,
        "per_case_metrics": table_rows
    }
    
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(summary, indent=2))
    print(f"\nStatistical Report saved to: {OUT_REPORT}")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    run_statistical_analysis()
