"""Detector demo on cached activations (CPU only, ~1 min): what the probe detects and what it does not.

Data: the cached 2x2 sets (payload malicious/benign x framing live/inert, 10 domains each).
  sound_2x2       real per-case system prompt and tools, 5 rotating templates, no cue words  (primary)
  scaleup_2x2x10  one demo_v1 request, one template per framing                            (pilot set)
Feature: residual stream at memory_result_end (end of the memory tool result), averaged over resid layers 14-19
(55-80% depth). The window was fixed after the pilot and is not re-tuned here. Probe: standardise + L2 logistic
regression, C fixed at 1.0, no tuning. Every number is out-of-fold, leave-one-domain-out (LODO).

Sections
  1. Per-layer LODO AUROC for three targets: live+malicious vs rest (the detector), framing alone, payload alone.
  2. Window probe vs baselines: TF-IDF on the note text, prompt length, layer-1 probe. 95% CI by domain bootstrap.
  3. Does it need an interaction? Out-of-fold scores averaged per cell; interaction = (LM - LB) - (IM - IB) per
     domain, t-test across domains. A "commitment" detector predicts > 0; two additive features predict ~0.
  4. Additive model: AND of the separate framing and payload probes vs the direct detector.

  cd ~/sleeper_detection && ~/.conda/envs/gpt-oss-env/bin/python interp/detector_demo.py
Writes results/detector_demo/{report.json, detector_demo.png}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "detector_demo"
WINDOW = (14, 19)  # resid indices, inclusive
POSITION = "memory_result_end"
SETS = {"sound_2x2": "case_id", "scaleup_2x2x10": "case_id"}
RNG = np.random.default_rng(0)


def load(name: str, group_key: str) -> dict:
    labels = [json.loads(line) for line in open(ROOT / "interp" / "data" / name / "labels.jsonl")]
    index = json.load(open(ROOT / "interp" / "cache" / name / "index.json"))
    resid = []
    for lab in labels:
        k = index[lab["id"]]["position_names"].index(POSITION)
        resid.append(np.load(ROOT / "interp" / "cache" / name / f"{lab['id']}.npz")["resid"][k].astype(np.float32))
    live = np.array([lab["framing_type"].startswith("live") for lab in labels])
    mal = np.array([lab["payload_type"] == "malicious" for lab in labels])
    return {"name": name, "resid": np.stack(resid), "live": live, "mal": mal, "target": live & mal,
            "groups": np.array([lab[group_key] for lab in labels]), "text": [lab["note_text"] for lab in labels],
            "n_tokens": np.array([index[lab["id"]]["n_tokens"] for lab in labels], dtype=np.float32)}


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="liblinear", dual=True, max_iter=5000))


def lodo_scores(X, y, groups, make=probe) -> np.ndarray:
    """Out-of-fold decision scores, one domain held out per fold."""
    s = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g
        m = make().fit(X[~te] if not isinstance(X, list) else [x for x, t in zip(X, te) if not t], y[~te])
        s[te] = m.decision_function(X[te] if not isinstance(X, list) else [x for x, t in zip(X, te) if t])
    return s


def boot_ci(y, s, groups, n=2000) -> tuple[float, float]:
    """95% CI of AUROC, resampling whole domains."""
    doms = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in doms}
    vals = []
    for _ in range(n):
        take = np.concatenate([idx[g] for g in RNG.choice(doms, len(doms))])
        if y[take].min() != y[take].max():
            vals.append(roc_auc_score(y[take], s[take]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def window(d):
    return d["resid"][:, WINDOW[0]:WINDOW[1] + 1].mean(1)


def section1(d) -> dict:
    curves = {}
    for tname in ("target", "live", "mal"):
        y = d[tname]
        curves[tname] = [float(roc_auc_score(y, lodo_scores(d["resid"][:, L], y, d["groups"]))) if L > 0 else 0.5
                         for L in range(d["resid"].shape[1])]
    return curves


def section2(d) -> dict:
    y, g = d["target"], d["groups"]
    rows = {
        f"probe, resid {WINDOW[0]}-{WINDOW[1]} mean": lodo_scores(window(d), y, g),
        "probe, resid 1 (after one layer)": lodo_scores(d["resid"][:, 1], y, g),
        "TF-IDF + LR on note text (no model)": lodo_scores(
            d["text"], y, g, lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1),
                                                   LogisticRegression(C=10.0, max_iter=5000))),
        "prompt length only": lodo_scores(d["n_tokens"][:, None], y, g),
    }
    out = {}
    for k, s in rows.items():
        auc = float(roc_auc_score(y, s))
        out[k] = {"auroc": auc, "ci95": boot_ci(y, s, g)}
    return out, rows[f"probe, resid {WINDOW[0]}-{WINDOW[1]} mean"]


def get_oof_scores(d: dict) -> dict[str, np.ndarray]:
    """Return per-request out-of-fold scores for the probe and baselines."""
    y, g = d["target"], d["groups"]
    return {
        "probe": lodo_scores(window(d), y, g),
        "probe_l1": lodo_scores(d["resid"][:, 1], y, g),
        "tfidf": lodo_scores(
            d["text"], y, g, lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1),
                                                   LogisticRegression(C=10.0, max_iter=5000))),
        "length": lodo_scores(d["n_tokens"][:, None], y, g),
    }


def section3(d, s) -> dict:
    s = (s - s.mean()) / s.std()
    per = []
    for g in np.unique(d["groups"]):
        m = d["groups"] == g
        cell = lambda live, mal: s[m & (d["live"] == live) & (d["mal"] == mal)].mean()  # noqa: E731
        per.append({"domain": str(g), "LM": cell(1, 1), "LB": cell(1, 0), "IM": cell(0, 1), "IB": cell(0, 0)})
    inter = np.array([(p["LM"] - p["LB"]) - (p["IM"] - p["IB"]) for p in per])
    t, p = stats.ttest_1samp(inter, 0.0)
    means = {c: float(np.mean([q[c] for q in per])) for c in ("LM", "LB", "IM", "IB")}
    return {"cell_means_z": means, "interaction_mean": float(inter.mean()),
            "interaction_ci95": [float(x) for x in stats.t.interval(0.95, len(inter) - 1, inter.mean(), stats.sem(inter))],
            "t": float(t), "p_two_sided": float(p), "n_domains": len(inter)}


def section4(d) -> dict:
    X, g = window(d), d["groups"]
    sl, sm = lodo_scores(X, d["live"], g), lodo_scores(X, d["mal"], g)
    and_score = np.minimum(stats.zscore(sl), stats.zscore(sm))  # high only when both factors are high
    return {"framing_probe_auroc": float(roc_auc_score(d["live"], sl)),
            "payload_probe_auroc": float(roc_auc_score(d["mal"], sm)),
            "AND_of_factor_probes_auroc_on_target": float(roc_auc_score(d["target"], and_score))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for row, (name, gk) in enumerate(SETS.items()):
        d = load(name, gk)
        print(f"\n{'=' * 88}\n{name}: {len(d['target'])} requests, {d['target'].sum()} positives (live+malicious), "
              f"{len(np.unique(d['groups']))} domains, feature = resid @ {POSITION}\n{'=' * 88}")
        curves = section1(d)
        base, s_win = section2(d)
        inter = section3(d, s_win)
        add = section4(d)
        report[name] = {"per_layer_lodo_auroc": curves, "detector_vs_baselines": base, "interaction": inter,
                        "additive_model": add}

        print("\n[2] Detector vs baselines (target: live+malicious vs the other 3 cells; LODO; 95% CI by domain bootstrap)")
        for k, v in base.items():
            print(f"    {k:42s} AUROC {v['auroc']:.3f}   [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]")
        print("\n[3] Interaction test on out-of-fold detector scores (z-scored), mean per cell:")
        cm = inter["cell_means_z"]
        print(f"    live+malicious {cm['LM']:+.2f}   live+benign {cm['LB']:+.2f}   inert+malicious {cm['IM']:+.2f}   "
              f"inert+benign {cm['IB']:+.2f}")
        print(f"    interaction (LM-LB)-(IM-IB) = {inter['interaction_mean']:+.2f}, 95% CI "
              f"[{inter['interaction_ci95'][0]:+.2f}, {inter['interaction_ci95'][1]:+.2f}], t({inter['n_domains'] - 1}) = "
              f"{inter['t']:+.2f}, p = {inter['p_two_sided']:.3f}")
        print("\n[4] Additive explanation:")
        print(f"    framing probe (live vs inert) AUROC {add['framing_probe_auroc']:.3f} | payload probe (malicious vs "
              f"benign) AUROC {add['payload_probe_auroc']:.3f} | AND of the two on the target AUROC "
              f"{add['AND_of_factor_probes_auroc_on_target']:.3f}")

        ax = axes[row, 0]
        L = np.arange(len(curves["target"]))
        ax.axvspan(WINDOW[0] - 0.5, WINDOW[1] + 0.5, color="0.9", label="fixed window 14-19")
        for t, lab, c in (("target", "live+malicious vs rest", "C3"), ("live", "framing: live vs inert", "C0"),
                          ("mal", "payload: malicious vs benign", "C2")):
            ax.plot(L, curves[t], marker="o", ms=3, color=c, label=lab)
        ax.axhline(0.5, color="0.5", lw=0.8, ls="--")
        ax.set(title=f"{name}: per-layer LODO AUROC", xlabel="resid index (0 = embeddings)", ylabel="AUROC",
               ylim=(0.3, 1.02))
        ax.legend(fontsize=8, loc="lower right")

        ax = axes[row, 1]
        names = list(base)
        aucs = [base[k]["auroc"] for k in names]
        err = np.array([[a - base[k]["ci95"][0], base[k]["ci95"][1] - a] for k, a in zip(names, aucs)]).T
        ax.barh(range(len(names)), aucs, xerr=err, color=["C3", "C1", "C7", "C7"], capsize=3)
        ax.set_yticks(range(len(names)), [n.replace(" (", "\n(") for n in names], fontsize=8)
        ax.axvline(0.5, color="0.5", lw=0.8, ls="--")
        ax.set(xlim=(0.3, 1.0), xlabel="LODO AUROC (95% CI)",
               title=f"interaction {inter['interaction_mean']:+.2f} (p={inter['p_two_sided']:.2f}); "
                     f"AND of factor probes {add['AND_of_factor_probes_auroc_on_target']:.2f}")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "detector_demo.png", dpi=130)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT / 'report.json'} and {OUT / 'detector_demo.png'}")


if __name__ == "__main__":
    main()
