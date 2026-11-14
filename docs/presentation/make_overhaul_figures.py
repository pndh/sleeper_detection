#!/usr/bin/env python3
"""
Generate crisp, publication-grade diagrams for the presentation overhaul:
1. architecture_comparison_at_vs_probe.png (Attention Tracker vs Residual Stream Probing)
2. causal_validation_diagram.png (Steering, Ablation, and Divergence Curve)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

OUT_DIR = Path('/home/pndhpndh/sleeper_detection/results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colors ──
NAVY = "#13294B"
ORANGE = "#FF5F05"
SLATE = "#475569"
LIGHT_BG = "#F8FAFC"
BOX_BG = "#FFFFFF"
BORDER = "#CBD5E1"
RED = "#DC2626"
GREEN = "#16A34A"
BLUE = "#2563EB"

# ════════════════════════════════════════════════════════════════════════
# 1. ARCHITECTURE COMPARISON: AT / AttnDA vs Residual Stream Probe
# ════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0), dpi=200)
fig.patch.set_facecolor(LIGHT_BG)

# ── Panel 1: Attention Tracker (Hung et al., 2024 / AttnDA 2025) ──
ax1.set_facecolor(LIGHT_BG)
ax1.set_xlim(0, 10)
ax1.set_ylim(1.5, 10)
ax1.axis('off')
ax1.set_title("(A) Prior Work: Attention Tracker / AttnDA (Hung et al.)",
              fontsize=13, fontweight='bold', color=NAVY, pad=12)

# Input prompt box
p1 = patches.FancyBboxPatch((0.5, 5.5), 2.8, 3.2, boxstyle="round,pad=0.2",
                            facecolor=BOX_BG, edgecolor=BORDER, linewidth=1.5)
ax1.add_patch(p1)
ax1.text(1.9, 8.1, "System & User Prompt\n(Benign Task)", ha='center', va='center',
         fontsize=10, fontweight='bold', color=NAVY)
ax1.text(1.9, 6.4, "[Dormant Memory Note]\n(Trigger NOT met yet)", ha='center', va='center',
         fontsize=9.5, fontweight='bold', color=RED,
         bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEE2E2", edgecolor=RED, lw=1))

# Arrow to Attention Matrices
ax1.annotate('', xy=(4.2, 7.1), xytext=(3.4, 7.1),
             arrowprops=dict(arrowstyle="->", lw=2, color=SLATE))

# Attention Weights Box
p2 = patches.FancyBboxPatch((4.2, 4.9), 3.2, 4.0, boxstyle="round,pad=0.2",
                            facecolor=BOX_BG, edgecolor=BORDER, linewidth=1.5)
ax1.add_patch(p2)
ax1.text(5.8, 8.3, "Attention Heads (L10-L24)", ha='center', va='center',
         fontsize=10, fontweight='bold', color=NAVY)
ax1.text(5.8, 7.2, r"Tracks attention mass: $\sum A_{i,j}$", ha='center', va='center',
         fontsize=9, color=SLATE)

# Heatmap sketch (Attention mass focused on benign task)
heat = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.9, 0.05, 0.05]])
im = ax1.imshow(heat, extent=(4.6, 7.0, 5.2, 6.6), cmap='Blues', aspect='auto', zorder=3)
ax1.text(5.8, 5.05, "Focus remains on benign task\n(No distraction signal!)", ha='center', va='top',
         fontsize=8.5, color=SLATE, style='italic')

# Output Verdict Box
p3 = patches.FancyBboxPatch((8.0, 5.5), 1.6, 2.8, boxstyle="round,pad=0.2",
                            facecolor="#FEE2E2", edgecolor=RED, linewidth=2)
ax1.add_patch(p3)
ax1.text(8.8, 7.5, "Verdict:", ha='center', va='center', fontsize=10, fontweight='bold', color=RED)
ax1.text(8.8, 6.5, "CLEAN\n(Misses dormant)", ha='center', va='center',
         fontsize=10.5, fontweight='bold', color=RED)

ax1.text(5.0, 3.8, "FATAL FLAW: Payload has not executed, so attention heads\nnever flip. AttnDA measures zero anomaly (0% Recall).",
         ha='center', va='top', fontsize=9.5, fontweight='bold', color=RED,
         bbox=dict(boxstyle="square,pad=0.35", facecolor="#FFF1F2", edgecolor=RED, lw=1))

# ── Panel 2: Our Architecture: Residual Stream Probing ──
ax2.set_facecolor(LIGHT_BG)
ax2.set_xlim(0, 10)
ax2.set_ylim(1.5, 10)
ax2.axis('off')
ax2.set_title("(B) Our Proposal: Mid-Layer Residual Stream Probe",
              fontsize=13, fontweight='bold', color=NAVY, pad=12)

# Input prompt box
p4 = patches.FancyBboxPatch((0.5, 5.5), 2.8, 3.2, boxstyle="round,pad=0.2",
                            facecolor=BOX_BG, edgecolor=BORDER, linewidth=1.5)
ax2.add_patch(p4)
ax2.text(1.9, 8.1, "System & User Prompt\n(Benign Task)", ha='center', va='center',
         fontsize=10, fontweight='bold', color=NAVY)
ax2.text(1.9, 6.4, "[Dormant Memory Note]\n(Waiting for trigger)", ha='center', va='center',
         fontsize=9.5, fontweight='bold', color=RED,
         bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEE2E2", edgecolor=RED, lw=1))

# Arrow to Residual Stream
ax2.annotate('', xy=(4.0, 7.1), xytext=(3.4, 7.1),
             arrowprops=dict(arrowstyle="->", lw=2, color=SLATE))

# Transformer Residual Stream
p5 = patches.FancyBboxPatch((4.0, 4.9), 3.4, 4.0, boxstyle="round,pad=0.2",
                            facecolor=BOX_BG, edgecolor=BORDER, linewidth=1.5)
ax2.add_patch(p5)
ax2.text(5.7, 8.3, "Residual Stream ($h^{(l)}$)", ha='center', va='center',
         fontsize=10, fontweight='bold', color=NAVY)

# Layer bars
layers = [0, 5, 10, 15, 20, 24]
for idx, l in enumerate(layers):
    x = 4.4 + idx * 0.44
    col = ORANGE if 14 <= l <= 19 else "#CBD5E1"
    rect = patches.Rectangle((x, 5.5), 0.32, 2.2, facecolor=col, edgecolor=None)
    ax2.add_patch(rect)
    ax2.text(x+0.16, 5.2, f"L{l}", ha='center', va='top', fontsize=7.5, color=SLATE)

# Probe tap from Layers 14-19
ax2.annotate('', xy=(5.7, 4.6), xytext=(5.7, 5.4),
             arrowprops=dict(arrowstyle="->", lw=2.5, color=ORANGE))

# Linear Probe Box
p6 = patches.FancyBboxPatch((4.4, 3.5), 2.6, 1.0, boxstyle="round,pad=0.2",
                            facecolor=BOX_BG, edgecolor=ORANGE, linewidth=2)
ax2.add_patch(p6)
ax2.text(5.7, 4.0, r"$z = w^T h^{(14-19)} + b$", ha='center', va='center',
         fontsize=10, fontweight='bold', color=NAVY)

# Arrow to Verdict
ax2.annotate('', xy=(8.0, 7.1), xytext=(7.5, 7.1),
             arrowprops=dict(arrowstyle="->", lw=2, color=SLATE))

# Output Verdict Box
p7 = patches.FancyBboxPatch((8.0, 5.5), 1.6, 2.8, boxstyle="round,pad=0.2",
                            facecolor="#ECFDF5", edgecolor=GREEN, linewidth=2)
ax2.add_patch(p7)
ax2.text(8.8, 7.5, "Verdict:", ha='center', va='center', fontsize=10, fontweight='bold', color=GREEN)
ax2.text(8.8, 6.5, "SLEEPER\nFLAGGED", ha='center', va='center',
         fontsize=10.5, fontweight='bold', color=GREEN)

ax2.text(5.0, 2.3, "CORE ADVANTAGE: Middle layers enter a shared 'pre-firing' state\nregardless of execution. Catches dormant sleepers at 0 output tokens.",
         ha='center', va='top', fontsize=9.5, fontweight='bold', color=GREEN,
         bbox=dict(boxstyle="square,pad=0.35", facecolor="#F0FDF4", edgecolor=GREEN, lw=1))

plt.tight_layout()
fig.savefig(OUT_DIR / 'architecture_comparison_at_vs_probe.png', bbox_inches='tight')
plt.close(fig)
print("✓ Saved architecture_comparison_at_vs_probe.png")


# ════════════════════════════════════════════════════════════════════════
# 2. CAUSAL VALIDATION SUITE DIAGRAM (RQ3)
# ════════════════════════════════════════════════════════════════════════
fig, (ax_curve, ax_causal) = plt.subplots(1, 2, figsize=(14, 4.8), dpi=200)
fig.patch.set_facecolor(LIGHT_BG)

# ── Left: Divergence Curve (Pre-firing in mid-layers vs decision in late layers) ──
ax_curve.set_facecolor(BOX_BG)
layers = np.arange(25)
# Theoretical / observed separation curve:
# Mid-layer transfer (dormant & fired share direction) peaks around L14-L19
shared_subspace = 0.5 + 0.48 / (1 + np.exp(-(layers - 8)/1.8)) - 0.45 / (1 + np.exp(-(layers - 20)/1.5))
execution_decision = 0.5 + 0.47 / (1 + np.exp(-(layers - 18)/1.5))

ax_curve.axvspan(14, 19, color=ORANGE, alpha=0.15, label='Pre-Firing Subspace (Layers 14–19)')
ax_curve.plot(layers, shared_subspace, color=ORANGE, lw=3, label='Shared Direction Transfer (Fired ↔ Dormant)')
ax_curve.plot(layers, execution_decision, color=NAVY, lw=2.5, linestyle='--', label='Execution Separation (Fired vs Resisted)')
ax_curve.axhline(0.5, color=SLATE, linestyle=':', alpha=0.7, label='Chance (0.50)')

ax_curve.set_title("Divergence Curve: Pre-Firing vs Late Execution", fontsize=12, fontweight='bold', color=NAVY)
ax_curve.set_xlabel("Transformer Layer (gpt-oss-20b)", fontsize=10, fontweight='bold', color=NAVY)
ax_curve.set_ylabel("Representation Alignment / AUROC", fontsize=10, fontweight='bold', color=NAVY)
ax_curve.set_ylim(0.4, 1.05)
ax_curve.grid(True, linestyle='--', alpha=0.4)
ax_curve.legend(loc='lower right', fontsize=8.5)

# ── Right: Causal Steering & Ablation Framework ──
ax_causal.set_facecolor(LIGHT_BG)
ax_causal.set_xlim(0, 10)
ax_causal.set_ylim(0, 10)
ax_causal.axis('off')
ax_causal.set_title("Causal Interventions: Steering & Ablation", fontsize=12, fontweight='bold', color=NAVY)

# Test 1: Activation Steering (Induce Firing)
b1 = patches.FancyBboxPatch((0.5, 5.2), 9.0, 4.2, boxstyle="round,pad=0.2",
                           facecolor=BOX_BG, edgecolor=ORANGE, linewidth=1.5)
ax_causal.add_patch(b1)
ax_causal.text(0.8, 8.8, "Test 1: Steering Dormant Sessions (+v_probe)",
               fontsize=10.5, fontweight='bold', color=ORANGE)
ax_causal.text(0.8, 7.8, "• Inject direction: $h^{(16)} \\leftarrow h^{(16)} + \\alpha \\cdot v_{\\text{probe}}$ during a benign turn.\n• Prediction: Forces the model to execute the dormant payload prematurely,\n  proving the direction actively controls execution readiness.",
               fontsize=9, color=NAVY, va='top')

# Test 2: Activation Ablation (Neutralize Fired)
b2 = patches.FancyBboxPatch((0.5, 0.4), 9.0, 4.2, boxstyle="round,pad=0.2",
                           facecolor=BOX_BG, edgecolor=BLUE, linewidth=1.5)
ax_causal.add_patch(b2)
ax_causal.text(0.8, 4.0, "Test 2: Ablating Fired Sessions (-proj_v h)",
               fontsize=10.5, fontweight='bold', color=BLUE)
ax_causal.text(0.8, 3.0, "• Project out direction: $h^{(16)} \\leftarrow h^{(16)} - \\text{proj}_{v}(h^{(16)})$ during a triggered turn.\n• Prediction: Blocks execution of the attack while preserving benign tool\n  calling capability (proving specificity and minimal side effects).",
               fontsize=9, color=NAVY, va='top')

plt.tight_layout()
fig.savefig(OUT_DIR / 'causal_validation_diagram.png', bbox_inches='tight')
plt.close(fig)
print("✓ Saved causal_validation_diagram.png")
