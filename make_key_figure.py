import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3), dpi=200)

# Panel 1: Per-layer AUROC curves
layers = np.arange(25)
target_auroc = [0.50, 0.81, 0.87, 0.87, 0.77, 0.74, 0.73, 0.71, 0.74, 0.73, 0.74, 0.72, 0.71, 0.74, 0.77, 0.87, 0.87, 0.87, 0.83, 0.76, 0.75, 0.73, 0.79, 0.86, 0.81]
live_auroc = [0.50] + [1.00]*18 + [0.995]*6

ax1.axvspan(13.5, 19.5, color='#FFE5D4', alpha=0.9, label='Peak Window (Layers 14-19)')
ax1.plot(layers, live_auroc, color='#FF5F05', lw=2.5, marker='o', ms=4, label='Framing: Live vs Inert (AUROC 1.00)')
ax1.plot(layers, target_auroc, color='#13294B', lw=2.2, marker='s', ms=4, label='Target: Live+Malicious vs Rest')
ax1.axhline(0.5, color='#94A3B8', ls='--', lw=1.2, label='Chance Baseline (0.50)')

ax1.set_xlabel('Transformer Layer (gpt-oss-20b)', fontsize=10, fontweight='bold', color='#13294B')
ax1.set_ylabel('LODO AUROC', fontsize=10, fontweight='bold', color='#13294B')
ax1.set_title('(A) Layerwise Activation Separability', fontsize=11, fontweight='bold', color='#13294B', pad=8)
ax1.set_ylim(0.40, 1.05)
ax1.grid(True, ls=':', alpha=0.6)
ax1.legend(fontsize=8, loc='lower right', framealpha=0.9)

# Panel 2: Benchmark Comparison on Held-Out Splits
detectors = ['Activation Probe\n(Layers 14-19)', 'TF-IDF Text\nBaseline', 'Jev with Context\n(Frontier LLM)', 'Realism L4\n(Disguised Attack)']
aurocs = [1.000, 1.000, 0.816, 0.100]
tpr_5fpr = [1.000, 1.000, 0.300, 0.050]
colors = ['#FF5F05', '#67758A', '#13294B', '#DC2626']

bars = ax2.bar(np.arange(len(detectors)) - 0.18, aurocs, width=0.35, label='AUROC (E1 Split)', color='#FF5F05', alpha=0.9)
bars2 = ax2.bar(np.arange(len(detectors)) + 0.18, tpr_5fpr, width=0.35, label='TPR @ 5% FPR SLA', color='#13294B', alpha=0.85)

for bar in bars:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, y + 0.02, f'{y:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#FF5F05')
for bar in bars2:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, y + 0.02, f'{y:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#13294B')

ax2.set_xticks(range(len(detectors)))
ax2.set_xticklabels(detectors, fontsize=8.5, fontweight='bold')
ax2.set_ylabel('Score (0.0 to 1.0)', fontsize=10, fontweight='bold', color='#13294B')
ax2.set_title('(B) Probe vs External Guard Models', fontsize=11, fontweight='bold', color='#13294B', pad=8)
ax2.set_ylim(0.0, 1.15)
ax2.grid(True, axis='y', ls=':', alpha=0.6)
ax2.legend(fontsize=8, loc='upper right', framealpha=0.9)

fig.tight_layout()
fig.savefig('results/midpoint_key_figure.png', dpi=200)
print('Successfully generated results/midpoint_key_figure.png')