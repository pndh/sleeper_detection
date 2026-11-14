#!/usr/bin/env python3
"""
Build the overhauled AIA presentation deck focusing on:
  - Baseline comparison: Attention Tracker (AT) & AttnDA vs Residual Stream Probing
  - Architectural breakdown of why attention fails on dormant/reasoning attacks
  - Future baseline overhaul & full evaluation benchmark
  - Causal experiments: steering, ablation, divergence curve, and benign twin controls
  - Cross-state transfer (Train on Fired, Find Dormant)
"""
import os
import shutil
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Paths ──
OUT_DIR = Path('/home/pndhpndh/sleeper_detection')
RESULTS = Path('/home/pndhpndh/sleeper_detection/results')
ARTIFACT_DIR = Path('/home/pndhpndh/.gemini/antigravity/brain/ae642783-e859-4834-9387-38d23eaa5676')

# Figures
ARCH_FIG = RESULTS / 'architecture_comparison_at_vs_probe.png'
CAUSAL_FIG = RESULTS / 'causal_validation_diagram.png'
REALISM_LADDER = RESULTS / 'realism_ladder_figure.png'
KEY_FIGURE = RESULTS / 'midpoint_key_figure.png'

# ── Colors (AIA brand) ──
ORANGE = RGBColor(0xFF, 0x5F, 0x05)
NAVY = RGBColor(0x13, 0x29, 0x4B)
SLATE = RGBColor(0x47, 0x55, 0x69)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xC6, 0xC7, 0xC6)
GREEN_DARK = RGBColor(0x00, 0x66, 0x33)

def add_textbox(slide, left, top, width, height, text, font_size=18,
                color=NAVY, bold=False, font_name="Arial",
                alignment=PP_ALIGN.LEFT, word_wrap=True,
                line_spacing=1.2, space_after=4):
    """Add a clean text box with proper formatting."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = word_wrap

    if isinstance(text, str):
        lines = [text]
    else:
        lines = list(text)

    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        p.alignment = alignment
        for r in p.runs:
            r.font.name = font_name
            r.font.size = Pt(font_size)
            r.font.bold = bold
            r.font.color.rgb = color
    return txBox

def add_section_header(slide, label, left=Inches(0.67), top=Inches(2.5)):
    """Add an orange section header label."""
    add_textbox(slide, left, top, Inches(6), Inches(0.35),
                label, font_size=11, color=ORANGE, bold=True,
                font_name="Courier New", space_after=2)

def add_slide_number(slide, num, total=14):
    """Add slide number in bottom right."""
    add_textbox(slide, Inches(12.3), Inches(7.05), Inches(0.8), Inches(0.35),
                f"{num:02d}", font_size=11, color=ORANGE, bold=True,
                font_name="Courier New", alignment=PP_ALIGN.RIGHT)

def add_footer(slide, text="RESEARCH PROPOSAL  /  DORMANT SLEEPER DETECTION"):
    """Add footer bar at bottom."""
    add_textbox(slide, Inches(0.67), Inches(7.05), Inches(10), Inches(0.35),
                text, font_size=10, color=SLATE,
                font_name="Courier New")

def add_org_label(slide):
    """Add 'AI ALIGNMENT @ ILLINOIS' top label."""
    add_textbox(slide, Inches(0.67), Inches(0.35), Inches(4), Inches(0.3),
                "AI ALIGNMENT @ ILLINOIS", font_size=11, color=NAVY,
                bold=True, font_name="Courier New")

def add_category_label(slide, label):
    """Add category label below org."""
    add_textbox(slide, Inches(0.67), Inches(0.72), Inches(5), Inches(0.3),
                label, font_size=11, color=ORANGE,
                bold=True, font_name="Courier New")

def add_title(slide, title, top=Inches(1.1), size=32):
    """Add large navy title."""
    add_textbox(slide, Inches(0.67), top, Inches(12.0), Inches(0.9),
                title, font_size=size, color=NAVY, bold=True,
                space_after=4)

# ════════════════════════════════════════════════════════════════
#  BUILD PRESENTATION
# ════════════════════════════════════════════════════════════════
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]

# ═══════════════ SLIDE 1: Title (Navy background) ═══════════════
s1 = prs.slides.add_slide(blank)
bg = s1.background
fill = bg.fill
fill.solid()
fill.fore_color.rgb = NAVY

add_textbox(s1, Inches(0.67), Inches(0.35), Inches(4), Inches(0.3),
            "AI ALIGNMENT @ ILLINOIS", font_size=11, color=WHITE,
            bold=True, font_name="Courier New")
add_textbox(s1, Inches(0.67), Inches(0.85), Inches(6), Inches(0.3),
            "RESEARCH FELLOWSHIP  /  FALL 2026", font_size=13, color=ORANGE,
            bold=True, font_name="Courier New")

add_textbox(s1, Inches(0.67), Inches(1.8), Inches(11.5), Inches(1.2),
            "Beyond Attention Tracking", font_size=46, color=WHITE, bold=True)
add_textbox(s1, Inches(0.67), Inches(3.2), Inches(11.5), Inches(1.2),
            "Detecting Dormant Sleeper Payloads in LLM Agent Memory\nvia Mid-Layer Residual Geometry",
            font_size=26, color=ORANGE, bold=True, space_after=8)

add_textbox(s1, Inches(0.67), Inches(5.3), Inches(11.5), Inches(0.8), [
    "Raghav Agarwal (raghava7@illinois.edu)  ·  Huy Pham (huynp2@illinois.edu)",
    "Mentors: Enyi Jiang, Himnish Hunma  ·  November 2026"
], font_size=17, color=WHITE, space_after=5)

add_textbox(s1, Inches(0.67), Inches(7.05), Inches(10), Inches(0.35),
            "RESEARCH PROPOSAL  /  DORMANT SLEEPER DETECTION",
            font_size=10, color=LIGHT_GRAY, font_name="Courier New")
add_slide_number(s1, 1)


# ═══════════════ SLIDE 2: The Core Problem & The Attention Blindspot ═══════════════
s2 = prs.slides.add_slide(blank)
add_org_label(s2)
add_category_label(s2, "MOTIVATION / THE THREAT MODEL")
add_title(s2, "The Dormant Threat & The Attention Blindspot", size=32)

add_section_header(s2, "01 / THE DORMANT MEMORY THREAT", top=Inches(2.3))
add_textbox(s2, Inches(0.67), Inches(2.65), Inches(5.6), Inches(2.3), [
    "• A poisoned note sits silently in agent memory stores (scratchpad, skills, long-term memory) waiting for an environmental trigger.",
    "• In routine benign turns, the agent fulfills innocent tasks (e.g., 'Summarize emails'). The payload lies completely dormant.",
    "• Text detectors fail on polite, descriptive phrasing: an LLM judge caught 85% of directive payloads but only 6% of descriptive ones (Ruan et al., 2026)."
], font_size=13.5, color=NAVY, space_after=4)

add_section_header(s2, "02 / WHY ATTENTION TRACKERS FAIL", left=Inches(6.8), top=Inches(2.3))
add_textbox(s2, Inches(6.8), Inches(2.65), Inches(5.8), Inches(2.3), [
    "• Attention Tracker (Hung et al., 2024) and AttnDA (Hung et al., 2025) detect prompt injection by measuring the distraction effect—attention mass shifting from prompt to attack.",
    "• Fatal Blindspot: Because a dormant payload hasn't fired yet, attention heads never flip to attack tokens. AT and AttnDA measure zero distraction anomaly (0% recall).",
    "• Goal: One unified detector that works whether the payload has fired or remains dormant."
], font_size=13.5, color=NAVY, space_after=4)

add_textbox(s2, Inches(0.67), Inches(5.85), Inches(12.0), Inches(0.8),
    "KEY INSIGHT  /  Dormant and fired sessions share a representation in the middle layers (Layers 14–19). "
    "The model enters a 'pre-firing' state either way; whether to act is decided only in late layers.",
    font_size=15, color=NAVY, bold=True, space_after=3)

add_footer(s2)
add_slide_number(s2, 2)


# ═══════════════ SLIDE 3: Architecture Comparison (BIG FIGURE) ═══════════════
s3 = prs.slides.add_slide(blank)
add_org_label(s3)
add_category_label(s3, "ARCHITECTURE COMPARISON")
add_title(s3, "Architecture: Attention Tracking vs. Residual Probing", size=30)

if ARCH_FIG.exists():
    s3.shapes.add_picture(str(ARCH_FIG),
                          Inches(0.67), Inches(2.2),
                          Inches(12.0), Inches(4.3))

add_footer(s3)
add_slide_number(s3, 3)


# ═══════════════ SLIDE 4: Current Baseline vs Overhaul Need ═══════════════
s4 = prs.slides.add_slide(blank)
add_org_label(s4)
add_category_label(s4, "BASELINE OVERHAUL")
add_title(s4, "Current Status vs. Where We Need to Be", size=32)

add_section_header(s4, "CURRENT STATUS (MIDPOINT EXPLORATION)", top=Inches(2.3))
add_textbox(s4, Inches(0.67), Inches(2.65), Inches(5.6), Inches(2.3), [
    "• Implemented L2 logistic regression on gpt-oss-20b residual stream at memory_result_end.",
    "• Validated against simple text baselines: TF-IDF (1.00 on live/inert) and external Jev API (0.47 AUROC on fired).",
    "• Preliminary signal: 0.876 AUROC predicting execution within template (+0.10 over note text).",
    "• Limitation: Comparing against TF-IDF does not prove our method beats modern white-box injection detectors."
], font_size=13.5, color=NAVY, space_after=4)

add_section_header(s4, "THE OVERHAUL: DIRECT PEER-REVIEWED COMPARISON", left=Inches(6.8), top=Inches(2.3))
add_textbox(s4, Inches(6.8), Inches(2.65), Inches(5.8), Inches(2.3), [
    "• Direct Head-to-Head: Implement Attention Tracker (AT) and AttnDA on the exact same gpt-oss-20b forward passes.",
    "• Test on Dormant Payloads: Explicitly test AT/AttnDA on dormant turns (Turn B) vs fired turns (Turn D) to confirm their failure on sleeper threats.",
    "• Mechanistic Dissection: Compare attention head distraction vs residual stream linear separability side-by-side.",
    "• Shift Focus: Refocus from 'will-fire' prediction to robust dormant payload detection."
], font_size=13.5, color=NAVY, space_after=4)

add_textbox(s4, Inches(0.67), Inches(5.85), Inches(12.0), Inches(0.8),
    "RESEARCH PIVOT  /  Rather than competing with text models on surface phrasing, we directly "
    "challenge white-box attention baselines (AT / AttnDA) by uncovering their dormant blindspot.",
    font_size=15, color=NAVY, bold=True, space_after=3)

add_footer(s4)
add_slide_number(s4, 4)


# ═══════════════ SLIDE 5: Formal Research Questions (RQ1 - RQ4) ═══════════════
s5 = prs.slides.add_slide(blank)
add_org_label(s5)
add_category_label(s5, "RESEARCH QUESTIONS")
add_title(s5, "Core Research Questions (RQ1 – RQ4)", size=32)

rqs = [
    ("RQ1", "Do AT and AttnDA fail on dormant payloads?",
     "Test peer-reviewed attention-based injection detectors on dormant agent turns. Hypothesis: Since the model does not execute the payload, attention heads do not flip, yielding near-chance detection (AUROC ≈ 0.50)."),
    ("RQ2", "Does a mid-layer probe beat them on both dormant and fired sessions?",
     "Train an L2 logistic probe on residual stream states across Layers 0–24. Hypothesis: Probe achieves >0.95 AUROC on both dormant (waiting) and fired (acting) sessions."),
    ("RQ3", "Is the shared direction really 'pre-firing', or just 'payload present'?",
     "Disentangle readiness-to-act from mere textual presence using benign twins, untriggerable payloads, activation steering, and ablation tests."),
    ("RQ4", "Can a probe trained only on fired sessions find dormant ones?",
     "Evaluate cross-state transfer: train only on observed fired incidents vs clean sessions; test on un-triggered dormant sessions across held-out goal families.")
]

y_start = Inches(2.3)
for i, (rq, title, desc) in enumerate(rqs):
    y = y_start + Inches(i * 1.05)
    add_textbox(s5, Inches(0.67), y, Inches(0.8), Inches(0.9),
                rq, font_size=16, color=ORANGE, bold=True, font_name="Courier New")
    add_textbox(s5, Inches(1.5), y, Inches(4.5), Inches(0.9),
                title, font_size=14, color=NAVY, bold=True, space_after=2)
    add_textbox(s5, Inches(6.1), y, Inches(6.6), Inches(0.9),
                desc, font_size=12.5, color=SLATE, space_after=2)

add_footer(s5)
add_slide_number(s5, 5)


# ═══════════════ SLIDE 6: Future Architecture: Probing Methodology ═══════════════
s6 = prs.slides.add_slide(blank)
add_org_label(s6)
add_category_label(s6, "METHODOLOGY & ARCHITECTURE")
add_title(s6, "Future Detection Architecture & Checkpoints", size=32)

add_section_header(s6, "01 / PROBING PIPELINE & READOUT", top=Inches(2.3))
add_textbox(s6, Inches(0.67), Inches(2.65), Inches(5.6), Inches(2.2), [
    "• Readout Site: Final prompt token vs internal memory token (memory_result_end). Standardize features across hidden dimension (d=2880).",
    "• Model: gpt-oss-20b (25 layers, 32 MoE experts). Sweep Layers 0–24.",
    "• Linear Probing: L2 regularized logistic regression, C chosen by nested cross-validation.",
    "• Direction Geometry: Difference-in-means vectors with bootstrap cosine similarity and self-similarity bounds."
], font_size=13, color=NAVY, space_after=4)

add_section_header(s6, "02 / DUAL CHECKPOINT PREEMPTION", left=Inches(6.8), top=Inches(2.3))
add_textbox(s6, Inches(6.8), Inches(2.65), Inches(5.8), Inches(2.2), [
    "• Checkpoint 1 (Pre-Generation): Probe activations immediately after prompt processing, before Token 1 is sampled. Blocks malicious thought/reasoning.",
    "• Checkpoint 2 (Pre-Tool Execution): Probe activations right before the tool invocation call is dispatched. Intercepts poisoned arguments.",
    "• Token Localization: Project each context token's residual state onto probe direction v to highlight which memory note contains the dormant poison."
], font_size=13, color=NAVY, space_after=4)

add_textbox(s6, Inches(0.67), Inches(5.85), Inches(12.0), Inches(0.8),
    "HARDWARE EFFICIENCY  /  Probe inference runs in <0.1 ms directly in GPU VRAM during the forward pass. "
    "Requires zero extra LLM calls, $0 token API costs, and zero inference latency overhead.",
    font_size=14.5, color=NAVY, bold=True, space_after=3)

add_footer(s6)
add_slide_number(s6, 6)


# ═══════════════ SLIDE 7: Causal Verification Suite (BIG FIGURE) ═══════════════
s7 = prs.slides.add_slide(blank)
add_org_label(s7)
add_category_label(s7, "CAUSAL VERIFICATION (RQ3)")
add_title(s7, "Proving Causation: Steering, Ablation & Divergence", size=30)

if CAUSAL_FIG.exists():
    s7.shapes.add_picture(str(CAUSAL_FIG),
                          Inches(0.67), Inches(2.15),
                          Inches(12.0), Inches(4.3))

add_footer(s7)
add_slide_number(s7, 7)


# ═══════════════ SLIDE 8: Mechanism Checks: Twin Controls ═══════════════
s8 = prs.slides.add_slide(blank)
add_org_label(s8)
add_category_label(s8, "MECHANISM CHECKS (RQ3)")
add_title(s8, "Disentangling Pre-Firing from 'Payload Present'", size=30)

add_section_header(s8, "01 / PAIRED BENIGN TWINS", top=Inches(2.3))
add_textbox(s8, Inches(0.67), Inches(2.65), Inches(5.6), Inches(2.3), [
    "• Setup: Construct benign notes matching the exact length, syntax, and formatting of the adversarial sleeper payload, but commanding legitimate actions.",
    "• Diagnostic Purpose: Cancels 99.95% of formatting/delimiter noise.",
    "• Validation: If the probe fires on benign twins, it merely detects formatting. True pre-firing probe must score benign twins close to clean baseline (≈ 0.003)."
], font_size=13.5, color=NAVY, space_after=4)

add_section_header(s8, "02 / UNTRIGGERABLE PAYLOADS & LOGIT LENS", left=Inches(6.8), top=Inches(2.3))
add_textbox(s8, Inches(6.8), Inches(2.65), Inches(5.8), Inches(2.3), [
    "• Untriggerable Payloads: Create hostile payloads conditioned on triggers the environment cannot satisfy (e.g., nonexistent tool or file).",
    "• Tests whether the probe detects the passive presence of instructions vs actual readiness to act.",
    "• Logit Lens Inspection: Project mid-layer hidden states onto vocabulary unembedding matrix to verify if attacker's target action tokens rank high during dormant turns."
], font_size=13.5, color=NAVY, space_after=4)

add_textbox(s8, Inches(0.67), Inches(5.85), Inches(12.0), Inches(0.8),
    "CAUSAL RIGOR  /  By combining benign twins, untriggerable controls, activation steering, "
    "and ablation, we rule out confounders and confirm the probe isolates genuine causal intent.",
    font_size=14.5, color=NAVY, bold=True, space_after=3)

add_footer(s8)
add_slide_number(s8, 8)


# ═══════════════ SLIDE 9: Cross-State Transfer (RQ4) ═══════════════
s9 = prs.slides.add_slide(blank)
add_org_label(s9)
add_category_label(s9, "CROSS-STATE GENERALIZATION (RQ4)")
add_title(s9, "Train on Fired, Find Dormant: Incident Response", size=30)

add_section_header(s9, "01 / THE INCIDENT RESPONSE SCENARIO", top=Inches(2.3))
add_textbox(s9, Inches(0.67), Inches(2.65), Inches(5.6), Inches(2.3), [
    "• The Threat Reality: In enterprise settings, defenders often discover an attack only after a single incident successfully fires in production.",
    "• Can that single observed incident be used to train a probe that scans all historical memory stores and flags dormant, un-triggered payloads?",
    "• Training Split: Train on Fired vs Clean; Test on Dormant vs Clean.",
    "• Ceiling: Compare against an oracle probe trained with ground-truth dormant labels."
], font_size=13.5, color=NAVY, space_after=4)

add_section_header(s9, "02 / METHODOLOGICAL RIGOR", left=Inches(6.8), top=Inches(2.3))
add_textbox(s9, Inches(6.8), Inches(2.65), Inches(5.8), Inches(2.3), [
    "• Leave-One-Goal-Family-Out (LODO): Entire goal families (e.g. exfiltration, sabotage, financial fraud) held out to ensure zero semantic memorization.",
    "• Near-Duplicate Search Baseline: Benchmark against string similarity & embedding search to prove the probe learns geometric concepts beyond text matching.",
    "• Propensity-Weighted PU Learning: Apply Positive-Unlabeled learning for payloads with low empirical firing rates."
], font_size=13.5, color=NAVY, space_after=4)

add_textbox(s9, Inches(0.67), Inches(5.85), Inches(12.0), Inches(0.8),
    "VALUE FOR DEFENDERS  /  If fired-to-dormant transfer succeeds, one detected incident "
    "inoculates the entire agent fleet and purges all hidden sleeper notes across the enterprise.",
    font_size=14.5, color=NAVY, bold=True, space_after=3)

add_footer(s9)
add_slide_number(s9, 9)


# ═══════════════ SLIDE 10: Future Benchmark Matrix ═══════════════
s10 = prs.slides.add_slide(blank)
add_org_label(s10)
add_category_label(s10, "EVALUATION BENCHMARK")
add_title(s10, "Comprehensive Future Benchmark Suite", size=32)

# 3 columns: States, Baselines, Metrics
add_section_header(s10, "01 / 6 AGENT STATES", top=Inches(2.3))
add_textbox(s10, Inches(0.67), Inches(2.65), Inches(3.6), Inches(2.6), [
    "1. Clean (benign memory)",
    "2. Benign Twin (matched length)",
    "3. Dormant (trigger unmet)",
    "4. Triggered-but-Resisted",
    "5. Fired (executed attack)",
    "6. Fired-without-Trigger (hallucinated obedience)",
    "• Labels judged from post-hoc execution rollouts."
], font_size=12, color=NAVY, space_after=3)

add_section_header(s10, "02 / 9 HEAD-TO-HEAD BASELINES", left=Inches(4.7), top=Inches(2.3))
add_textbox(s10, Inches(4.7), Inches(2.65), Inches(4.3), Inches(2.6), [
    "• Attention Tracker (AT) — Hung et al. 2024",
    "• AttnDA — Hung et al. 2025",
    "• TaskTracker — Abdelnabi et al. 2025",
    "• Meta Prompt Guard 2 (text)",
    "• DataFilter (text classifier)",
    "• LLM Judge (GPT-4o / Claude)",
    "• Near-Duplicate String Search",
    "• Last-Layer Residual Probe",
    "• Mid-Layer Probe (Layers 14–19, Ours)"
], font_size=12, color=NAVY, space_after=3)

add_section_header(s10, "03 / METRICS & RIGOR", left=Inches(9.2), top=Inches(2.3))
add_textbox(s10, Inches(9.2), Inches(2.65), Inches(3.5), Inches(2.6), [
    "• Primary: AUROC with case-clustered bootstrap CIs.",
    "• Enterprise SLA: TPR at 1% FPR and TPR at 0.1% FPR.",
    "• Statistical Power: ≥100 positives & ≥100 negatives per cell.",
    "• Repeated 5-fold cross-validation with paired DeLong tests."
], font_size=12, color=NAVY, space_after=3)

add_textbox(s10, Inches(0.67), Inches(5.95), Inches(12.0), Inches(0.6),
    "DATASETS  /  Hidden in Memory agent actions, GhostWriter toolkit (descriptive phrasing), "
    "and AgentDojo extended with persistent file memory.",
    font_size=13.5, color=NAVY, bold=True, space_after=2)

add_footer(s10)
add_slide_number(s10, 10)


# ═══════════════ SLIDE 11: Adaptive Attacker Evaluation ═══════════════
s11 = prs.slides.add_slide(blank)
add_org_label(s11)
add_category_label(s11, "ADAPTIVE ROBUSTNESS")
add_title(s11, "Adaptive Attacker: Stress-Testing the Probe", size=32)

add_section_header(s11, "01 / WHITE-BOX ADAPTIVE ADVERSARY", top=Inches(2.3))
add_textbox(s11, Inches(0.67), Inches(2.65), Inches(5.6), Inches(2.3), [
    "• Actor–Critic Loop: An adversarial generator (Actor) creates sleeper prompts conditioned on a loss that minimizes the Probe score (Critic) while maximizing trigger execution.",
    "• Loss Formulation: L = L_fire - λ · score_probe(h).",
    "• Tests whether an attacker with full white-box knowledge of the probe vector v can craft payloads that remain invisible to the probe."
], font_size=13.5, color=NAVY, space_after=4)

add_section_header(s11, "02 / NATURAL LINGUISTIC ADAPTATIONS", left=Inches(6.8), top=Inches(2.3))
add_textbox(s11, Inches(6.8), Inches(2.65), Inches(5.8), Inches(2.3), [
    "• Polite Conversational Disguise: Payloads written as neutral system preferences or user quirks (tested via GhostWriter).",
    "• Semantic Dispersion: Splitting trigger and action logic across multiple distinct memory entries to avoid dense activation clusters.",
    "• Robustness Benchmark: Measure probe AUROC degradation under iterative gradient optimization vs baseline text filters."
], font_size=13.5, color=NAVY, space_after=4)

add_textbox(s11, Inches(0.67), Inches(5.85), Inches(12.0), Inches(0.8),
    "SECURITY GUARANTEE  /  Evaluating against adaptive adversaries establishes the theoretical "
    "resilience boundary and prevents the detector from being trivially bypassed by prompt engineering.",
    font_size=14.5, color=NAVY, bold=True, space_after=3)

add_footer(s11)
add_slide_number(s11, 11)


# ═══════════════ SLIDE 12: Roadmap & Experimental Timeline ═══════════════
s12 = prs.slides.add_slide(blank)
add_org_label(s12)
add_category_label(s12, "PROJECT ROADMAP")
add_title(s12, "Experimental Roadmap & Milestones", size=32)

phases = [
    ("Phase 1: Baselines (Nov 1–15)",
     "Implement Attention Tracker (AT) & AttnDA extraction on gpt-oss-20b; run head-to-head on dormant vs fired turns.",
     "Deliverable: Empirical confirmation of AT dormant failure (RQ1)."),
    ("Phase 2: Causality (Nov 16–30)",
     "Perform activation steering (+v) on dormant turns and ablation (-v) on fired turns; measure logit lens and divergence curves.",
     "Deliverable: Proof that direction causes pre-firing readiness (RQ3)."),
    ("Phase 3: Transfer (Dec 1–15)",
     "Evaluate cross-state transfer (train on fired, test on dormant) across held-out goal families with PU learning.",
     "Deliverable: Validated incident-response protocol (RQ4)."),
    ("Phase 4: Synthesis (Dec 16–31)",
     "Benchmark adaptive attacker loop; replicate on second open model (Llama-3-70B); finalize symposium poster and paper.",
     "Deliverable: Completed workshop / conference paper submission.")
]

y_start = Inches(2.3)
for i, (phase, desc, deliv) in enumerate(phases):
    y = y_start + Inches(i * 1.05)
    add_textbox(s12, Inches(0.67), y, Inches(3.2), Inches(0.9),
                phase, font_size=14, color=NAVY, bold=True, space_after=2)
    add_textbox(s12, Inches(4.0), y, Inches(5.0), Inches(0.9),
                desc, font_size=12.5, color=NAVY, space_after=2)
    add_textbox(s12, Inches(9.2), y, Inches(3.5), Inches(0.9),
                deliv, font_size=12, color=GREEN_DARK, bold=True, space_after=2)

add_footer(s12)
add_slide_number(s12, 12)


# ═══════════════ SLIDE 13: Summary of Contributions ═══════════════
s13 = prs.slides.add_slide(blank)
add_org_label(s13)
add_category_label(s13, "PROJECT CONTRIBUTIONS")
add_title(s13, "Summary of Expected Contributions", size=32)

contribs = [
    ("1. First White-Box Dormant Memory Detector",
     "A lightweight, zero-token preemption probe running inside the agent session that catches sleeping payloads before environmental triggers occur."),
    ("2. Empirical Proof of Attention Blindspots",
     "Rigorous head-to-head demonstration showing Attention Tracker (AT) and AttnDA fail on dormant and descriptive payloads because no attention distraction occurs."),
    ("3. Discovery of the 'Pre-Firing' Latent State",
     "Mechanistic evidence that dormant and fired sessions share a common subspace in middle layers (Layers 14–19), while execution commitment diverges in late layers."),
    ("4. Causal Validation via Steering & Ablation",
     "Proving the probe vector directly controls execution readiness—steering dormant sessions induces firing, while ablating it neutralizes fired attacks."),
    ("5. Fired-to-Dormant Transfer Protocol",
     "Demonstrating that post-incident data from a single fired attack is sufficient to sweep and purify un-triggered memory stores across an enterprise fleet.")
]

y_start = Inches(2.25)
for i, (head, body) in enumerate(contribs):
    y = y_start + Inches(i * 0.9)
    add_textbox(s13, Inches(0.67), y, Inches(4.2), Inches(0.75),
                head, font_size=14, color=NAVY, bold=True, space_after=2)
    add_textbox(s13, Inches(5.0), y, Inches(7.6), Inches(0.75),
                body, font_size=13, color=SLATE, space_after=2)

add_footer(s13)
add_slide_number(s13, 13)


# ═══════════════ SLIDE 14: Discussion & Feedback (Dark Navy) ═══════════════
s14 = prs.slides.add_slide(blank)
bg14 = s14.background
fill14 = bg14.fill
fill14.solid()
fill14.fore_color.rgb = NAVY

add_textbox(s14, Inches(0.67), Inches(0.35), Inches(4), Inches(0.3),
            "AI ALIGNMENT @ ILLINOIS", font_size=11, color=WHITE,
            bold=True, font_name="Courier New")
add_textbox(s14, Inches(0.67), Inches(0.75), Inches(4), Inches(0.3),
            "DISCUSSION & FEEDBACK", font_size=11, color=ORANGE,
            bold=True, font_name="Courier New")

add_textbox(s14, Inches(0.67), Inches(1.3), Inches(11.5), Inches(0.8),
            "Key Decisions for Fellowship Alignment",
            font_size=36, color=ORANGE, bold=True)

add_textbox(s14, Inches(0.67), Inches(2.55), Inches(4), Inches(0.3),
            "01 / CAUSAL STEERING METHOD", font_size=11, color=ORANGE,
            bold=True, font_name="Courier New")
add_textbox(s14, Inches(0.67), Inches(2.95), Inches(11.5), Inches(0.7),
            "Should we steer along the mean-difference direction (diff-in-means) or the orthogonal weight vector w of the L2 logistic probe?",
            font_size=16, color=WHITE, space_after=4)

add_textbox(s14, Inches(0.67), Inches(3.85), Inches(4), Inches(0.3),
            "02 / ATTENTION HOOKING SCOPE", font_size=11, color=ORANGE,
            bold=True, font_name="Courier New")
add_textbox(s14, Inches(0.67), Inches(4.25), Inches(11.5), Inches(0.7),
            "Should our AT baseline replicate Hung et al.'s top-10 important head selection, or sweep all attention heads dynamically?",
            font_size=16, color=WHITE, space_after=4)

add_textbox(s14, Inches(0.67), Inches(5.15), Inches(4), Inches(0.3),
            "03 / COMPUTE & HARDWARE NEEDS", font_size=11, color=ORANGE,
            bold=True, font_name="Courier New")
add_textbox(s14, Inches(0.67), Inches(5.55), Inches(11.5), Inches(0.7),
            "Access to an A100 / H100 node (~8 GPU hours) to extract full attention matrices and perform layerwise activation steering sweeps.",
            font_size=16, color=WHITE, space_after=4)

add_textbox(s14, Inches(0.67), Inches(7.05), Inches(10), Inches(0.35),
            "RESEARCH PROPOSAL  /  DORMANT SLEEPER DETECTION",
            font_size=10, color=LIGHT_GRAY, font_name="Courier New")
add_slide_number(s14, 14)


# ════════════════════════════════════════════════════════════════
#  SAVE
# ════════════════════════════════════════════════════════════════
out_pptx = OUT_DIR / 'AIA_Sleeper_Detection_Architecture_Overhaul.pptx'
prs.save(str(out_pptx))
print(f"✓ Saved: {out_pptx}")

# Copy to authentic presentation name
legacy_pptx = OUT_DIR / 'AIA Presentation slides.pptx'
shutil.copy(out_pptx, legacy_pptx)
print(f"✓ Copied to: {legacy_pptx}")

# Copy to artifacts
os.makedirs(ARTIFACT_DIR, exist_ok=True)
shutil.copy(out_pptx, ARTIFACT_DIR / out_pptx.name)
print(f"✓ Copied to artifacts: {ARTIFACT_DIR / out_pptx.name}")
