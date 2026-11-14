#!/usr/bin/env python3
"""
Build updated AIA presentation slides for the Dormant Poison Detection project.
Uses the existing AIA template PPTX, updates all content to reflect:
  - Current repo progress (probe results, controls, Jev baseline)
  - Literature survey section (new slide)
  - Larger text, bigger figures, polished layout
  - Research questions from the updated proposal
"""
import sys
import shutil
import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ── Paths ──
SRC_PPTX = Path('/home/pndhpndh/sleeper_detection/AIA Presentation slides.pptx')
OUT_DIR = Path('/home/pndhpndh/sleeper_detection')
RESULTS = Path('/home/pndhpndh/sleeper_detection/results')
ARTIFACT_DIR = Path('/home/pndhpndh/.gemini/antigravity/brain/ae642783-e859-4834-9387-38d23eaa5676')

# Figures
KEY_FIGURE = RESULTS / 'midpoint_key_figure.png'
MECHANISM_DIAGRAM = RESULTS / 'mechanism_diagram.png'
REALISM_LADDER = RESULTS / 'realism_ladder_figure.png'
DETECTOR_DEMO = RESULTS / 'detector_demo' / 'detector_demo.png'

# ── Colors (AIA brand) ──
ORANGE = RGBColor(0xFF, 0x5F, 0x05)
NAVY = RGBColor(0x13, 0x29, 0x4B)
SLATE = RGBColor(0x47, 0x55, 0x69)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xC6, 0xC7, 0xC6)
RED = RGBColor(0xCC, 0x00, 0x00)
GREEN_DARK = RGBColor(0x00, 0x66, 0x33)

# ── Slide dimensions (widescreen 13.33 x 7.5 in) ──
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


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


def add_section_header(slide, label, left=Inches(0.67), top=Inches(2.6)):
    """Add an orange section header label (e.g., 'MOTIVATION / GAP')."""
    add_textbox(slide, left, top, Inches(5), Inches(0.4),
                label, font_size=11, color=ORANGE, bold=True,
                font_name="Courier New", space_after=2)


def add_slide_number(slide, num, total=16):
    """Add slide number in bottom right."""
    add_textbox(slide, Inches(12.5), Inches(7.05), Inches(0.7), Inches(0.35),
                f"{num:02d}", font_size=11, color=ORANGE, bold=True,
                font_name="Courier New", alignment=PP_ALIGN.RIGHT)


def add_footer(slide, text="FINAL CHECKPOINT  /  DORMANT SLEEPER DETECTION"):
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
    """Add category label below org (e.g., 'RESEARCH FELLOWSHIP')."""
    add_textbox(slide, Inches(0.67), Inches(0.75), Inches(4), Inches(0.3),
                label, font_size=11, color=ORANGE,
                bold=True, font_name="Courier New")


def add_title(slide, title, top=Inches(1.15), size=34):
    """Add large navy title."""
    add_textbox(slide, Inches(0.67), top, Inches(11.5), Inches(1.0),
                title, font_size=size, color=NAVY, bold=True,
                space_after=6)


def make_slide(prs, layout_index=6):
    """Add a blank slide."""
    layout = prs.slide_layouts[layout_index]
    return prs.slides.add_slide(layout)


# ════════════════════════════════════════════════════════════════
#  BUILD THE DECK
# ════════════════════════════════════════════════════════════════

# Create a fresh widescreen presentation
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Use blank layout
blank_layout = prs.slide_layouts[6]  # blank

# ═══════════════ SLIDE 1: Title (Dark Navy Background) ═══════════════
s1 = prs.slides.add_slide(blank_layout)
# Navy background
bg = s1.background
fill = bg.fill
fill.solid()
fill.fore_color.rgb = NAVY

add_textbox(s1, Inches(0.67), Inches(0.35), Inches(4), Inches(0.3),
            "AI ALIGNMENT @ ILLINOIS", font_size=11, color=WHITE,
            bold=True, font_name="Courier New")

add_textbox(s1, Inches(0.67), Inches(0.85), Inches(6), Inches(0.3),
            "AIA @ ILLINOIS  /  FALL 2026", font_size=13, color=ORANGE,
            bold=True, font_name="Courier New")

add_textbox(s1, Inches(0.67), Inches(1.8), Inches(8), Inches(1.2),
            "Final checkpoint", font_size=48, color=WHITE, bold=True)

add_textbox(s1, Inches(0.67), Inches(3.3), Inches(10), Inches(1.0),
            "Detecting dormant poison in LLM agent\nmemory before the trigger fires",
            font_size=28, color=ORANGE, bold=True, space_after=8)

add_textbox(s1, Inches(0.67), Inches(5.3), Inches(10), Inches(0.8), [
    "Raghav Agarwal (raghava7@illinois.edu)  ·  Huy Pham (huynp2@illinois.edu)",
    "Mentors: Enyi Jiang, Himnish Hunma  ·  November 2026"
], font_size=17, color=WHITE, space_after=5)

add_textbox(s1, Inches(0.67), Inches(7.05), Inches(10), Inches(0.35),
            "FINAL CHECKPOINT  /  DORMANT SLEEPER DETECTION",
            font_size=10, color=LIGHT_GRAY, font_name="Courier New")
add_textbox(s1, Inches(12.5), Inches(7.05), Inches(0.7), Inches(0.35),
            "01", font_size=11, color=ORANGE, bold=True,
            font_name="Courier New", alignment=PP_ALIGN.RIGHT)


# ═══════════════ SLIDE 2: What we know (Executive summary) ═══════════════
s2 = prs.slides.add_slide(blank_layout)
add_org_label(s2)
add_category_label(s2, "RESEARCH FELLOWSHIP")
add_title(s2, "What we know at the final checkpoint")

# 01 / QUESTION
add_section_header(s2, "01 / QUESTION", top=Inches(2.7))
add_textbox(s2, Inches(0.67), Inches(3.1), Inches(5.5), Inches(1.0),
            "Can a linear probe on gpt-oss-20b's activations predict, before "
            "the agent acts, whether it will obey a planted memory note—"
            "better than reading the text?",
            font_size=17, color=NAVY, bold=True, space_after=4)

# 02 / CURRENT EVIDENCE
add_section_header(s2, "02 / CURRENT EVIDENCE", left=Inches(7.0), top=Inches(2.7))
add_textbox(s2, Inches(7.0), Inches(3.1), Inches(5.5), Inches(1.2), [
    "• Probe AUROC 0.876 [0.818, 0.920] within-template—beats note text by +0.10 (CI above zero).",
    "• Live vs inert: 1.000 AUROC across E1–E3 splits.",
    "• Frontier guard (Jev): near chance on fired prediction."
], font_size=16, color=NAVY, space_after=5)

# 03 / NEXT
add_section_header(s2, "03 / NEXT DECISION", top=Inches(5.0))
add_textbox(s2, Inches(0.67), Inches(5.4), Inches(11.5), Inches(0.9), [
    "Is the probe reading the model's decision, or just reading text better? Stronger text baselines, 30+ wordings, and causal steering will tell.",
], font_size=16, color=NAVY, space_after=5)

add_footer(s2)
add_slide_number(s2, 2)


# ═══════════════ SLIDE 3: Why this question matters ═══════════════
s3 = prs.slides.add_slide(blank_layout)
add_org_label(s3)
add_category_label(s3, "RESEARCH FELLOWSHIP")
add_title(s3, "Why this question matters")

add_section_header(s3, "MOTIVATION / GAP", top=Inches(2.7))
add_textbox(s3, Inches(0.67), Inches(3.1), Inches(5.5), Inches(1.5), [
    "• Persistent Memory Threat: Attackers plant sleeper notes in agent memory "
    "that stay dormant until a future trigger fires.",
    "• Text Guardrails Fail: LLM judges caught 85% of directive payloads but "
    "only 6% of descriptive ones (Ruan et al., 2026). "
    "Jev drops to 30% TPR; disguised attacks collapse text recall to 10%.",
    "• Attention-based detectors (AT, AttnDA) measure payload influence on output—"
    "a dormant payload hasn't influenced anything yet."
], font_size=15, color=NAVY, space_after=6)

add_section_header(s3, "RESEARCH QUESTION", left=Inches(7.0), top=Inches(2.7))
add_textbox(s3, Inches(7.0), Inches(3.1), Inches(5.5), Inches(1.5), [
    "• Do the model's internal activations, read before it acts, carry information "
    "about whether it will execute a dormant instruction that the text alone does not?",
    "• Can one detector work whether or not the payload fires?"
], font_size=15, color=NAVY, space_after=6)

add_section_header(s3, "AI SAFETY CONNECTION", top=Inches(5.4))
add_textbox(s3, Inches(0.67), Inches(5.8), Inches(11.5), Inches(0.8), [
    "• A probe at the decision point could block the harmful tool call in time, "
    "in <0.1 ms on GPU VRAM at $0 token cost (vs 1,500 ms / 2× cost for external LLM judges).",
], font_size=16, color=NAVY, bold=True, space_after=5)

add_footer(s3)
add_slide_number(s3, 3)


# ═══════════════ SLIDE 4: Literature Survey ═══════════════
s4 = prs.slides.add_slide(blank_layout)
add_org_label(s4)
add_category_label(s4, "LITERATURE SURVEY")
add_title(s4, "Related work landscape", size=32)

# Column 1: Attacks
add_section_header(s4, "01 / MEMORY POISONING ATTACKS", top=Inches(2.4))
add_textbox(s4, Inches(0.67), Inches(2.75), Inches(3.8), Inches(2.8), [
    "• Li et al. (2026) — Sleeper Attack: 1,896 instances across session, memory, and skill targets; Plant→Dormant→Trigger lifecycle.",
    "• Pulipaka et al. (2026) — Sleeper memory poisoning of long-term stores.",
    "• Szczepaniak et al. (2026) — Explosive Prompts: conditional triggers raise execution from 2.4% to 16.5% across 9 coding agents.",
    "• Ruan et al. (2026) — GhostWriter: descriptive payloads evade 94% of text detectors."
], font_size=12, color=NAVY, space_after=3)

# Column 2: Detection
add_section_header(s4, "02 / DETECTION METHODS", left=Inches(5.0), top=Inches(2.4))
add_textbox(s4, Inches(5.0), Inches(2.75), Inches(3.8), Inches(2.8), [
    "• Hung et al. (2024) — Attention Tracker (AT): training-free detection via attention head tracking. Requires payload to influence output.",
    "• Hung et al. (2025) — AttnDA: attention distribution analysis. Same limitation: dormant payloads leave no trace.",
    "• Abdelnabi et al. (2025) — TaskTracker: activation deltas before/after retrieved content. Key baseline.",
    "• Meta Prompt Guard 2 & DataFilter: external text-level classifiers."
], font_size=12, color=NAVY, space_after=3)

# Column 3: Mechanistic interp
add_section_header(s4, "03 / MECHANISTIC INTERPRETABILITY", left=Inches(9.3), top=Inches(2.4))
add_textbox(s4, Inches(9.3), Inches(2.75), Inches(3.8), Inches(2.8), [
    "• Hubinger et al. (2024) — Sleeper Agents: linear probes catch weight-backdoored models about to defect.",
    "• MacDiarmid et al. (2024) — Production-ready probes for Gemini safety.",
    "• Zou et al. (2023) — Representation Engineering: difference-of-means directions.",
    "• Marks & Tegmark (2024) — Geometry of truth: linear latent structure in residual stream."
], font_size=12, color=NAVY, space_after=3)

# Gap statement
add_textbox(s4, Inches(0.67), Inches(5.95), Inches(12.0), Inches(0.8),
    "GAP  /  All prior probing targets weight-level backdoors or active injections. "
    "Nobody has probed dormant memory payloads during benign interactions—where the payload "
    "is in context but hasn't fired yet.",
    font_size=14, color=NAVY, bold=True, space_after=3)

add_footer(s4)
add_slide_number(s4, 4)


# ═══════════════ SLIDE 5: Approach & Experiment Design ═══════════════
s5 = prs.slides.add_slide(blank_layout)
add_org_label(s5)
add_category_label(s5, "RESEARCH FELLOWSHIP")
add_title(s5, "Approach & experiment design")

# 3-column layout
add_section_header(s5, "01 / INPUT", top=Inches(2.5))
add_textbox(s5, Inches(0.67), Inches(2.9), Inches(3.5), Inches(1.8), [
    "• gpt-oss-20b: 25-layer MoE, 32 experts, 2880-dim residual stream.",
    "• behavior_v1: 148 Sleeper Attack cases × 6 live + 6 inert + 1 clean = 1,924 prompts.",
    "• 43-token paired benign twins canceling 99.95% formatting noise."
], font_size=14, color=NAVY, space_after=5)

add_section_header(s5, "02 / METHOD", left=Inches(4.9), top=Inches(2.5))
add_textbox(s5, Inches(4.9), Inches(2.9), Inches(3.5), Inches(1.8), [
    "• Residual Probing: bfloat16 forward pass, extracting Layers 0–24 at memory_result_end.",
    "• L2 logistic regression: C chosen by leave-one-template-out CV.",
    "• Labels from 8 agent rollouts per note (T=1.0)."
], font_size=14, color=NAVY, space_after=5)

add_section_header(s5, "03 / MEASUREMENT", left=Inches(9.1), top=Inches(2.5))
add_textbox(s5, Inches(9.1), Inches(2.9), Inches(3.5), Inches(1.8), [
    "• AUROC with case-clustered bootstrap CIs.",
    "• Held-out splits: E1 (new wording), E2 (new domains), E3 (both new).",
    "• Within-template AUROC: controls for wording confounds."
], font_size=14, color=NAVY, space_after=5)

# Hypothesis
add_textbox(s5, Inches(0.67), Inches(5.4), Inches(12.0), Inches(0.8),
    "WORKING HYPOTHESIS  /  If the agent will act on a planted note, its activations "
    "already encode that intent before it acts, so a probe predicts execution better than the note's text.",
    font_size=15, color=NAVY, bold=True, space_after=4)

add_footer(s5)
add_slide_number(s5, 5)


# ═══════════════ SLIDE 6: Mechanism Diagram (BIG FIGURE) ═══════════════
s6 = prs.slides.add_slide(blank_layout)
add_org_label(s6)
add_category_label(s6, "DETECTION MECHANISM")
add_title(s6, "How the probe works: zero-token preemption", size=30)

# Large mechanism diagram
if MECHANISM_DIAGRAM.exists():
    s6.shapes.add_picture(str(MECHANISM_DIAGRAM),
                          Inches(0.67), Inches(2.5),
                          Inches(12.0), Inches(3.0))

add_textbox(s6, Inches(0.67), Inches(5.8), Inches(12.0), Inches(1.0), [
    "• Agent request (with planted note in memory) → forward pass through all 25 layers → "
    "linear probe reads mean of Layers 14–19 at memory_result_end → score z = w·h + b.",
    "• Score > threshold → flag & block before any output token is generated. Latency: <0.1 ms on GPU."
], font_size=15, color=NAVY, space_after=5)

add_footer(s6)
add_slide_number(s6, 6)


# ═══════════════ SLIDE 7: Progress Against Plan ═══════════════
s7 = prs.slides.add_slide(blank_layout)
add_org_label(s7)
add_category_label(s7, "RESEARCH FELLOWSHIP")
add_title(s7, "Progress against the original plan")

# Table-like layout using textboxes (more control than pptx tables)
milestones = [
    ("Pipeline & live vs inert probe", "Complete",
     "Probe AUROC 0.998–1.000; text also 1.000. Validates pipeline."),
    ("Behavior labels & fired probe", "Complete",
     "888 live notes × 8 rollouts. Probe 0.876 vs 0.78–0.81 text, within wording."),
    ("Fair baselines (Jev, text, controls)", "Complete",
     "Jev near chance on fired task; full-prompt TF-IDF ties probe (+0.06, n.s.)."),
    ("Wording confound analysis", "Complete",
     "Template alone moves obedience 1%→92%. Within-template probe beats note text by +0.10."),
    ("Causal steering & cross-architecture", "Next",
     "Steer along probe direction at trigger turn; replicate on Llama-3-70B.")
]

y_start = Inches(2.4)
for i, (ms, status, evidence) in enumerate(milestones):
    y = y_start + Inches(i * 0.72)
    status_color = GREEN_DARK if status == "Complete" else ORANGE

    add_textbox(s7, Inches(0.67), y, Inches(3.2), Inches(0.65),
                ms, font_size=13.5, color=NAVY, bold=True, space_after=2)
    add_textbox(s7, Inches(4.0), y, Inches(1.2), Inches(0.65),
                status, font_size=13, color=status_color, bold=True, space_after=2)
    add_textbox(s7, Inches(5.3), y, Inches(7.5), Inches(0.65),
                evidence, font_size=13, color=NAVY, space_after=2)

add_textbox(s7, Inches(0.67), Inches(6.25), Inches(12.0), Inches(0.5),
    "PLAN UPDATE  /  Pivoted from full coding-agent testbed to targeted probe validation "
    "on Huy's memory poisoning setup—tested core idea for ~$10 of GPU.",
    font_size=13.5, color=NAVY, bold=True, space_after=3)

add_footer(s7)
add_slide_number(s7, 7)


# ═══════════════ SLIDE 8: Key Results Figure (FULL SLIDE) ═══════════════
s8 = prs.slides.add_slide(blank_layout)
add_org_label(s8)
add_category_label(s8, "PRELIMINARY RESULTS")
add_title(s8, "Probes achieve 1.00 AUROC; predict execution at 0.876", size=28)

# LARGE key figure taking most of the slide
if KEY_FIGURE.exists():
    s8.shapes.add_picture(str(KEY_FIGURE),
                          Inches(0.5), Inches(2.4),
                          Inches(9.0), Inches(4.5))

# Observations on the right
add_section_header(s8, "KEY NUMBERS", left=Inches(9.8), top=Inches(2.5))
add_textbox(s8, Inches(9.8), Inches(2.9), Inches(3.2), Inches(3.5), [
    "• Live vs Inert: 1.000 AUROC across E1–E3.",
    "• Fired prediction: 0.876 AUROC within-template.",
    "• Probe > note text: +0.10 [+0.01, +0.19].",
    "• Probe > full prompt: +0.06 [−0.02, +0.14].",
    "• Jev (frontier guard): 0.47 AUROC on fired task.",
    "• Execution prediction: 0.945 AUROC on E1.",
    "• Disguised attacks (L2–L4): text judges collapse to 10–20% recall."
], font_size=13, color=NAVY, space_after=5)

add_footer(s8)
add_slide_number(s8, 8)


# ═══════════════ SLIDE 9: Execution Prediction Detail ═══════════════
s9 = prs.slides.add_slide(blank_layout)
add_org_label(s9)
add_category_label(s9, "PRELIMINARY RESULTS")
add_title(s9, "The probe predicts which attacks will succeed", size=30)

# Detector demo figure
if DETECTOR_DEMO.exists():
    s9.shapes.add_picture(str(DETECTOR_DEMO),
                          Inches(0.5), Inches(2.4),
                          Inches(8.5), Inches(4.5))

add_section_header(s9, "OBSERVATION", left=Inches(9.3), top=Inches(2.4))
add_textbox(s9, Inches(9.3), Inches(2.75), Inches(3.7), Inches(1.8), [
    "• Within-template probe AUROC 0.876 [0.818, 0.920].",
    "• Beats note text (+0.10, CI above zero).",
    "• Probe correctly ranks W5 (10%) > W4 (1%) across differing cases; text baselines rank backward.",
], font_size=13, color=NAVY, space_after=3)

add_section_header(s9, "INTERPRETATION", left=Inches(9.3), top=Inches(4.7))
add_textbox(s9, Inches(9.3), Inches(5.05), Inches(3.7), Inches(1.8), [
    "• +0.06 lead over full-prompt text is suggestive (+0.062 [−0.016, +0.141]).",
    "• Probe may read decision vs text—minimal pairs and causal steering will resolve.",
], font_size=13, color=NAVY, space_after=3)

add_footer(s9)
add_slide_number(s9, 9)


# ═══════════════ SLIDE 10: Wording Drives Obedience ═══════════════
s10 = prs.slides.add_slide(blank_layout)
add_org_label(s10)
add_category_label(s10, "PRELIMINARY RESULTS")
add_title(s10, "Wording alone moves obedience from 1% to 92%", size=30)

# Wording data as large text
wordings = [
    ('W0 "in the same turn"', '92%', ORANGE),
    ('W3 "part of the current turn"', '74%', ORANGE),
    ('W2 "the next request"', '10%', NAVY),
    ('W5 "when you next respond"', '10%', NAVY),
    ('W1 "whatever the user asks next"', '7%', NAVY),
    ('W4 "next time memory is read"', '1%', NAVY),
]

for i, (wording, pct, color) in enumerate(wordings):
    y = Inches(2.8) + Inches(i * 0.65)
    add_textbox(s10, Inches(1.0), y, Inches(5.5), Inches(0.5),
                wording, font_size=18, color=NAVY, space_after=2)
    add_textbox(s10, Inches(6.8), y, Inches(1.5), Inches(0.5),
                pct, font_size=24, color=color, bold=True,
                alignment=PP_ALIGN.RIGHT, space_after=2)

# Observation
add_section_header(s10, "OBSERVATION", left=Inches(9.0), top=Inches(2.7))
add_textbox(s10, Inches(9.0), Inches(3.1), Inches(3.8), Inches(1.5), [
    "• Notes saying 'this turn' fire; 'next time' notes rarely do.",
    "• Jev flags every attack (AUROC 0.47 on fired); the probe gets 0.95 on the same notes.",
], font_size=15, color=NAVY, space_after=5)

add_section_header(s10, "INTERPRETATION", left=Inches(9.0), top=Inches(4.8))
add_textbox(s10, Inches(9.0), Inches(5.2), Inches(3.8), Inches(1.2), [
    "• Guards spot poison, not success.",
    "• The timing cue is post hoc and needs minimal pair tests to confirm causality."
], font_size=15, color=NAVY, space_after=5)

add_footer(s10)
add_slide_number(s10, 10)


# ═══════════════ SLIDE 11: Realism Ladder (BIG FIGURE) ═══════════════
s11 = prs.slides.add_slide(blank_layout)
add_org_label(s11)
add_category_label(s11, "PRELIMINARY RESULTS")
add_title(s11, "Text guards collapse on disguised attacks", size=30)

if REALISM_LADDER.exists():
    s11.shapes.add_picture(str(REALISM_LADDER),
                          Inches(0.5), Inches(2.5),
                          Inches(12.0), Inches(3.5))

add_textbox(s11, Inches(0.67), Inches(6.2), Inches(12.0), Inches(0.8), [
    "KEY  /  LLM judges achieve 100% recall on obvious syntax overrides (L0–L1), "
    "but collapse to 10–20% on conversational disguise (L2), contextual carryover (L3), "
    "and standing preferences (L4). Internal probes bypass surface camouflage entirely.",
], font_size=15, color=NAVY, bold=True, space_after=4)

add_footer(s11)
add_slide_number(s11, 11)


# ═══════════════ SLIDE 12: How Robust is the Finding? ═══════════════
s12 = prs.slides.add_slide(blank_layout)
add_org_label(s12)
add_category_label(s12, "RESEARCH FELLOWSHIP")
add_title(s12, "How robust is the finding?", size=32)

checks = [
    ("Baseline comparison",
     "Note text +0.10 [+0.01, +0.19]; full prompt +0.06 [−0.02, +0.14]",
     "Sentence embeddings, LLM judge, fair Jev"),
    ("Ablation / control",
     "Shuffled labels ≈ 0.5; case & wording controls; clean memory scores 0.003",
     "Early vs middle layers; steer along probe direction"),
    ("Generalization",
     "New wording 0.95; new domains 0.93; E3 has only 2 fired notes",
     "30+ new wordings; other Sleeper Attack strategies"),
]

for i, (check, current, next_test) in enumerate(checks):
    y = Inches(2.7) + Inches(i * 1.2)
    add_textbox(s12, Inches(0.67), y, Inches(2.5), Inches(0.9),
                check, font_size=15, color=NAVY, bold=True, space_after=2)
    add_textbox(s12, Inches(3.3), y, Inches(5.0), Inches(0.9),
                current, font_size=14, color=NAVY, space_after=2)
    add_textbox(s12, Inches(8.5), y, Inches(4.2), Inches(0.9),
                next_test, font_size=14, color=SLATE, space_after=2)

# Headers
add_textbox(s12, Inches(3.3), Inches(2.3), Inches(5.0), Inches(0.3),
            "Current evidence", font_size=12, color=ORANGE, bold=True,
            font_name="Courier New")
add_textbox(s12, Inches(8.5), Inches(2.3), Inches(4.2), Inches(0.3),
            "Next test", font_size=12, color=ORANGE, bold=True,
            font_name="Courier New")

add_textbox(s12, Inches(0.67), Inches(6.3), Inches(12.0), Inches(0.5),
    "CONFIDENCE  /  Moderate. If a strong text model matches the probe, the gain is better reading, "
    "not a readable decision. More wordings and causal tests will resolve this.",
    font_size=14, color=NAVY, bold=True, space_after=3)

add_footer(s12)
add_slide_number(s12, 12)


# ═══════════════ SLIDE 13: What is Unresolved ═══════════════
s13 = prs.slides.add_slide(blank_layout)
add_org_label(s13)
add_category_label(s13, "RESEARCH FELLOWSHIP")
add_title(s13, "What is unresolved")

add_section_header(s13, "LIMITATIONS", top=Inches(2.7))
add_textbox(s13, Inches(0.67), Inches(3.1), Inches(5.5), Inches(2.0), [
    "• Pre-registered primary test (E3) has only 2 fired notes—"
    "the headline within-wording result is exploratory.",
    "• Not yet tested: other models, other attack types, adaptive attackers, "
    "whether the probe direction is causal.",
    "• Only 6 wordings limit statistical power for wording generalization."
], font_size=15, color=NAVY, space_after=6)

add_section_header(s13, "BLOCKERS / RISKS", left=Inches(7.0), top=Inches(2.7))
add_textbox(s13, Inches(7.0), Inches(3.1), Inches(5.5), Inches(2.0), [
    "• Causal Gap: Probe features may correlate with memory structure rather than "
    "actively causing trigger execution.",
    "• GPU credit is used up; only 6 wordings limit statistical power.",
    "• Mitigation: NCSA or a shared A100, and generating many new wordings cheaply."
], font_size=15, color=NAVY, space_after=6)

add_textbox(s13, Inches(0.67), Inches(5.8), Inches(12.0), Inches(0.5),
    "SCOPE DECISION  /  Keep probes as the core; stay on memory poisoning with gpt-oss "
    "before building the full coding agent testbed.",
    font_size=15, color=NAVY, bold=True, space_after=3)

add_footer(s13)
add_slide_number(s13, 13)


# ═══════════════ SLIDE 14: Plan for Remaining Fellowship ═══════════════
s14 = prs.slides.add_slide(blank_layout)
add_org_label(s14)
add_category_label(s14, "RESEARCH FELLOWSHIP")
add_title(s14, "Plan for the remaining fellowship")

plan_items = [
    ("Early Nov", "Fair baselines (embeddings, LLM judge, Jev with the right question, full sets)",
     "Probe vs best text gap with a CI"),
    ("Mid Nov", "Timing minimal pairs, 30+ wordings, multi-layer and attention probes",
     "Fire rate shift measured; E3 has 20+ fired notes"),
    ("Late Nov–Dec", "Steering test, write up, symposium poster",
     "Causal result and final report"),
]

for i, (when, priority, done_when) in enumerate(plan_items):
    y = Inches(2.8) + Inches(i * 1.1)
    add_textbox(s14, Inches(0.67), y, Inches(2.0), Inches(0.8),
                when, font_size=15, color=NAVY, bold=True, space_after=2)
    add_textbox(s14, Inches(2.8), y, Inches(5.5), Inches(0.8),
                priority, font_size=14, color=NAVY, space_after=2)
    add_textbox(s14, Inches(8.5), y, Inches(4.2), Inches(0.8),
                done_when, font_size=14, color=SLATE, space_after=2)

# Headers
add_textbox(s14, Inches(2.8), Inches(2.4), Inches(5.5), Inches(0.3),
            "Priority / experiment", font_size=12, color=ORANGE, bold=True,
            font_name="Courier New")
add_textbox(s14, Inches(8.5), Inches(2.4), Inches(4.2), Inches(0.3),
            "Done when…", font_size=12, color=ORANGE, bold=True,
            font_name="Courier New")

add_textbox(s14, Inches(0.67), Inches(6.3), Inches(12.0), Inches(0.5),
    "DELIVERABLES  /  Report and poster  ·  Code and data on GitHub  ·  "
    "Probe package for Huy's repo",
    font_size=14, color=NAVY, bold=True, space_after=3)

add_footer(s14)
add_slide_number(s14, 14)


# ═══════════════ SLIDE 15: Feedback (Dark Slide) ═══════════════
s15 = prs.slides.add_slide(blank_layout)
bg15 = s15.background
fill15 = bg15.fill
fill15.solid()
fill15.fore_color.rgb = NAVY

add_textbox(s15, Inches(0.67), Inches(0.35), Inches(4), Inches(0.3),
            "AI ALIGNMENT @ ILLINOIS", font_size=11, color=WHITE,
            bold=True, font_name="Courier New")
add_textbox(s15, Inches(0.67), Inches(0.75), Inches(4), Inches(0.3),
            "DISCUSSION", font_size=11, color=ORANGE,
            bold=True, font_name="Courier New")

add_textbox(s15, Inches(0.67), Inches(1.3), Inches(10), Inches(0.8),
            "Feedback that would help",
            font_size=36, color=ORANGE, bold=True)

add_textbox(s15, Inches(0.67), Inches(2.7), Inches(3), Inches(0.3),
            "01 / RESEARCH DIRECTION", font_size=11, color=ORANGE,
            bold=True, font_name="Courier New")
add_textbox(s15, Inches(0.67), Inches(3.1), Inches(11), Inches(0.7),
            "Stay on memory poisoning with gpt-oss, or move to the full coding agent testbed now?",
            font_size=17, color=WHITE, space_after=4)

add_textbox(s15, Inches(0.67), Inches(4.0), Inches(3), Inches(0.3),
            "02 / EXPERIMENT DESIGN", font_size=11, color=ORANGE,
            bold=True, font_name="Courier New")
add_textbox(s15, Inches(0.67), Inches(4.4), Inches(11), Inches(0.7),
            "Which baseline should we pre-register as the bar to beat: "
            "full prompt TF-IDF, embeddings, or an LLM judge?",
            font_size=17, color=WHITE, space_after=4)

add_textbox(s15, Inches(0.67), Inches(5.3), Inches(3), Inches(0.3),
            "03 / SUPPORT NEEDED", font_size=11, color=ORANGE,
            bold=True, font_name="Courier New")
add_textbox(s15, Inches(0.67), Inches(5.7), Inches(11), Inches(0.7),
            "GPU access (NCSA or a shared A100) for about 6 GPU hours by mid November.",
            font_size=17, color=WHITE, space_after=4)

add_textbox(s15, Inches(0.67), Inches(7.05), Inches(10), Inches(0.35),
            "FINAL CHECKPOINT  /  DORMANT SLEEPER DETECTION",
            font_size=10, color=LIGHT_GRAY, font_name="Courier New")
add_textbox(s15, Inches(12.5), Inches(7.05), Inches(0.7), Inches(0.35),
            "15", font_size=11, color=ORANGE, bold=True,
            font_name="Courier New", alignment=PP_ALIGN.RIGHT)


# ═══════════════ SLIDE 16: Appendix / Reproducibility ═══════════════
s16 = prs.slides.add_slide(blank_layout)
add_org_label(s16)
add_category_label(s16, "APPENDIX  /  OPTIONAL")
add_title(s16, "Setup and reproducibility", size=32)

add_section_header(s16, "SETUP", top=Inches(2.5))
add_textbox(s16, Inches(0.67), Inches(2.9), Inches(5.5), Inches(2.0), [
    "• Model & Harness: gpt-oss-20b (25 layers, 32 MoE experts, 2880 dim); "
    "bfloat16 forward pass with 93–100% rollout agreement.",
    "• Probing: L2 logistic regression on mean of residual layers 14–19 "
    "at memory_result_end. C chosen by leave-one-template-out CV.",
    "• Activations from gpt-oss-20b in bf16; behavior from vLLM 0.30.0 MXFP4. "
    "k = 8 samples per note, T = 1.0, fired = at least 4 of 8."
], font_size=13, color=NAVY, space_after=5)

add_section_header(s16, "EVIDENCE / REPRODUCIBILITY", left=Inches(7.0), top=Inches(2.5))
add_textbox(s16, Inches(7.0), Inches(2.9), Inches(5.5), Inches(2.0), [
    "• Code, reports, features and labels on GitHub "
    "(RaghavAgarwal08/sleeper_detection, branch raghav/probe-live-inert).",
    "• Controls rerun on a laptop in 2 minutes.",
    "• Live Benchmark Site: pndh.github.io/sleeper_detection/"
], font_size=13, color=NAVY, space_after=5)

# Contact
add_textbox(s16, Inches(7.0), Inches(5.0), Inches(5.5), Inches(0.5),
    "Contact: Raghav Agarwal (raghava7@illinois.edu) · Huy Pham (huynp2@illinois.edu)",
    font_size=14, color=ORANGE, bold=True, space_after=4)

# References
add_section_header(s16, "REFERENCES / SOURCE", top=Inches(5.7))
add_textbox(s16, Inches(0.67), Inches(6.1), Inches(12.0), Inches(0.8), [
    "Li et al. 2026, Sleeper Attack benchmark  ·  Hung et al. 2024, Attention Tracker  ·  "
    "Zou et al. 2023, Representation Engineering  ·  Hubinger et al. 2024, Sleeper Agents  ·  "
    "Ruan et al. 2026, GhostWriter  ·  Abdelnabi et al. 2025, TaskTracker  ·  "
    "MacDiarmid et al. 2024, Production-Ready Probes for Gemini",
], font_size=12, color=NAVY, space_after=3)

add_footer(s16)
add_slide_number(s16, 16)


# ════════════════════════════════════════════════════════════════
#  SAVE
# ════════════════════════════════════════════════════════════════

out_pptx = OUT_DIR / 'AIA_Final_Checkpoint_Sleeper_Detection.pptx'
prs.save(str(out_pptx))
print(f"✓ Saved: {out_pptx}")

# Also save copies
legacy_name = OUT_DIR / 'AIA Presentation slides.pptx'
shutil.copy(out_pptx, legacy_name)
print(f"✓ Copied to: {legacy_name}")

# Copy to artifacts
os.makedirs(ARTIFACT_DIR, exist_ok=True)
shutil.copy(out_pptx, ARTIFACT_DIR / out_pptx.name)
print(f"✓ Copied to artifacts: {ARTIFACT_DIR / out_pptx.name}")

print("\nDone! Run process_slides.py to export to PDF and PNG.")
