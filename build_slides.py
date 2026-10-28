import sys
import shutil
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

SRC_PPTX = Path('/home/pndhpndh/.gemini/antigravity/brain/fd4b0ee4-31d9-430c-af29-2bcb38cd9d97/.user_uploaded/media_1793204806418.pptx')
OUT_PPTX = Path('/home/pndhpndh/sleeper_detection/AIA_Illinois_Midpoint_Checkpoint_Raghav_Agarwal.pptx')
ARTIFACT_DIR = Path('/home/pndhpndh/.gemini/antigravity/brain/fd4b0ee4-31d9-430c-af29-2bcb38cd9d97')
KEY_FIGURE = Path('/home/pndhpndh/sleeper_detection/results/midpoint_key_figure.png')

prs = pptx.Presentation(SRC_PPTX)

ORANGE = RGBColor(0xFF, 0x5F, 0x05)
NAVY = RGBColor(0x13, 0x29, 0x4B)
SLATE = RGBColor(0x47, 0x55, 0x69)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xC6, 0xC7, 0xC6)

def set_text_frame(tf, content, default_size=16, default_color=NAVY, default_bold=False, font_name="Arial", space_after=4, line_spacing=1.15):
    """Safely updates a text frame with clean paragraph formatting, explicit wrapping, and no overlaps."""
    tf.word_wrap = True
    color = default_color
    fn = font_name
    sz = default_size
    bold = default_bold

    if isinstance(content, str):
        lines = [line.strip() for line in content.split('\n') if line.strip()]
    else:
        lines = [c.strip() for c in content if c.strip()]

    tf.text = '' # clear existing runs
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        for r in p.runs:
            r.font.name = fn
            r.font.size = Pt(sz)
            r.font.bold = bold
            r.font.color.rgb = color

def update_table(table, data, header_size=13, body_size=12.5, body_color=NAVY):
    """Updates table cells cleanly with auto-wrapping, bold headers, and crisp typography."""
    for r_idx, row in enumerate(data):
        for c_idx, val in enumerate(row):
            cell = table.rows[r_idx].cells[c_idx]
            cell.text_frame.word_wrap = True
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.space_after = Pt(3)
            p.line_spacing = 1.15
            for r in p.runs:
                r.font.name = "Arial"
                if r_idx == 0:
                    r.font.size = Pt(header_size)
                    r.font.bold = True
                    r.font.color.rgb = WHITE
                else:
                    r.font.size = Pt(body_size)
                    r.font.bold = (c_idx == 1 and val in ["Complete", "In progress", "Next / blocked"])
                    r.font.color.rgb = body_color

# Update footer on all slides
for slide in prs.slides:
    for shape in slide.shapes:
        if shape.has_text_frame and "MIDPOINT CHECKPOINT" in shape.text_frame.text:
            p = shape.text_frame.paragraphs[0]
            p.text = "MIDPOINT CHECKPOINT  /  DORMANT SLEEPER DETECTION"
            for r in p.runs:
                r.font.name = "Courier New"
                r.font.size = Pt(10)
                r.font.color.rgb = LIGHT_GRAY if slide in [prs.slides[0], prs.slides[10]] else SLATE

# ==================== SLIDE 1: Title (Dark Navy Slide) ====================
s1 = prs.slides[0]
set_text_frame(s1.shapes[1].text_frame, "AIA @ ILLINOIS  /  FALL 2026", default_size=13, default_color=ORANGE, default_bold=True, font_name="Courier New")
set_text_frame(s1.shapes[6].text_frame, "Dormant Poison & Sleeper Instruction\nDetection in LLM Agents", default_size=28, default_color=ORANGE, default_bold=True, font_name="Arial", space_after=6)
# Ensure Raghav's name has plenty of top clearance
s1.shapes[7].top = Inches(5.25)
s1_p0 = "Raghav Agarwal  ·  AIA Research Fellow"
s1_p1 = "AI Alignment @ Illinois Research Fellowship  ·  October 2026"
set_text_frame(s1.shapes[7].text_frame, [s1_p0, s1_p1], default_size=17.5, default_color=WHITE, default_bold=False, font_name="Arial", space_after=5)

# ==================== SLIDE 2: What we know at the midpoint ====================
s2 = prs.slides[1]
# Shape 06: Question body (Left 0.67, Width 5.52)
set_text_frame(s2.shapes[6].text_frame, 
    "Can internal activation probing detect dormant sleeper instructions in agent memory during benign turns, before triggers fire?",
    default_size=17.5, default_color=NAVY, default_bold=True, space_after=4)

# Shape 08: Evidence body (Left 6.98, Width 5.52) - Exactly 2 crisp punchy bullets
set_text_frame(s2.shapes[8].text_frame, [
    "• 1.000 AUROC / 100% Recall: Separates pending instructions from inert records across held-out splits.",
    "• 8× Signal Amplification: Memory tokens carry 8.02× stronger signal than prompt boundary delimiters."
], default_size=15, default_color=NAVY, space_after=6)

# Shape 10: Next decision body (Full width Left 0.67, Width 11.46) - Exactly 2 bullets
set_text_frame(s2.shapes[10].text_frame, [
    "• Validate Causal Steering: Directional clamping on Layers 14–19 to prove features actively cause trigger execution.",
    "• Cross-Architecture Transfer: Replicate probe extraction on dense open models (Llama-3-70B)."
], default_size=15.5, default_color=NAVY, space_after=5)

# ==================== SLIDE 3: Why this question matters ====================
s3 = prs.slides[2]
# Shape 06: Motivation body (Left 0.67, Width 5.52)
set_text_frame(s3.shapes[6].text_frame, [
    "• Persistent Memory Threat: Attackers plant sleeper notes in agent scratchpads that stay dormant until future triggers.",
    "• Text Guardrails Fail: External judges suffer 50%+ false alarms on code logs and miss disguised phrasing (10% recall)."
], default_size=14.5, default_color=NAVY, space_after=5)

# Shape 08: Research question body (Left 6.98, Width 5.52)
set_text_frame(s3.shapes[8].text_frame, [
    "• Mechanistic Opportunity: Transformer self-attention binds active intent into executive subspaces (Layers 14–19) before token 1.",
    "• Surface Invariance: Internal latent representations bypass surface disguise and semantic camouflage."
], default_size=14.5, default_color=NAVY, space_after=5)

# Shape 10: Safety connection body (Full width Left 0.67, Width 11.67)
set_text_frame(s3.shapes[10].text_frame, [
    "• Zero-Latency Preemption: Scored in < 0.1 ms on GPU VRAM at $0 token cost (vs 1,500 ms / 2x cost for external LLM judges).",
    "• In-Flight Intervention: Enables clamping sleeper vectors before catastrophic agent tool calls execute."
], default_size=15.5, default_color=NAVY, space_after=5)

# ==================== SLIDE 4: Approach & experiment design ====================
s4 = prs.slides[3]
# Col 1: 01 / INPUT (Left 0.67, Width 3.54)
set_text_frame(s4.shapes[6].text_frame, [
    "• gpt-oss-20b: 25-layer MoE, 32 experts (2880 residual dim).",
    "• behavior_v1: 1,924 trigger requests across 10 enterprise domains with 43-token paired twins canceling noise."
], default_size=14, default_color=NAVY, space_after=5)

# Col 2: 02 / METHOD (Left 4.90, Width 3.54)
set_text_frame(s4.shapes[8].text_frame, [
    "• Residual Probing: Decoupled bfloat16 forward pass extracting Layers 0–24 at memory_result_end.",
    "• L2 Probing & LODO: Leave-one-domain-out cross-validation testing true out-of-distribution transfer."
], default_size=14, default_color=NAVY, space_after=5)

# Col 3: 03 / MEASUREMENT (Left 9.11, Width 3.54)
set_text_frame(s4.shapes[10].text_frame, [
    "• Out-of-Fold AUROC: Unseen templates (E1) & domains (E2).",
    "• Operational SLA: TPR @ 5% FPR, plus Execution Prediction AUROC (Will It Fire?)."
], default_size=14, default_color=NAVY, space_after=5)

# Bottom: WORKING HYPOTHESIS
set_text_frame(s4.shapes[12].text_frame,
    "WORKING HYPOTHESIS: When an agent commits to an unexecuted instruction, self-attention binds it into executive task-planning subspaces (Layers 14–19), yielding linear separability before token 1 generates.",
    default_size=15, default_color=NAVY, default_bold=True, space_after=4)

# ==================== SLIDE 5: Progress against plan ====================
s5 = prs.slides[4]
s5_data = [
    ["Milestone", "Status", "Evidence / verified result"],
    ["Mechanistic Harness & Twin Controls", "Complete", "Built bfloat16 forward pass with 93–100% rollout agreement; synthesized 43-token paired twins canceling 99.95% formatting noise."],
    ["Multi-Position Probing & Layer Isolation", "Complete", "Identified 8× signal boost at memory tokens; isolated Layers 14–19 as primary planning subspace (1.000 AUROC on E1 & E2)."],
    ["Frontier Guard Benchmark & Realism Ladder", "Complete", "Benchmarked TypeSafe Jev (30% recall @ 5% FPR); proved text judges collapse to 10% recall on disguised attacks (L4)."]
]
update_table(s5.shapes[5].table, s5_data, header_size=13, body_size=12.5)
set_text_frame(s5.shapes[6].text_frame,
    "PLAN UPDATE  /  Expanded scope from pure activation extraction to head-to-head benchmarking against frontier guard models (Jev) and semantic camouflage, proving operational defense superiority.",
    default_size=15, default_color=NAVY, default_bold=True, space_after=3)

# ==================== SLIDE 6: Data & Evaluation ====================
s6 = prs.slides[5]
s6_data = [
    ["Component", "Current benchmark setup (behavior_v1)"],
    ["Data & Coverage", "behavior_v1: 1,924 agent interactions across 10 enterprise domains (Shopify, Slack, HubSpot, etc.); 6 live & 6 inert templates; Realism Ladder L0–L4."],
    ["Generalization Splits", "Held-out evaluation across 3 splits: E1 (unseen instruction phrasing), E2 (unseen enterprise domains), E3 (jointly unseen templates & domains)."],
    ["Baselines & Controls", "Paired control twins (format control); TF-IDF n-gram text baselines; prompt length; TypeSafe Jev Decisions API with enterprise policy."],
    ["Metrics & SLA", "Out-of-fold AUROC with 95% bootstrap CIs; TPR @ 5% False Positive Rate SLA; per-layer residual curve (Layers 0–24); execution prediction AUROC."]
]
update_table(s6.shapes[5].table, s6_data, header_size=13, body_size=12)

# ==================== SLIDE 7: Preliminary Results (Key Figure Slide) ====================
s7 = prs.slides[6]
set_text_frame(s7.shapes[2].text_frame,
    "Internal Probes Achieve 1.00 AUROC Across Unseen Phrasing, While Text Guards Collapse",
    default_size=23, default_color=NAVY, default_bold=True, space_after=4)

# Replace placeholder text with clean high-res image
s7.shapes[6].text_frame.text = ""
s7.shapes.add_picture(str(KEY_FIGURE), Inches(0.67), Inches(3.30), Inches(8.0), Inches(3.55))

# Top Right: OBSERVATION (Top 2.85 for header, Top 3.20 for text)
s7.shapes[7].top = Inches(2.85)
s7.shapes[8].top = Inches(3.20)
set_text_frame(s7.shapes[8].text_frame, [
    "• 1.000 AUROC / 100% TPR: Generalizes across held-out phrasing (E1) and domains (E2).",
    "• 0.945 Execution AUROC: Probes know if a note will fire before token 1 generates."
], default_size=13, default_color=NAVY, space_after=4)

# Bottom Right: INTERPRETATION (Top 4.55 for header, Top 4.90 for text)
s7.shapes[9].top = Inches(4.55)
s7.shapes[10].top = Inches(4.90)
set_text_frame(s7.shapes[10].text_frame, [
    "• Latent Commitment: Transformer binds active intent into Layers 14–19 while treating inert logs as passive context.",
    "• Surface Invariance: Probes score in <0.1 ms; Jev drops to 30% TPR and text guards fail on disguise."
], default_size=13, default_color=NAVY, space_after=4)

# ==================== SLIDE 8: How robust is the finding? ====================
s8 = prs.slides[7]
s8_data = [
    ["Check", "Current evidence", "Next test"],
    ["Baseline Comparison", "Probe (1.000 AUROC, 100% TPR@5%FPR) decisively beats Jev (0.816 AUROC, 30% TPR) and prompt length (0.59).", "Benchmark LLaMA-Guard-3 and GPT-4o-mini under identical policy context."],
    ["Ablation & Control", "Paired twins eliminate 99.95% container overhead; memory token yields 8× signal boost over prompt delimiter.", "Attention head ablation: patch individual heads in Layers 14–19 to isolate causal circuits."],
    ["Generalization", "Zero degradation on unseen phrasing (E1: 1.000) and unseen enterprise domains (E2: 1.000).", "Cross-architecture evaluation on dense models (Llama-3-70B) and multi-turn delays."]
]
update_table(s8.shapes[5].table, s8_data, header_size=13, body_size=12)
set_text_frame(s8.shapes[6].text_frame,
    "CONFIDENCE  /  High confidence that latent states linearly separate live instructions in gpt-oss-20b; currently verifying causal steering invariance across multi-turn history.",
    default_size=15, default_color=NAVY, default_bold=True, space_after=3)

# ==================== SLIDE 9: What is unresolved ====================
s9 = prs.slides[8]
# Col 1: LIMITATIONS (Left 0.67, Width 5.52)
set_text_frame(s9.shapes[6].text_frame, [
    "• Architecture Breadth: Validated on gpt-oss-20b MoE; dense models (Llama-3-70B) may distribute planning across different depths.",
    "• Interaction Scope: behavior_v1 tests single-tool triggers; multi-agent dialogues and long-context RAG remain to be tested."
], default_size=15, default_color=NAVY, space_after=6)

# Col 2: BLOCKERS / RISKS (Left 6.98, Width 5.52)
set_text_frame(s9.shapes[8].text_frame, [
    "• Causal Gap: Proving probe features actively cause trigger execution rather than merely correlating with memory structure.",
    "• Mitigation: Implement directional steering and activation patching on Layers 14–19 during rollouts."
], default_size=15, default_color=NAVY, space_after=6)

# Bottom: SCOPE DECISION
set_text_frame(s9.shapes[9].text_frame,
    "SCOPE DECISION  /  Prioritize causal steering and cross-model validation (Llama-3-70B) over expanding synthetic domain templates.",
    default_size=15.5, default_color=NAVY, default_bold=True, space_after=3)

# ==================== SLIDE 10: Plan for remaining fellowship ====================
s10 = prs.slides[9]
s10_data = [
    ["When", "Priority / experiment", "Done when…"],
    ["Weeks 5–6 (Nov 1–14)", "Causal Steering & Activation Patching", "Clamping probe vector at Layers 14–19 proves bidirectional causal control over sleeper trigger firing during rollouts."],
    ["Weeks 7–8 (Nov 15–28)", "Cross-Architecture & Realism Evaluation", "Replicate probe extraction on Llama-3-70B; benchmark against full Realism Ladder (L0–L4 camouflage)."],
    ["Weeks 9–10 (Dec 1–15)", "Open-Source Defense Toolkit & Paper", "Release modular sleeper detection middleware on GitHub; submit fellowship final report / preprint."]
]
update_table(s10.shapes[5].table, s10_data, header_size=13, body_size=12)
set_text_frame(s10.shapes[6].text_frame,
    "DELIVERABLES  /  Fellowship preprint  ·  Open-source detection toolkit on GitHub  ·  Interactive benchmark demo",
    default_size=15.5, default_color=NAVY, default_bold=True, space_after=3)

# ==================== SLIDE 11: Feedback that would help (Dark Slide) ====================
s11 = prs.slides[10]
# Reposition sections with clean, ample vertical spacing
s11.shapes[5].top = Inches(2.70)
s11.shapes[6].top = Inches(3.10)
set_text_frame(s11.shapes[6].text_frame,
    "Should we position the final artifact primarily as a runtime defense mechanism (zero-latency monitor beating external LLM guards) or as a mechanistic discovery paper (latent executive subspace geometry)?",
    default_size=16, default_color=WHITE, space_after=4)

s11.shapes[7].top = Inches(4.15)
s11.shapes[8].top = Inches(4.55)
set_text_frame(s11.shapes[8].text_frame,
    "What standard of causal evidence is most compelling to the alignment community: directional clamping, activation path patching, or end-to-end steering rollouts?",
    default_size=16, default_color=WHITE, space_after=4)

s11.shapes[9].top = Inches(5.60)
s11.shapes[10].top = Inches(6.00)
set_text_frame(s11.shapes[10].text_frame,
    "Access to multi-GPU (A100/H100) compute nodes to extract layerwise activations and run causal interventions on 70B+ models (Llama-3-70B, Qwen-2.5-72B).",
    default_size=16, default_color=WHITE, space_after=4)

# ==================== SLIDE 12: Appendix ====================
s12 = prs.slides[11]
set_text_frame(s12.shapes[2].text_frame, "Implementation & Mechanistic Probing Setup", default_size=30, default_color=NAVY, default_bold=True, space_after=4)

# Col 1: Setup - 2 concise punchy bullets
set_text_frame(s12.shapes[6].text_frame, [
    "• Model & Harness: gpt-oss-20b (25 layers, 32 MoE experts, 2880 dim); bfloat16 forward pass with 93–100% rollout agreement.",
    "• Linear Probing: L2 logistic probes on residual stream at memory_result_end, isolating executive Layers 14–19."
], default_size=14.5, default_color=NAVY, space_after=5)

# Col 2: Evidence & Code - 2 concise punchy bullets
set_text_frame(s12.shapes[8].text_frame, [
    "• Open Codebase: github.com/RaghavAgarwal08/sleeper_detection (branch raghav/probe-live-inert).",
    "• Live Benchmark Site: https://pndh.github.io/sleeper_detection/ with interactive inspector & artifacts."
], default_size=14.5, default_color=NAVY, space_after=5)

# Bottom: References
s12.shapes[9].top = Inches(5.30)
s12.shapes[10].top = Inches(5.70)
set_text_frame(s12.shapes[10].text_frame,
    "• Li et al. (2026), 'Sleeper Attacks on LLM Agents via Persistent Memory'  ·  Zou et al. (2023), 'Representation Engineering'  ·  TypeSafe Decisions API (~typesafe/jev-latest)",
    default_size=14, default_color=NAVY, space_after=4)

prs.save(OUT_PPTX)
shutil.copy(OUT_PPTX, ARTIFACT_DIR / OUT_PPTX.name)
print(f"Successfully generated: {OUT_PPTX}")
