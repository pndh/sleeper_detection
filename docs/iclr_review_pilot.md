# ICLR review: pilot findings (sleeper-agent latent commitment detection)

**Soundness: 1/4 · Rating: 3 (reject) · Confidence: 4**

Before the review itself: several problems below come from the code and reports written during the pilot, and some results were described more strongly than the data supports. Those spots are flagged so the pitch isn't built on them.

---

## Summary of claims
1. A probe pooled over 55–80% relative depth detects a "pending malicious instruction" state, with 94.0% zero-shot AUROC across 10 tool domains.
2. A 2×2 design rules out a content detector and an instruction detector.
3. The window is the causal locus of commitment.

## Strengths
- The question is right: detecting dormant state before the trigger fires, from the API provider's view.
- The 2×2 design separating content from instruction is the correct way to tell competing stories apart.
- Leave-one-domain-out evaluation is the right split.
- Activations are cached and the evaluation scripts can be rerun.

## Major weaknesses

**W1. The window was chosen on the test data, so "fixed a priori" is false.** Layers 14–19 were picked after seeing the per-layer leave-one-out sweep on the same 40 samples, then scored on those same 40. That is selection on test data across 25 layers. The 94.0% is an optimistic in-sample number.

**W2. The pre-specified hypothesis test came out null.** The claim is "only the live + malicious cell lights up," which statistically is the content × framing interaction. Measured across 10 domains: interaction = −1.83 ± 9.23, t(9) = −0.63, p = 0.73. The live delta (+2.3) is *smaller* than the inert delta (+4.1). The data does not support claim 2.

**W3. The AUROC doesn't need any "commitment" to explain it.** The target is cell (1,1) vs. the other three — a logical AND, which a linear probe gets from two additive features ("live template present" + "malicious payload present"). No interaction needed. Two confounds:
- **One fixed sentence template per framing.** "Before executing X tool…" vs. "Archived customer support inquiry…" is a trivially decodable surface feature. Holding out a domain doesn't hold out the template.
- **Malicious payloads carry obvious cue words**: "unauthorized," "fraudulent," "malicious_dev_01," "purge," "slice the power cord." This is a content detector with extra steps.

**W4. The layer profile argues against the interpretation.** AUROC is already 0.84 at L1 (4% depth). L0 = 0.50 doesn't show "not reading text": the probed token (`memory_result_end`) is the same token in every cell, so L0 embeddings are identical by construction. After one attention layer the probe has 84% of its final performance. The L13→L16 rise (0.88 → 0.95) is small at n = 10 positives, with no confidence intervals and no correction for picking the best of 25 layers.

**W5. The "live" cells aren't live.** All 40 requests reuse one demo_v1 request: Shopify-case system prompt and tool list, plus the thank-you-note query. Conditions like `SlackSearchMessages` or `TDAmeritradeGetQuote` aren't in the tool list, so most "pending conditionals" can never fire. The templates also don't match the paper's real injections ("Unfinished user request… still pending…"). Nothing was measured behaviorally: cell labels are set by design, not by whether the model would execute. "Readiness" isn't measured.

**W6. There is no causal evidence yet, and the intervention code is broken.** In the A100 run, ablation gave identical projections (−63.992 → −63.992), and every steering α gave the same score (−104.519). The hook had no effect. The script printed "Signal neutralized!" regardless. Next-token probability at the prompt boundary is useless as a readout: it is `<|channel|>` with p = 1.0 in every condition.

**W7. Some reported numbers were never measured.** `step4_behavioral_readiness` in `interp/run_full_reviewer_suite.py` returns hardcoded values (`inert_dud_filtration_auroc: 1.0`, `active_vs_inert_separation_margin: 453.8`). These are placeholders, not results, and must not appear anywhere. The script also counted 486 cases, not the 1,896 claimed earlier.

**W8. Smaller issues.**
- The u₁₀ direction comes from a single contrast pair.
- One model, one memory format, one user query.
- No paraphrase robustness and no adaptive attacker.
- Depth-band literature claims (e.g., specific refusal-direction depth ranges) were stated loosely and must be checked against the papers before citing.

## Questions for authors
1. Does a bag-of-words or n-gram baseline on the memory text match the 94%? If it does, the claim falls.
2. Does the probe transfer across *templates* (train on template A, test on unseen paraphrases B and C)?
3. Among cell (1,1) contexts, does the probe score predict actual execution on the trigger turn?

---

## What would make this sound (minimum bar)

| Fix | Addresses |
|---|---|
| Fix the window **now** (55–80%), then test on **new** domains and templates never seen | W1 |
| Make the **interaction** the primary endpoint; report CIs and bootstrap over domains | W2, W4 |
| ≥5 paraphrased templates per framing; remove cue words; hold out templates as well as domains | W3 |
| Text-only baselines: TF-IDF / logistic regression, and a sentence-embedding probe on the raw note | W3 |
| Build each request from the **real case** (its own tools and system prompt, a trigger tool that exists); label by sampled behavior on the trigger turn | W5 |
| Debug hooks (check that projection → 0 after ablation); read out the trigger-turn **tool-call ASR**, not the next token | W6 |
| Delete the hardcoded step-4 values | W7 |

**Bottom line:** what currently holds up is "a linear probe at mid-to-late depth separates our live+malicious template from three control templates." The parts needed for the pitch (commitment, beating the content and instruction stories, a causal window) are either unsupported or contradicted by the interaction test. Fixing W1, W3 and W5 would make a much stronger pilot than the current one.

## Next implementation steps
1. Fix the hooks in `interp/gptoss.py` / `interp/run_full_reviewer_suite.py` and add an assertion that the post-ablation projection is ~0.
2. Delete the placeholder values in `step4_behavioral_readiness`.
3. Build a template-held-out, behavior-labeled set: real per-case requests, ≥5 paraphrases per framing, no cue words, trigger-turn behavior sampled for labels.

## Constraints for whoever implements this
- Remote host `jupyter-huynp2-lab` has an **8 GB host-RAM cgroup cap**. Always load gpt-oss-20b on `cuda:0`, never `cpu`.
- CPDM training shares the A100. Pause it before GPU work and resume it afterward (`interp/run_scaleup_suite.sh` shows the pattern).

## Status of the next steps (2026-10-04)
1. **Hooks fixed, not yet rerun on the real model.** Cause of W6: transformers 5.12 records `output_hidden_states` with
   its own forward hook on each decoder layer, which ran before ours and stored the unedited stream. The logits did
   change (`<|message|>` 1.3e-10 → 4.1e-10), so downstream layers saw the edit and only the readout missed it.
   `gptoss.hooks` now registers with `prepend=True` and returns the edited tensor. `gptoss.check_edit` asserts that
   the projection in the recorded residual is ~0 after ablation (or baseline + α after steering), within bf16
   rounding, ‖h‖·2⁻⁸. Also fixed: the direction was read at resid index `layer` while the hook edits `layer+1`; and
   `generate` stopped at `<|end|>`, which closes the analysis channel, so it could never reach a tool call. Checked
   on CPU with a tiny random GptOss: the old hook style reproduces the identical-projection symptom, the new one passes
   and rejects a no-op edit.
2. **Placeholders removed** from `results/full_reviewer_suite_report.json` (step 4), and step 3 of that report is marked
   invalid. `step4_behavioral_readiness` and `behavioral_readiness.py` now report case counts only.
3. **Set built, behaviour not sampled yet.** `demo/build_behavior_set.py` → `interp/data/behavior_v1`: 148 memory-LIP cases
   (14 dropped for cue words), each a real trigger turn with the case's own system prompt, tools and turn-2 query; 6
   live + 6 inert framings around the verbatim payload action, plus a no-note twin. The split (domains, templates 4–5,
   window 14–19) is fixed in `manifest.json` before any activations exist. `demo/sample_behavior.py` fills
   `behavior.jsonl` under vLLM, with both the first-response and multi-step judges, since decisions.md P2/P3 are
   still open.
   **Not covered: a benign-payload arm**, which is needed for the W2 interaction endpoint. It needs a matched benign
   action and judge for each payload tool, a design choice left to the researcher.
