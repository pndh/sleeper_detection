# Proposal v2: Trusted-channel dormant poisoning of agent memory

Written for: the owner (and advisor / external readers of a short proposal). Replaces the "internals detector"
framing. Rationale and measurements are from this repo's recent work; verify the two competitor papers in full
before committing (they are strong and recent).

## 0. What changed, and why (the honest reckoning)

The original thesis — an activation probe that detects dormant payloads — is dead, and now it is dead *with a
number*:

- On the 40-note `sound_2x2` benchmark, **Jev (TypeSafe's System-One guardrail, one ~200 ms / $0.00002 API call)
  scores AUROC 0.962** for "live malicious note vs the rest", beating our activation probe (0.84) and a TF-IDF text
  baseline (0.87). For payloads whose signal is in the text, the guardrail wins. Internals add nothing.
- Jev's one weakness is the same as everyone's: the **malicious-vs-benign payload axis is only 0.72**, and its
  misses are the **benign-phrased** payloads ("move file", "add to cart", "buy stock"). Semantic indistinguishability
  is the residual hard core.
- This benchmark always places the note in the request, so Jev can read it. The genuinely hard case is **cross-
  session dormancy**, which this data never tests.

So the open problem is not detection of harmful-looking text. It is memory whose harm is not in its text and whose
origin looks legitimate.

## 1. The landscape (three strong 2026 works bracket the problem)

| Work | What it does | Where it stops |
|---|---|---|
| **Jev / System-One guards** | fast, stateless, per-request content classification | blind to benign-phrased payloads; stateless, never sees cross-session plant |
| **ShadowMem (CCS'26)** | stateful safety memory over the *current* trajectory; vetoes risky tool calls | explicitly excludes **memory poisoning**; assumes full-trajectory visibility and a **trusted** shadow memory |
| **TMA-NM (origin-bound authority)** | binds memory authority to the channel it arrived on; non-malleable; 0% ASR incl. dormant triggers | guarantees hold "modulo correct origin labeling"; **does not block malicious content arriving through a trusted channel**; implicit reconstruction and answer-biasing are stated gaps |

The clean gap all three leave open: **a payload that enters through a channel the system trusts.** In the Sleeper
Attack benchmark the payload is planted by the agent's *own* `MemoryUpdate` during an earlier, ostensibly
legitimate turn. Origin-binding labels that *trusted*. Jev sees no harmful words. ShadowMem never saw the plant.
It fires sessions later on a benign query.

## 2. Research question

> When a dormant payload enters an agent's long-term memory through a **trusted channel** (the user's own earlier
> request, or the agent writing to its own memory), content guards see no harmful text and origin-based provenance
> labels it authorized. **What signal distinguishes it from a legitimate saved task before it fires — and can a
> defense bind memory authority to the user *intent/consent* at write time rather than to the channel?**

This is sound because the answer is valuable either way: if nothing distinguishes it, trusted-channel poisoning is a
fundamental limit worth stating; if intent-binding works, it closes TMA-NM's named gap.

## 3. Why this is both a product and a research contribution

- **Research:** it is the explicit open gap of the strongest prior work (TMA-NM's untested "authorized-channel
  poisoning"; ShadowMem's excluded "memory poisoning"), evaluated on the realistic plant→dormant→trigger chain with
  **behaviour labels** (did the agent actually execute), which none of the three use end-to-end. Fits exactly what
  CCS'26 is rewarding: long-horizon agent threats, memory security, provenance/governance.
- **Product:** a **memory-authority / consent layer** for agent state — "a memory entry can only drive a consequential
  action if a real user turn authorized *that* action, not merely because it sits in trusted memory." This is the
  agent-identity / runtime-governance market (Alter, Prisma AIRS, etc.), and the moat is a systems+data property that
  does not erode as base models improve. It is not a classifier that a better LLM replaces.

## 4. Contributions

1. **Benchmark of the kill chain, behaviour-labelled.** Extend the existing testbed (`interp/data/behavior_v1`,
   1,924 requests, held-out domains+templates) with cross-session plant→dormant→trigger episodes and a
   trusted-channel plant (self-`MemoryUpdate`). Label by sampled execution, not by design.
2. **A measurement across the three defense classes** — content guard (Jev, measured via OpenRouter Decisions API),
   trajectory guard (ShadowMem, open code), origin-binding (a TMA-NM-style labeler) — showing where each breaks on
   trusted-channel dormant poisoning. This head-to-head does not exist today.
3. **Intent/consent-bound memory authority (the defense wedge):** at write time, bind a memory entry to the specific
   user intent that authorized it; at trigger time, gate the memory-driven action on whether any real user turn
   authorized *that action*. Catches the case channel-provenance misses.

## 5. Method and evaluation (reuses what exists)

- Model: gpt-oss-20b under vLLM; the paper's `Evaluator` as the execution oracle; Jev via `api/alpha/decisions`.
- Episodes: real cases with a trusted-channel plant turn, N benign turns, a trigger turn; the benign-payload control
  arm from §1 for false positives.
- Metrics: attack success (execution) and benign utility per defense; **TPR at 1%/5% FPR** on legitimate saved tasks;
  the executed-but-unflagged region as a function of plant realism.
- Decision gates: (0) does intent-binding add over origin-binding on trusted-channel plants? if no → report the limit;
  (1) does it hold across states/sources/models? (2) false-positive cost on real standing user tasks.

## 6. Honest risks

- **Novelty vs TMA-NM is the make-or-break.** Read it in full first. If "intent-binding" collapses into their
  "fresh, action-bound user authorization", the contribution is the *measurement + the trusted-channel attack*, not
  a new defense — still publishable, but scope it honestly.
- Intent/consent capture may need user-in-the-loop friction; quantify it.
- Jev and ShadowMem are moving targets; pin versions and dates.
- Measurement, not an open-ended attack generator: keep the trusted-channel plants scoped to this benchmark.
