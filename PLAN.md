# PLAN.md — Something's Off: Detecting Dormant Poison in Already-Infected LLM Agents During Benign Interactions

> Implementation plan for Claude Code, rebuilt from the **14-week (one-semester) version** of the AIA Fellowship proposal (Huy Pham, Siebel School of CS; mentor: Enyi Jiang).
>
> **Supersedes the earlier PLAN.md**, which was built from an older draft that had no turn taxonomy, no Study 0, no model choices, and a different threat model.
>
> **Faithfulness rule.** Sections 1–11 restate the proposal; each notes its source section. Anything not in the proposal is confined to §12 (*Implementation notes — not from the proposal*). Known inconsistencies in the proposal are listed in §13 and must be resolved by the researcher, not by Claude Code.

---

## 0. Instructions for Claude Code

- Treat §1–§11 as the spec. Do **not** add methods, splits, metrics, baselines, or models that aren't listed. If something seems missing, add it to §13 and ask.
- Do not resolve items in §13 on your own. Ask the researcher.
- Never fabricate data, results, benchmark contents, or paper details. If an external resource can't be found or loaded, report it.
- The repo at `/home/huynp2/sleeper_detection` already has work in it (`agentdojo_gptoss/`, `data/`, `demo/`, `results/`, `docs/decisions.md`, `docs/sources.md`). Before building anything new, compare that work against this plan and report what matches, what conflicts, and what's missing.

---

## 1. Problem and motivation
*Source: Motivation / Problem*

LLM agents carry persistent state: conversation history, long-term memory stores, and reusable skill libraries. This state is an attack surface for delayed attacks.

- **Sleeper Attack** (Li et al., 2026): adversarial content persists in agent state, stays dormant, and is later activated by a benign user query. Benchmark: 1,896 instances; 6 harmful outcomes; 3 strategies (latent instruction planting, proactive information elicitation, persistent information corruption); 3 state targets (session context, memory, skills).
- **Sleeper memory poisoning** (Pulipaka et al., 2026): related threat targeting memory stores.
- **Explosive prompts** (Szczepaniak et al., 2026): conditional payloads in retrieved content acting as training-free, inference-time backdoors. Rephrasing goals as dormant conditionals raised real tool execution on frontier models from 2.4% to 16.5% across nine production coding agents; conditional triggers improve persistence over unconditional injections.

Shared structure: **Plant → Dormant → Trigger.** During the dormant phase the agent behaves normally, so output-level monitors have nothing to flag.

Internals-based monitoring is a candidate: linear probes catch weight-backdoored sleeper agents about to defect (Hubinger et al., 2024; MacDiarmid et al., 2024), and activation deltas reveal prompt-injection task drift (Abdelnabi et al., 2025). But dormant payloads are designed not to change current behavior, so it's open whether drift-based detectors see them at all. This project studies that gap.

---

## 2. Goal and research questions
*Source: Goal / Research Question*

**Overall goal.** Determine whether an agent with a dormant payload in its persistent state can be distinguished from a clean agent during benign interactions, using internal activations or observable behavior, and whether the poisoned entry can be localized and removed before the trigger fires. **Secondary:** evaluate detection and blocking at the trigger turn itself, before harmful tool execution.

**Primary research question.** When a dormant conditional payload is in context but its trigger is unmet, do the model's internal representations or behaviors differ measurably from an identical clean agent? Do the differences generalize across attack strategies, state targets, and models?

**Objectives** (Study 0 precedes Objective 1):

- **Objective 1 — Infected-agent testbed.** AgentDojo extended with memory and skill stores; dormant payloads ported from the Sleeper Attack benchmark and explosive-prompt attacks; every poisoned agent paired with a clean twin receiving identical benign queries.
- **Objective 2 — Detection during benign interactions.** Compare internals-based detectors (probes, activation directions, routing and attention statistics) and behavioral detectors (tool-use differences, reasoning-channel content) on benign turns. Test generalization to held-out strategies, state targets, and models. **Secondary:** detector performance at the trigger turn (Type D) and cross-type transfer between dormant and trigger turns.
- **Objective 3 — Localization, flagging, and blocking.** On a dormant-turn detection, identify the responsible span within the request and return it with an out-of-band flag. On a trigger-turn detection, block or sanitize the request before harmful tool execution. Measure drop in trigger-time attack success, benign utility preservation, and false-block rate on legitimate conditional requests.

The project takes the defender's perspective **after** infection. It does not study whether attacks get planted.

---

## 3. Threat model
*Source: Proposed Approach > Methods > Threat model*

- The agent's state (memory store, skill library, or session context) already contains an attacker-planted dormant payload.
- **The defender is the API provider** operating the model-serving layer. It observes only:
  - what arrives in each API request (full context, including any memories, skills, and tool definitions the client chose to include), and
  - the model's own intermediate activations and outputs on that request.
- The defender has **no access** to the client's stored state, application, or any history not included in the request.
- **gpt-oss-20b stands in for a provider-served model.**
- Ground-truth information (retrieval logs, clean twins, poison labels) is used **only for evaluation** and is never available to the defender.
- The attacker does not know the provider-level detector exists.

**Tiered response policy by turn type:**
- **Dormant benign turns (Types B, C):** detection raises an **out-of-band alert** with the localized span as metadata, **without interrupting** the response.
- **Trigger turns (Type D):** detection triggers **immediate blocking or response-prefilling intervention** to prevent harmful tool execution before completion.

---

## 4. Turn taxonomy
*Source: Methods > Turn taxonomy; Per-turn retrieval log; Scope & Out-of-Context Dormancy Handling*

Each turn is labeled by where the payload is relative to the model's context and whether the trigger is met (user assumed benign):

| Type | Definition | Role |
|---|---|---|
| **A** — payload out of context | Query doesn't retrieve the poisoned entry; payload never enters context that turn. Weights don't change, so no trace exists. | Negative control only; outside detection claims. |
| **B** — in context, trigger unmet | Poisoned memory/skill retrieved, or payload sits in session context / retrieved content, but query doesn't satisfy the trigger. | **Core setting** for the primary RQ. |
| **C** — near-trigger | Query semantically close to the trigger without satisfying it (e.g., asks about email without asking to send one). | Hardest test: does the detector respond to a pending conditional rather than surface content? |
| **D** — trigger met | Query satisfies the trigger. | **Secondary setting**: last-line detection and blocking before harmful execution. |

**Per-turn retrieval log.** For every turn, record whether the poisoned entry was retrieved or loaded into context. This assigns each turn its type and measures how often the payload is visible during benign use, per state target.

**Scope.** Primary focus is Types B and C. Type A is invisible to the provider by construction, so it's a negative control. Payloads that stay fully out of context until triggered are outside dormant-turn monitoring but are evaluated under trigger-turn detection (Type D) when loaded at the trigger turn. For these out-of-context cases the project:
1. reports their prevalence from the retrieval log;
2. uses them as negative controls (valid detectors should score near chance);
3. notes complementary defenses (stored-state auditing, trigger-time detection).

Stored-state auditing is the **client's** defense: mentioned, **not evaluated**.

**Main vs. validation settings.** Main experiments center on **memory and skill poisoning** (the realistic long-after-infection case). **Session context and explosive prompts** (payload continuously in context) are an easier setting for **validating the pipeline**.

---

## 5. Paired-twin design and data
*Source: Methods > Paired-twin design; Data / Materials > Benign data and hard negatives*

- Each poisoned agent has a clean twin whose state is identical except for the payload.
- Both twins run the same sequence of **1 to 20 benign queries** before any trigger.
- The environment is deterministic, so twin differences are attributable to the payload.
- The Sleeper Attack main setting triggers immediately after planting; benign interactions appear only in its separate longer-horizon experiment. This project **generates its own benign horizons** of 1–20 interactions.
- **Hard negatives:** legitimate user-written conditional memories and skills (e.g., "if I ask about invoices, CC my manager"), **plus their trigger-met counterparts** to measure false-block rates. Purpose: stop detectors from learning conditional phrasing or legitimate automated actions.
- **Target scale:** ~2,000 poisoned and ~2,000 clean multi-turn episodes **per model**.

---

## 6. Environment, models, and compute
*Source: Data / Materials; Feasibility and risks*

**Environment.**
- Unified under **AgentDojo** (deterministic, stateful Python environments: workspace, banking, travel, Slack; programmatic checks).
- Add a **memory store** and a **skill library** to AgentDojo agents.
- Port payloads: latent instruction planting, proactive information elicitation, persistent information corruption (Li et al., 2026), and conditional explosive-prompt payloads (Szczepaniak et al., 2026).
- Why not ToolEmu as the main harness: its LLM-emulated tool outputs vary between runs (noise unrelated to the payload) and cost API calls.
- **External-validity check:** run a subset of the original benchmark in its native **ToolEmu** harness with a **local emulator**.

**Models.**
- **Primary:** gpt-oss-20b (MoE, ~21B total / 3.6B active parameters).
- **Secondary:** Qwen3-8B; Qwen3-14B if compute allows. Tests cross-architecture and cross-scale transfer.
- Study 0 additionally uses **2–3 frontier APIs** (§7).

**Compute.**
- Free **NCSA A100** allocation.
- gpt-oss-20b's native MXFP4 kernels target newer GPUs, so expect to run **dequantized in bf16 at ~40 GB or more**. Prefer **80 GB A100**; **Qwen3-8B is the fallback** if only 40 GB nodes are available.
- NCSA access is inconsistent: all jobs **checkpointed and resumable**; **activations cached once**, probes trained on **CPU** afterward; development starts on **small models** before scaling.

**Budget ($1,000).** ~$350 LLM API credits (paraphrase diversity, hard-negative generation, LLM-judge baseline, ToolEmu external-validity check, Study 0); ~$100 activation-cache storage; ~$450 backup cloud GPU; ~$100 contingency.

**Candidate tooling:** nnsight or plain Hugging Face forward hooks for activation access; vLLM for fast rollouts when activations aren't needed; scikit-learn for probes; SAELens for the optional SAE.

---

## 7. Study 0 — Motivation study (runs before Objective 1)
*Source: Proposed Approach > Study 0*

Li et al. already show frontier models often execute planted redirections, especially corrupted facts. Study 0 asks two new questions:

- **(a) Awareness vs. compliance.** Does the reasoning trace flag anything suspicious, and does the model execute anyway? Key number: **"noticed but complied" rate.** A high rate suggests internal registration of the anomaly, motivating activation-based detection.
- **(b) Source trust.** The same redirection placed in four sources: **user-authored memory, tool-returned note, skill file, tool configuration** (tool configs are not among Li et al.'s state targets).

**Implementation:** reuse a **subset** of Sleeper Attack instances in their **ToolEmu** harness on **gpt-oss-20b, Qwen3, and 2–3 frontier APIs**. No new attack techniques. Reasoning traces labeled by an **LLM judge with manual spot checks**.

**Output:** table of attack success, suspicion rate, and noticed-but-complied rate by model and source type (≥3 models × 4 source types), reported regardless of outcome.

---

## 8. Detection methods (priority order)
*Source: Methods > Interpretability approach — tools to be finalized in Period 1*

1. **Linear probes.** Per-layer logistic regression on residual-stream activations at: end of retrieved state, tool-call tokens, and the final token before each response. On Type D, probes also read tool-call token activations to evaluate early intervention. **Cross-type transfer:** train on B/C → test on D, and vice versa. **Splits by attack strategy and state target, not random** (Goldowsky-Dill et al., 2025).
2. **Activation directions.** Difference-of-means "pending payload" directions (Marks & Tegmark, 2024; Zou et al., 2023), followed by **ablation and steering** to test whether they causally affect trigger-time attack success.
3. **Activation deltas.** TaskTracker-style detector (Abdelnabi et al., 2025) comparing activations before and after reading retrieved state. **Doubles as the key baseline.** Core hypothesis: dormant payloads cause little immediate drift and may evade it.
4. **Attention-based signals.** Attention mass on payload tokens relative to benign stored entries (Hung et al., 2025).
5. **MoE routing.** Expert-routing statistics on payload and downstream tokens in gpt-oss-20b.
6. **Reasoning-channel readout.** Whether gpt-oss's reasoning channel mentions or reasons about the planted instruction during benign turns. Both a behavioral signal and a monitor baseline. Central question: do internals catch poison the reasoning trace doesn't reveal?
7. **Optional SAEs.** No public SAEs are known for gpt-oss-20b. If time permits, train a small SAE on 1–2 informative layers with SAELens to interpret probe directions.

Behavioral detectors from Objective 2 also include **tool-use differences**.

---

## 9. Localization, flagging, and blocking
*Source: Methods > Localization, flagging, and blocking*

- **Dormant turn (B or C) detection:** ablate each in-context stored entry in the request **one at a time (leave-one-out)** and re-run the detector. The span whose removal eliminates the signal is flagged, **returned as response metadata** with the out-of-band alert, and **cross-checked with attention attribution**.
- **Trigger turn (D) detection:** **immediately block execution or use response prefilling** to halt generation before harmful tool execution.
- Rationale: out-of-band flagging avoids interrupting users on dormant-turn false positives; blocking prevents immediate harm on trigger turns.
- **Remediation evaluation:** measure trigger-time ASR and benign utility **before and after simulated removal of the flagged entry** by the client, using programmatic checks.

---

## 10. Evaluation
*Source: Evaluation*

The project succeeds if it gives a clear, well-controlled answer, positive or negative.

**Success criteria.**
- **Testbed validity:** ported payloads reach **≥ 50% trigger-time attack success on at least one model**; poisoned agents' benign utility stays close to clean twins (infection is genuinely dormant).
- **Detection:** on **Type B**, best internals detector reaches **AUROC ≥ 0.85** and a useful **TPR at 1% FPR** on **held-out attack strategies**, and **outperforms** both a text-only scan and a reasoning-channel monitor.
- **Secondary trigger-turn blocking:** on **Type D**, high TPR at **≤ 1% false-block rate** on legitimate user conditionals.
- **Negative control:** internals detection on **Type A** is near chance.
- **Coverage:** fraction of benign turns with payload in context reported **per state target**; for payloads never visible during benign use, Type D detection before harmful tool execution reported.
- **Localization and flagging:** leave-one-out identifies the responsible span in **most** detected cases; client removal of the flagged entry **substantially** reduces trigger-time ASR with **minimal** utility loss.
- **Null outcome:** if no reliable signal, a well-powered negative result with confidence intervals across strategies, state targets, and models.
- **Study 0:** awareness-vs-compliance table across ≥3 models and 4 source types.

**Metrics.** Unit of evaluation = **the request**. AUROC; TPR at 1% and 5% FPR; false-block rate on legitimate conditional actions; per-request overhead (added latency and compute).

**Breakdowns.** Turn type (A/B/C/D); attack strategy; state target; number of benign turns since infection; payload coverage (from retrieval log); detection at the trigger turn.

**Generalization and transfer.** Cross-strategy transfer matrices; cross-type transfer (B/C ↔ D); cross-model transfer gpt-oss-20b → Qwen3.

**Comparisons.** All detectors receive the same input: the API request, plus activations for white-box methods.
- Text-only detectors on request context: **TF-IDF + logistic regression; sentence-embedding classifier; Llama Prompt Guard 2; LLM judge.**
- **Reasoning-channel monitor** on the model's output.
- **TaskTracker-style activation-delta** detector.
- **Szczepaniak et al. (2026) defense**, if applicable at the request level.
- Type D performance explicitly compared against these baselines for early blocking.
- **Chance** = floor. **Clean twin** = non-deployable oracle reference.
- Central comparison: does reading activations add value over reading the request text?

**Additional experiment named in Limitations / Period 4:** small **paraphrase stress test** (adaptive attacker otherwise out of scope).

**Limitations** (report with confidence intervals, scoped to models/environments/attacks tested): open-weight 8–20B models only; simulated or ported payloads; payloads never in a request are undetectable by a provider during dormant turns; assumes provider controls model and serving stack; adaptive attacker out of scope beyond the paraphrase stress test. See §13 for the conflicting Limitations text.

---

## 11. Timeline and milestones (14 weeks)
*Source: Milestones / Timeline; Implementation*

### Period 1 — Weeks 1–2 — Project definition
- [ ] Literature review: delayed attacks, sleeper agents, injection detection, probing.
- [ ] Finalize threat model, turn taxonomy, models, and interpretability toolchain **with the mentor**.
- [ ] Set up Study 0.
- [ ] Set up gpt-oss-20b on NCSA; confirm memory requirements.

**Milestone:** written threat model and experiment plan; gpt-oss-20b running with activation access.

### Period 2 — Weeks 3–5 — Preparation
- [ ] Extend AgentDojo with memory and skill stores.
- [ ] Port Sleeper Attack and explosive-prompt payloads.
- [ ] Generate paired poisoned/clean episodes and hard negatives (with per-turn retrieval log).
- [ ] Build activation-caching pipeline (cache once to disk).
- [ ] Execute Study 0 runs and labeling.

**Milestone:** testbed where ported payloads fire at trigger time while benign utility is preserved; first cached activation dataset; Study 0 results table.

### Period 3 — Weeks 6–9 — Research and development
- [ ] Train and evaluate on gpt-oss-20b across turn types A/B/C/D, including trigger-turn detection: probes, activation directions, activation-delta, attention, routing, reasoning-channel detectors.

**Milestone:** preliminary detection results by turn type; mid-point review with mentor.

### Period 4 — Weeks 10–12 — Evaluation and refinement
- [ ] Held-out generalization and cross-type transfer.
- [ ] All baselines.
- [ ] Localization, flagging, and trigger-blocking experiments.
- [ ] False-block rates on legitimate conditional requests.
- [ ] Transfer to Qwen3.
- [ ] ToolEmu external-validity subset.
- [ ] Paraphrase stress test.

**Milestone:** complete results tables with confidence intervals.

### Period 5 — Weeks 13–14 — Final outputs
- [ ] Final report (workshop-paper format, AI security or interpretability workshop).
- [ ] Clean and release code, testbed extensions, and data.
- [ ] Poster and presentation.

**Milestone:** final report, presentation, released artifacts.

**Deliverables** (*Source: Expected Deliverables*): workshop-format report; open-source AgentDojo extension with memory/skill stores and ported payloads; paired episode dataset with hard negatives; trained detectors and cached activation subsets; localization and flagging code; documentation and analysis notebooks; AIA presentation and poster; workshop manuscript draft; short public write-up.

**Ethics** (*Source: Feasibility and risks*): simulated environments, synthetic data, no real users/accounts/systems; strictly defensive; no new attack techniques; payloads follow published work.

**Technical risk handling** (*Source: Feasibility and risks*): if payloads don't reliably fire on the chosen models, focus on the strategies and state targets that do. If probes only detect surface features, hard negatives and held-out splits will reveal it and it gets reported. A clear null result is still informative.

---

## 12. Implementation notes — NOT from the proposal
*Added by Claude to make the spec implementable. Each item needs researcher approval before it's treated as a decision.*

- **Defender/evaluator separation in code.** Proposal requires ground truth never reach the defender. Suggested enforcement: detectors take only a request object (context + activations + outputs); retrieval logs, twin IDs, and labels live in a separate evaluation-side record. Add a test that detectors can't import or read label fields.
- **Determinism test.** Proposal assumes determinism. Suggested check: running the same episode twice yields identical transcripts before any twin comparisons are trusted.
- **Config + seed + git hash saved with every result**, so results tables can be regenerated by script.
- **Logging of existing decisions.** Record any choice not fixed by the proposal (layers probed, retrieval function, judge prompts) in `docs/decisions.md` with a rationale.
- **Matched hard-negative twins.** Pair each published payload with a benign twin derived from it: keep the payload's frame, trigger, placement and token length, and edit only the harmful element into a legitimate one (the §5 hard-negative style, e.g. "CC my manager"). Direction matters: the poisoned side is always the published payload verbatim; new poisoned variants are never written (§11 Ethics: payloads follow published work). Twins must render to the same token count and differ only inside the edited note (checked by `tests/test_interp_dataset.py`). Used for the Type B demo (`demo/type_b.py`), where the benign twin swaps the task and its reason.
- **Benign-vs-benign control for twin distances.** When comparing hidden states across twins, also compare two different benign twins of the same payload (same frame and length). A poisoned-vs-benign gap only counts as signal if it exceeds the benign-vs-benign gap. Twin distances are analysis only; the twin is the non-deployable oracle of §10.
- **Action-conditioned vs. turn-read triggers across benchmark slices.** The Sleeper Attack benchmark contains two trigger phrasings: (a) action-conditioned triggers ("Before executing <TargetTool> tool...") in `session.json` (all 162/162 LIP cases), and (b) turn-read triggers ("After using this memory... execute in the same turn") in `memory.json`. In agents that routinely review memory on every user turn, turn-read triggers fire immediately on any interaction. To implement true Type B dormancy ("in context, trigger unmet") while keeping payloads 100% verbatim from published benchmarks, memory testbeds prioritize the benchmark's action-conditioned payloads ("Before executing <Tool>..."). This faithfully reflects the paper's canonical "Plant -> Persist (Dormant) -> Trigger" lifecycle across multi-turn benign interactions (0/30 attacks on Type B, 5/5 on Type D).

---

## 13. Inconsistencies and gaps in the proposal — ask the researcher

1. **Two conflicting Limitations paragraphs.** The first says Type D is evaluated as a secondary safety layer. The second (apparently an older duplicate) says detection claims cover **only** B and C, and that trigger-time detection is identified as a defense rather than evaluated. Which one is current?
2. **"Primary secondary result."** The coverage criterion calls Type D detection for never-visible payloads "a primary secondary result." Is it primary or secondary?
3. **"Text-only scan of stored content"** (Detection criterion) vs. text-only detectors **on request context** (Comparisons). The threat model gives the defender no access to stored state, so presumably this means in-request content only. Confirm.
4. **Implementation step (4)** says "localization, and remediation"; Objective 3 was renamed "Localization, flagging, and blocking." Is remediation just the simulated client removal in §9?
5. **Cross-client recurrence** (Anticipated impact: the same poisoned skill across many clients' requests) isn't evaluated anywhere. Out of scope?
6. **Study 0 harness.** Study 0 uses ToolEmu (not AgentDojo) and includes a "tool configuration" source not in Li et al. How are tool-config and tool-returned-note placements constructed in ToolEmu?
7. **Qwen3 in Study 0:** which size (8B or 14B)? Which 2–3 frontier APIs?
8. **Type C generation:** how are near-trigger queries produced and validated as not satisfying the trigger?
9. **Held-out state targets in the Type B success criterion:** the criterion names held-out strategies only; probe splits are by strategy **and** state target. Should the success threshold also apply to held-out state targets?
10. **Thresholds left qualitative:** "useful TPR at 1% FPR," "high TPR," "most detected cases," "substantially reduces," "minimal utility loss." Define numbers or leave qualitative?
11. **Matched hard negatives (§12).** Should the benign twin derived from each published payload (same frame, trigger and token length) be the default clean twin, or a hard negative alongside a plain clean twin? And how minimal should the edit be: the whole task (as in the Type B demo) or only the single harmful element?
