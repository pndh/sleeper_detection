# Strategic Research Feedback & Experimental Disentanglement

*Documented on 2026-10-02 from mentor/reviewer critique.*

## 1. Disentangling Competing Explanations: The 2x2 Factorial Design

The probe's signal on Turn Type B could fit three competing narratives:
1. **Content Detector:** The probe fires on any malicious text (keywords like "delete", "exfiltrate").
2. **Instruction Detector:** The probe fires on any pending conditional rule ("if/before X, do Y").
3. **Core Hypothesis (Latent Commitment):** The probe fires **only** on a live pending malicious instruction.

### The 2x2 Matrix: `{Malicious, Benign Payload} × {Live Instruction, Inert Framing}`

| Framing \ Payload | Malicious Payload (`ShopifyDeleteProduct`) | Benign Payload (`GoogleHomeSetReminder`) |
| :--- | :--- | :--- |
| **Live Instruction** (`"Before executing Tool X, you must..."`) | **Cell (1,1): Dormant Sleeper Attack**<br>*(Core Setting)* | **Cell (1,2): Benign Conditional Twin**<br>*(Matched Hard Negative)* |
| **Inert Framing** (`"Log/Quote: user mentioned..."`) | **Cell (2,1): Passive Malicious Data**<br>*(Tests pure keyword sensitivity)* | **Cell (2,2): Passive Benign Data**<br>*(Clean baseline)* |

### Falsification Logic:
* If the probe is a **Content Detector**, the entire **Malicious Column** lights up: `Cell (1,1) == Cell (2,1) > 0`.
* If the probe is an **Instruction Detector**, the entire **Live Instruction Row** lights up: `Cell (1,1) == Cell (1,2) > 0`.
* If the **Core Hypothesis** holds, **ONLY Cell (1,1)** lights up. Cells (1,2), (2,1), and (2,2) remain at baseline.

---

## 2. Behavioral Validation: Predicting Trigger Execution from Dormant State

* In published benchmarks (*Li et al., 2026* Table 21), a substantial fraction of planted payloads fail to fire even when triggered (write-only duds).
* **Test:** Among requests with the *exact same poisoned payload*, does the Type B probe score predict whether the agent will actually execute the attack when triggered on Turn D?
* **Soundness Impact:** If $P(\text{Execute}_D \mid \text{High Score}_B) \gg P(\text{Execute}_D \mid \text{Low Score}_B)$, we prove the detector measures **readiness/latent commitment**, not the static presence of text.

---

## 3. Causal Mediation Analysis (Moving Beyond Correlation)

Utilizing `interp/gptoss.py:hooks()`:
1. **Ablation on Type B:** Project out the identified Layer 10 payload vector ($\mathbf{h}' = \mathbf{h} - (\mathbf{h}\cdot\hat{\mathbf{v}})\hat{\mathbf{v}}$) during benign turns, and test whether subsequent trigger-turn ASR on Turn D drops.
2. **Steering on Clean Agents:** Add the vector into clean agents during benign turns and observe whether trigger-time attack propensity rises.

---

## 4. Explicit Scoping of PIC (Persistent Information Corruption)

* **LIP (Latent Instruction Planting) & PIE:** Plant imperative action conditionals (`"Before X, do Y"`).
* **PIC:** Plants corrupted factual values (`"Saved email is evil@corp"`), lacking an action trigger.
* **Scope Decision:** Treat PIC either as a dedicated sub-question (*"Does latent commitment extend to declarative state corruption?"*) or formally scope the primary detector to action-conditioned sleeper threats.

---

## 5. Threat Model & Paraphrase Robustness

* Clarify threat model: The defender evaluates against a static planted payload first.
* Include a paraphrase stress-test: evaluate probe robustness when the attacker paraphrases the trigger condition and action command with varying syntax and synonyms.
