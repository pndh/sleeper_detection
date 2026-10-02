# Dormant Poison Detection in LLM Agents

Mechanistic interpretability framework to detect dormant sleeper instructions in LLM agent memory during benign interactions, before trigger conditions are met.

---

### Core Findings

* **Threat & Setting:** Attackers plant persistent memory instructions that remain dormant during routine tasks and fire only when a specific action trigger occurs.
* **Fidelity to Original Attack:** Ported verbatim action-conditioned payloads from the Sleeper Attack benchmark (*Li et al., 2026*) into persistent memory to evaluate intermediate benign interactions (**Turn Type B**).
* **Behavioral Dormancy (45 Runs on `gpt-oss-20b`):**
  * **Benign Turns (Type B):** **0 / 30 attacks** across conversational and other-tool queries.
  * **Trigger Turn (Type D):** **5 / 5 attacks** on poisoned variant; **5 / 5** benign actions on twin.
* **Control Twins:** 43-token paired twins cancel **99.95%** of container formatting overhead ($\Delta = 0.000025$).

---

### White-Box Interpretability (`gpt-oss-20b`)

* **Delimiter Collapse:** Probing at standard prompt boundaries (`last_prompt`) mutes the signal to **0.00359** due to formatting token convergence.
* **Internal Signal Amplification:** Probing directly at internal memory tokens (`entry0`) recovers an **8.02x stronger signal** (cosine distance = **0.02882**).
* **MoE Routing Bifurcation (Layer 20):** Top-4 routing bifurcates sharply with **$J = 0.33$** overlap between poisoned and benign twins.
* **Subspace Alignment (Layer 10):** Difference vectors show **67.4% alignment** between dormant ($B$) and active ($D$) states, enabling 1D zero-shot linear separation.
* **Hidden Deliberation (CoT):** 100% of traces confirm the model explicitly verifies the unmet trigger condition and suppresses execution in hidden reasoning.

---

### Detection Guidelines

* ❌ **Avoid:** Probing at `last_prompt` (dampened by delimiter collapse).
* 🎯 **Probe Sites:** Internal memory tokens (`entry0`) or `memory_result_end`.
* 🎯 **Target Layers:** **Layer 10** for linear latent probes; **Layer 20** for MoE routing probes.
