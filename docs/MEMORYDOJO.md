# MemoryDojo Benchmark: Architecture, Formalisms & Evaluation Guide

This document specifies the design, mathematical foundations, and implementation of the **MemoryDojo** benchmark in this repository (`benchmarks/memorydojo/`), serving as a reference for benchmark validation and mechanistic interpretability analysis.

---

## 1. Executive Summary & Benchmark Purpose

**MemoryDojo** models long-horizon, persistent memory vulnerabilities in autonomous LLM agents. 

Unlike conventional prompt injection benchmarks (e.g., AgentDojo, BIPIA) where an attack occurs in a **single turn**, MemoryDojo evaluates the **temporal separation** between:
1. **Memory Ingestion ($t_0$)**: When a malicious payload $e$ is written into the agent's memory store.
2. **Intermediate Sessions ($t_1, \dots, t_{k-1}$)**: Benign user interaction where the payload remains dormant.
3. **Trigger Retrieval & Execution ($t_k$)**: When a trigger query $u_{t_k} = 1$ causes the memory system to retrieve $e$ into active context, driving the agent to execute malicious action $z_{t_k} = 1$.

---

## 2. Mathematical Formalisms & State Taxonomy

MemoryDojo formally defines the agent's state at any session step $t$ using the binary pair $(u_t, z_t)$:
- $u_t \in \{0, 1\}$: **Trigger State**. Indicates whether the user instruction / context activates the payload's execution conditions ($u_t=1$) or is a benign / untriggered request ($u_t=0$).
- $z_t \in \{0, 1\}$: **Execution State**. Indicates whether the agent executes the unauthorized payload behavior ($z_t=1$) or behaves normally ($z_t=0$).

### Formal State Classifications

| State Name | Formal Condition | Description | Interp Role |
| :--- | :--- | :--- | :--- |
| **`Clean Baseline`** | $u_t = 0, z_t = 0$ | Standard benign user request; no memory payload retrieved. | Baseline reference distribution. |
| **`Benign Twin`** | Control Pair | Structurally matched twin containing identical formatting/tokens without harmful instructions. | Eliminates prompt length & container token artifacts. |
| **`Retrieved Unreached`** | $u_t = 1, z_t = 0$ | Malicious payload is retrieved into active context $C_t$, but the agent has **not yet executed** the action (pre-execution / dormant state). | **Primary target for latent detection** (where Attention Tracker / AttnDA fail). |
| **`Reached Execution`** | $u_t = 1, z_t = 1$ | Payload is active and the agent proceeds to generate malicious tool calls or outputs. | Standard attack success state. |

---

## 3. Supported Agent Memory Architectures

MemoryDojo models three distinct, industry-standard agent memory architectures ([`benchmarks/memorydojo/memory_systems.py`](benchmarks/memorydojo/memory_systems.py)):

1. **`WorkingBufferMemory` (Sliding Window / FIFO Buffer)**:
   - Stores recent observations directly in context.
   - Payloads naturally decay or get evicted after $W$ interaction turns.
2. **`VectorStoreMemory` (External Vector Database / RAG)**:
   - Embeds memories into vector space.
   - Retrieval occurs if and only if the current query $q$ has cosine similarity exceeding threshold $\tau$ with the stored payload:
     $$\text{cosine\_sim}(q, e) \ge \tau$$
3. **`LettaCoreMemory` (Hierarchical Core / Persona Blocks)**:
   - Modeled after MemGPT / Letta core memory blocks.
   - Stored in persistent system blocks that are perpetually pinned in the system context until explicitly edited or deleted.

---

## 4. Benchmark Lifecycle Metrics

MemoryDojo quantifies the threat landscape through three decoupled, sequential probabilities:

$$P(\text{Breach}) = P(\text{Insertion}) \times P(\text{Retrieval} \mid \text{Inserted}) \times P(\text{Execution} \mid \text{Retrieved})$$

1. **Insertion Success Rate (ISR)**:
   $$\text{ISR} = \Pr(e \in M_{t_0})$$
   The fraction of attack attempts that successfully write the malicious payload $e$ into persistent memory $M_{t_0}$.
2. **Retrieval Success Rate (RSR)**:
   $$\text{RSR} = \Pr(e \in C_t \mid u_t = 1, e \in M_{t_0})$$
   The probability that when a trigger instruction occurs ($u_t=1$), the memory system retrieves the stored payload into the active context window $C_t$.
3. **Breach Rate (BR) / Execution Rate**:
   $$\text{BR} = \Pr(z_t = 1 \mid u_t = 1, e \in C_t)$$
   The probability that an agent executing with the retrieved payload in its active context actually carries out the unauthorized action.
4. **Retention Half-Life ($H_{1/2}$)**:
   The interaction horizon (number of turns or sessions) after which retrieval probability decays by 50%.

---

## 5. Architectural Comparison: Past "Sleeper Agent Dojo" vs. Current MemoryDojo

| Aspect | Legacy "Sleeper Agent Dojo" | Current MemoryDojo Benchmark |
| :--- | :--- | :--- |
| **Temporal Dynamic** | Single-turn prompt injection (direct tool execution) | Multi-session persistent memory lifecycle ($t_0 \to t_1 \dots \to t_k$) |
| **Memory Realism** | Mocked flat text string inside synthetic prompt delimiters | 3 realistic memory systems (Buffer, Vector RAG, Letta Core) |
| **Terminology** | Colloquial jargon: `fired`, `dormant`, `inert`, `live` | Formal peer-reviewed: `reached_execution`, `retrieved_unreached`, `benign_twin`, `clean` |
| **Evaluation Metrics** | Coarse binary breach rate | Structured pipeline: $\text{ISR} \to \text{RSR} \to \text{BR}$ + Retention Half-Life |
| **Control Pairs** | Unmatched synthetic examples | Exact 43-token paired benign twins canceling container token noise |
| **Model Scope** | Tied to single unquantized MoE (`gpt-oss-20b`) | AttnDA catalogue (`Llama-3-8B`, `Qwen2-1.5B`, `Phi-3-mini`, `Mistral-7B`) |
| **Code Structure** | Monolithic, mixed evaluation scripts | Modular: `taxonomy.py`, `memory_systems.py`, `trajectory_builder.py`, `metrics.py` |

---

## 6. How to Validate the Benchmark

### 1. Run Unit & Lifecycle Tests
```bash
PYTHONPATH=. pytest tests/test_new_framework.py -v
```

### 2. Run Synthetic Trajectory Simulation
```python
from benchmarks.memorydojo import (
    TaskDomain,
    MemorySystemType,
    build_synthetic_trajectory,
    compute_lifecycle_metrics,
)

# 1. Build a multi-session trajectory across 5 turns
traj = build_synthetic_trajectory(TaskDomain.WORKSPACE, length=5, t0=0)

# 2. Evaluate lifecycle rates
metrics = compute_lifecycle_metrics([traj])
print(f"ISR: {metrics.isr:.2f}, RSR: {metrics.rsr:.2f}, BR: {metrics.br:.2f}")
```

### 3. Cross-Validate with Mechanistic Probing
The latent activations extracted at the `memory_result_end` token position in `retrieved_unreached` states ($u_t=1, z_t=0$) form the exact dataset evaluated by the subspace probes in `rq2_probing/` and causal steering in `rq3_mechanisms/`.
