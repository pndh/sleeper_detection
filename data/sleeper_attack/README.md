# Anonymous Artifact Release

This repository contains the released materials for our paper on Sleeper Attack against tool-using agents. It includes the benchmark datasets, sanitized case-level results, supplementary experiments, and the Rule Defense, Guard Defense, and AgentDoG Defense implementations evaluated in the paper.

![Main Figure](figures/main_figure.png)

## Released Contents

- `datasets/`: benchmark datasets for the three attack strategies.
- `results/main/`: sanitized case-level outputs for the main experiments.
- `results/supplementary/`: supplementary experiment and defense results.
- `results/supplementary/human_audit/`: complete instructions for the independent human audit.
- `defense/`: defense implementations, AgentDoG serving code, and batch runners.
- `skill_data/`: skill files referenced by the released datasets.
- `run_batch.py` and `src/`: the runtime needed to execute a selected dataset slice.
- `config.py`, `.env.example`, and `requirements.txt`: sanitized runtime configuration and dependencies.

## Terminology And Layout

The directory names map to the paper's attack strategies as follows:

- `latent_instruction_planting` = Latent Instruction Planting (`LIP`)
- `proactive_information_elicitation` = Proactive Information Elicitation (`PIE`)
- `persistent_information_corruption` = Persistent Information Corruption (`PIC`)

Each strategy contains `session`, `memory`, and `skill` agent states. The `single` slice is the direct single-interaction baseline and is not an agent state.

```text
anonymous_main_experiments_repo/
|-- datasets/
|   |-- latent_instruction_planting/
|   |-- proactive_information_elicitation/
|   |-- persistent_information_corruption/
|   `-- supplementary/
|       |-- fresh_session_replay/
|       |-- conditional_session_trigger_taxonomy/
|       `-- longer_horizon_triggering/
|-- results/
|   |-- main/
|   `-- supplementary/
|       |-- defense_evaluation/
|       |-- fresh_session_replay/
|       |-- conditional_session_trigger_taxonomy/
|       |-- longer_horizon_triggering/
|       `-- open_model_scaling/
|-- defense/
|-- figures/
|-- skill_data/
|-- src/
|-- config.py
|-- run_batch.py
`-- requirements.txt
```

The open-model scaling results contain the `session`, `memory`, and `skill` states and use paths such as `results/supplementary/open_model_scaling/table1_memory/qwen3-32b.json`.

## Defense Evaluation

The defense experiment uses four models and a fixed stratified sample of 450 cases per model: 50 cases for each of the nine attack-strategy and agent-state combinations. The released methods are:

- **Rule Defense** (`rule_defense`): prepends reliability rules to the agent instruction.
- **Guard Defense** (`guard_defense`): uses LlamaGuard to review tool-return content during execution.
- **AgentDoG Defense** (`agentdog_defense`): reviews tool-return content and checks each proposed `MemoryUpdate` or `SkillUpdate` before it changes persistent state.

| Model | N | No Defense | Rule Defense | Guard Defense | AgentDoG Defense |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-Plus | 450 | 36.9 | 34.2 | 30.7 | 22.2 |
| Qwen3.5-Flash | 450 | 14.7 | 13.3 | 9.8 | 11.3 |
| DeepSeek-R1 | 450 | 42.4 | 18.0 | 22.0 | 16.9 |
| Llama-3.3-70B-Instruct | 450 | 33.3 | 20.2 | 16.9 | 15.1 |
| **All models** | **1,800** | **31.8** | **21.4** | **19.8** | **16.4** |

Values are ASR percentages. Case-level outputs and aggregate summaries are under `results/supplementary/defense_evaluation/`. The same normalized IDs, `450_001` through `450_450`, identify the sampled cases for every method and model.

## Supplementary Coverage

- `fresh_session_replay` contains cases in which the original persistent write succeeded but the same-session trigger did not.
- `conditional_session_trigger_taxonomy` contains the `original_trigger` baseline and the `conditional_rewrites` benchmark.
- `longer_horizon_triggering` contains the PIC turn sweep for turns `1`, `3`, `5`, `8`, `12`, and `20`.
- `open_model_scaling` contains the Qwen `session`, `memory`, and `skill` states. Direct single-interaction runs are not included in that supplementary table.

These experiments reuse the released main datasets and therefore do not add duplicate datasets under `datasets/supplementary/`.

## Running A Released Slice

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Provide model credentials through environment variables. No keys or private endpoints are included in this release.

Common variables are `AGENT_MODEL`, `AGENT_API_KEY`, `AGENT_BASE_URL`, `SIMULATOR_MODEL`, `SIMULATOR_API_KEY`, and `SIMULATOR_BASE_URL`. `API_KEY` and `BASE_URL` can serve as shared fallbacks when the same provider is used for both models.

Example: run PIC with memory state.

```bash
python run_batch.py ^
  --dataset datasets/persistent_information_corruption/memory.json ^
  --results outputs/persistent_information_corruption_memory.json ^
  --max-concurrent 10
```

Example: run one PIE case with session state.

```bash
python run_batch.py ^
  --single CASE_ID ^
  --dataset datasets/proactive_information_elicitation/session.json ^
  --results outputs/debug_case.json
```

## Running The Defense Matrix

The defense scripts are under `defense/`. The following command reproduces the paper's strategy-state coverage for one model after the required model endpoints have been configured:

```bash
python defense/run_defense_matrix.py ^
  --model qwen3.5-plus ^
  --mode rule_defense,guard_defense,agentdog_defense ^
  --table table1,table2,table3 ^
  --state session,skill,memory ^
  --sample-size 50 ^
  --results-dir outputs/defense_eval/matrix
```

Use `python defense/run_defense.py --list-tables` to inspect the aliases for individual slices. AgentDoG can be served through the OpenAI-compatible endpoint in `defense/agentdog_server.py`; its additional dependencies are listed in `defense/agentdog_requirements.txt`.

## Result Format

Each result file stores case-level trajectories and evaluation outputs for one model and benchmark slice. Runtime bookkeeping, local paths, credentials, and timestamps are excluded from the released files.

```text
results/main/<attack_strategy>/<agent_state>/<model>.json
results/supplementary/<experiment>/...
```
