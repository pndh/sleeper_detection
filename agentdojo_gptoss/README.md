# AgentDojo x gpt-oss-20b (local)

Goal: run the AgentDojo prompt-injection benchmark against a locally served
`openai/gpt-oss-20b`, then (phase 2, on hold) study the model's internals during
attacked vs. clean episodes.

## Layout
- `serve_gptoss.sh`   start vLLM OpenAI-compatible server on :8000 (tool calling + reasoning parser)
- `smoke_test.py`     verify the endpoint returns a structured tool call
- `run_dojo.py`       AgentDojo runner (fault-tolerant LLM wrapper, explicit temperature, logs reasoning)
- `run_benchmark.sh`  thin wrapper around run_dojo.py (single process)
- `run_parallel.sh`   one run_dojo.py process per user task, N at a time, then aggregates
- `stress_test.py`    N identical greedy requests; checks determinism / junk output
- `runs/`             first runs via stock agentdojo CLI (effectively temperature 1.0, ~20% server errors) - superseded
- `runs_t0/`          runs via run_dojo.py at temperature 0 (one JSON per user_task x injection_task)
- `install.log`       env build log (conda env: ~/.conda/envs/agentdojo, python 3.12)

## Usage
```bash
./serve_gptoss.sh > vllm.log 2>&1 &          # wait for "Application startup complete"
~/.conda/envs/agentdojo/bin/python smoke_test.py
./run_benchmark.sh --suite slack --user-task user_task_0 --logdir ./runs_t0      # sanity, no attack
./run_parallel.sh slack 8 --attack important_instructions --logdir ./runs_t0 --quiet   # full suite, 8-way parallel
./run_parallel.sh slack 8 --logdir ./runs_t0 --quiet                                    # clean utility
```
Why not the stock CLI (`python -m agentdojo.scripts.benchmark --model VLLM_PARSED`)? It works, but
(a) it passes `temperature=0.0 or NOT_GIVEN` so vLLM samples at T=1.0, (b) vLLM returns HTTP 500 when
gpt-oss emits a malformed harmony tool-call header and the CLI dies after 3 retries, losing the rest
of that user task, (c) `--max-workers>1` crashes. `run_dojo.py` fixes all three and also stores
`reasoning` on each assistant message (needed for phase 2).
Suites: workspace, slack, travel, banking. Attacks: important_instructions, tool_knowledge,
direct, ignore_previous, injecagent, dos, ... (`python -m agentdojo.scripts.benchmark --help`).

## Phase 2 (interpretability, on hold) - notes
- vLLM is a black box for activations. For mechanistic work, replay the exact AgentDojo
  transcripts (from `runs/*.json`) through HF `transformers` in `gpt_oss_lens/`-style code
  (existing logit-lens / attention-routing tooling in ~/gpt_oss_lens, ~/prompt_inj).
- Candidate frameworks: nnsight (works with HF gpt-oss MoE), TransformerLens (no gpt-oss
  support as of now), raw HF hooks (what gpt_oss_lens already does).
- AgentDojo drops `reasoning_content` from history between turns; if reasoning traces are
  needed, capture them from vLLM responses (wrap `OpenAILLM`) or re-generate during replay.

## Server errors
`runs_t0/server_errors.jsonl` lists episodes where all retries (T=0,0.3,0.7,1.0) hit a vLLM 500;
those episodes end with a `[SERVER_ERROR...]` assistant message and score utility=False.

## Metric semantics
Per-episode `security` in `runs/**.json` = injection task goal achieved = attack SUCCEEDED
(AgentDojo's "Average security" printout is the targeted attack success rate; lower is safer).

## First results (slack suite, T=0, 2026-09-22)
| run | episodes | utility | targeted ASR |
|---|---|---|---|
| clean | 21 | 61.9% | - |
| important_instructions | 105 | 51.4% | 71.4% |

`python summarize.py runs_t0` prints per-injection-task ASR and retry / bad-argument counts.
Remaining suites (workspace 40x14, travel 20x7, banking 16x9) not yet run:
`./run_parallel.sh <suite> 8 --attack important_instructions --logdir ./runs_t0 --quiet`.
