# Interp setup for gpt-oss-20b

Hugging Face (bf16) access to gpt-oss-20b's internals on the exact prompts the model received under vLLM.
vLLM stays the tool for fast rollouts; this is the tool for activations, lenses, and interventions.
Only one of the two fits on the A100 at a time.

## Layout

| Path | What it is | Python |
|---|---|---|
| `build_dataset.py` | Turns the demo generators and their vLLM runs into a dataset | `.venv-tools` |
| `data/<name>/requests.jsonl` | **Defender view.** Request, exact prompt token ids, label-free positions and memory-entry spans | any |
| `data/<name>/labels.jsonl` | **Evaluation only.** Case, variant, turn type, payload entry, observed behaviour | any |
| `data/<name>/outputs.jsonl` | vLLM's reply (repeat 0) and teacher-forced token ids | any |
| `gptoss.py` | Load model, forward with activations, logit lens, hooks, activation cache | `gpt-oss-env` |
| `validate_and_capture.py` | HF-vs-vLLM fidelity check, cache every request, first lens | `gpt-oss-env` |
| `cache/<name>/` | Cached activations: one `.npz` per request plus `index.json` | any (numpy) |
| `run_guarded.sh` | Kills a job before it hits the container's 8 GiB memory cap | shell |
| `../tests/test_interp_dataset.py` | Label isolation, twin equality, span decoding | `.venv-tools` |

## Quickstart

```bash
cd ~/sleeper_detection
.venv-tools/bin/python interp/build_dataset.py --out interp/data/demo_v1   # offline, no GPU
.venv-tools/bin/python -m pytest -q tests                                  # 4 checks
cd interp && ./run_guarded.sh ~/.conda/envs/gpt-oss-env/bin/python validate_and_capture.py
```

In your own script (run from `interp/`, under `run_guarded.sh`):

```python
import gptoss as G
reqs, labels, outs = G.load_dataset("data/demo_v1")
r = reqs[0]
pos = G.named_positions(r)                      # last_prompt, user_end, memory_result_end, entry<i>.occ<j>.last
o = G.forward(r["input_ids"], positions=pos.values(), attentions=True)
o["resid"].shape                                # [n_pos, 25, 2880]  embeddings + 24 layers
o["router"].shape                               # [n_pos, 24, 32]    router logits
lens = G.logit_lens(o["resid"][0])              # per layer top tokens at that position
ids, start = G.teacher_force_ids(outs[r["id"]]) # prompt + the model's own reply, reply starts at `start`
with G.hooks({12: lambda h: h * 1.0}):          # edit the residual stream after layer 12
    ...
c = G.load_cache("cache/demo_v1", "memory_result_end")   # stacked cached activations: c["resid"] [n, 25, 2880]
```

## Facts measured on this machine (2026-09-30)

- Load: MXFP4 checkpoint dequantized to bf16 (triton 3.2 is below the 3.4 the MXFP4 kernels need). About 75 s,
  39 GiB GPU, 3.1 to 3.3 GiB host memory at peak.
- Model: 24 layers, d_model 2880, 32 experts, top-4 routing, eager attention (attention weights available).
- Prompt tokens: `demo/harmony_render.py` reproduces the vLLM chat endpoint's prompt exactly (matched the server's
  `prompt_tokens` on all 21 requests checked). Do not use the HF chat template or vLLM's `/tokenize`: both differ.
- Fidelity: teacher-forcing the vLLM replies, HF bf16 argmax matches the recorded token on 93% to 100% of reply
  tokens (most requests about 98%). The gap is MXFP4 versus bf16 numerics.

## Things to know

- **Labels stay out of detectors.** Load `labels.jsonl` only in evaluation code. Request ids are opaque hashes.
- **The newest memory note is entry 0.** The paper's memory tool lists notes newest first, and repeats the newest
  one in `latest_entry` and `summary`, so it has 5 occurrences; older notes have 3.
- **Span edges.** The last token of an entry span can also hold the closing quote (`execute.',`), because the
  tokenizer merges them.
- **Where to read.** At `last_prompt` the next token is always `<|channel|>`, so every layer's lens converges on
  it. More informative positions: `memory_result_end`, the entry spans, and the reply tokens via teacher forcing.
- **Teacher forcing is a reconstruction.** vLLM returns parsed text, not raw sampled tokens. The reply is re-encoded
  in harmony format (analysis channel, then the first tool call or the final text).
- **Memory guard.** The container resets if anonymous memory reaches the 8 GiB cap. `run_guarded.sh` kills the job
  at 6.5 GiB (`LIMIT_MIB` to change). Loading plus one forward pass peaks near 3.3 GiB.
