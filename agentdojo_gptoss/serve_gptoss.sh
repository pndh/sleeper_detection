#!/usr/bin/env bash
# Serve openai/gpt-oss-20b via vLLM with an OpenAI-compatible API + tool calling.
# Weights are already in ~/.cache/huggingface. Logs to vllm.log.
set -euo pipefail
ENV=/home/huynp2/.conda/envs/agentdojo
PORT=${PORT:-8000}
export HF_HUB_OFFLINE=1
# no nvcc on this box: skip flashinfer JIT sampler (vLLM falls back to its own kernels)
export VLLM_USE_FLASHINFER_SAMPLER=0
exec $ENV/bin/vllm serve openai/gpt-oss-20b \
  --port "$PORT" \
  --served-model-name gpt-oss-20b \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  --max-num-seqs 32 \
  "$@"
