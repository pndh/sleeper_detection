#!/usr/bin/env bash
# Single-process AgentDojo run against local vLLM gpt-oss-20b (see run_dojo.py --help).
#   ./run_benchmark.sh --suite slack --user-task user_task_0
#   ./run_benchmark.sh --suite slack --attack important_instructions
cd "$(dirname "$(readlink -f "$0")")" && exec /home/huynp2/.conda/envs/agentdojo/bin/python run_dojo.py "$@"
