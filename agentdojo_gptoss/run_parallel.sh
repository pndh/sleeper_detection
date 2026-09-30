#!/usr/bin/env bash
# Parallelize run_dojo.py over user tasks (one process per user task, N at a time), then aggregate.
# Usage: ./run_parallel.sh <suite> [N_PARALLEL=8] [run_dojo.py args...]
#   ./run_parallel.sh slack 8 --attack important_instructions
#   ./run_parallel.sh slack 8                       # clean utility run
set -uo pipefail
SUITE=$1; N=${2:-8}; shift 2 || shift $#
ENV=/home/huynp2/.conda/envs/agentdojo
cd "$(dirname "$(readlink -f "$0")")"
TASKS=$($ENV/bin/python -c "
from agentdojo.task_suite.load_suites import get_suite
print('\n'.join(get_suite('v1.2.2','$SUITE').user_tasks))")
TAG=$(echo "${*:-clean}" | tr -c "a-zA-Z0-9_\n" "_")
mkdir -p logs
echo "$TASKS" | xargs -P "$N" -I{} bash -c \
  "$ENV/bin/python run_dojo.py --suite $SUITE --user-task {} $* > logs/${SUITE}_${TAG}_{}.log 2>&1 && echo done {} || echo FAILED {}"
echo "failed tasks: $(grep -l 'Traceback' logs/${SUITE}_${TAG}_*.log 2>/dev/null | wc -l) (re-run same command; finished episodes are cached)"
# aggregate: everything is cached now, so this only prints the summary
$ENV/bin/python run_dojo.py --suite "$SUITE" --quiet "$@" 2>/dev/null | grep RESULT
