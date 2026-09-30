#!/usr/bin/env bash
# Run a command under a container-memory guard. The container resets if anon memory hits the 8 GiB cgroup cap,
# so this kills the command's process group when anon crosses LIMIT_MIB (default 6656 = 6.5 GiB).
# Usage: ./run_guarded.sh <cmd> [args...]      Log of peaks: guard.log
LIMIT_MIB=${LIMIT_MIB:-6656}
setsid "$@" &
PID=$!
PEAK=0
while kill -0 "$PID" 2>/dev/null; do
  ANON=$(( $(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat) / 1048576 ))
  (( ANON > PEAK )) && PEAK=$ANON
  if (( ANON >= LIMIT_MIB )); then
    echo "[guard] anon ${ANON} MiB >= ${LIMIT_MIB} MiB, killing process group $PID" | tee -a guard.log >&2
    kill -9 -- -"$PID" 2>/dev/null
    wait "$PID" 2>/dev/null
    exit 137
  fi
  sleep 0.3
done
wait "$PID"; RC=$?
echo "[guard] exit=$RC peak_anon=${PEAK}MiB cmd=$*" | tee -a guard.log >&2
exit $RC
