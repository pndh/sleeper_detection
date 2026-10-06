#!/usr/bin/env bash
set -e

echo "================================================================================"
echo "STARTING A100 REVIEWER EXPERIMENTAL SUITE FOR SLEEPER DETECTION (gpt-oss-20b)"
echo "Host: $(hostname) | Date: $(date)"
echo "================================================================================"

CPDM_DIR="/home/huynp2/cpdm3d_attention3d_portable"
CPDM_ENV="/home/huynp2/.conda/envs/cpdm3d/bin/python"
GPT_ENV="/home/huynp2/.conda/envs/gpt-oss-env/bin/python"
DATASET="/home/huynp2/data/petct_paired_onefold_1100_140_136_int16_rescaled"

resume_cpdm() {
    echo ""
    echo "================================================================================"
    echo "GUARANTEED RESUMPTION: Restarting CPDM Training Pipeline"
    echo "================================================================================"
    cd "$CPDM_DIR"
    nohup "$CPDM_ENV" "$CPDM_DIR/scripts/run_pipeline.py" "$DATASET" --kind onefold >> "$CPDM_DIR/run_onefold.log" 2>&1 &
    NEW_PID=$!
    echo "CPDM run_pipeline.py launched with PID: $NEW_PID"
    sleep 3
    nohup bash "$CPDM_DIR/scripts/memory_watchdog_fast.sh" $NEW_PID > /dev/null 2>&1 &
    WATCH_PID=$!
    echo "CPDM memory watchdog launched with PID: $WATCH_PID"
    sleep 5
    echo "Current Process State:"
    ps -ef | grep -E 'python|cpdm|watchdog' | grep -v grep || true
    echo "GPU Status after Resumption:"
    nvidia-smi
    echo "CPDM successfully resumed!"
}

# Ensure resume_cpdm runs on exit or error
trap resume_cpdm EXIT

echo ""
echo "=== Step 1: Pausing CPDM to release GPU memory ==="
# Kill watchdog first
pkill -f memory_watchdog_fast.sh || true
# Gracefully signal pipeline and training
pkill -TERM -f run_pipeline.py || true
pkill -TERM -f train_cpdm3d_manifest.py || true
sleep 4

# Check if any training processes linger and terminate
pkill -9 -f train_cpdm3d_manifest.py 2>/dev/null || true
pkill -9 -f run_pipeline.py 2>/dev/null || true

# Wait until GPU memory is freed (< 2000 MiB)
for i in {1..20}; do
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    echo "Waiting for GPU release... Current memory used: ${MEM_USED} MiB"
    if [ "$MEM_USED" -lt 2000 ]; then
        echo "GPU VRAM released! (${MEM_USED} MiB)"
        break
    fi
    sleep 2
done

echo ""
echo "=== Step 2: Executing Full Reviewer Experimental Suite (MoE gpt-oss-20b) ==="
cd /home/huynp2/sleeper_detection
"$GPT_ENV" interp/run_full_reviewer_suite.py

echo ""
echo "=== Step 3: Experimental Suite Finished Successfully ==="
# The EXIT trap will automatically run resume_cpdm now
