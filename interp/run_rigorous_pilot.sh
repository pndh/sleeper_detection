#!/usr/bin/env bash
set -e

echo "================================================================================"
echo "STARTING RIGOROUS SOUNDNESS SUITE (ICLR SPECIFICATION)"
echo "Host: $(hostname) | Date: $(date)"
echo "================================================================================"

CPDM_DIR="/home/huynp2/cpdm3d_attention3d_portable"
CPDM_ENV="/home/huynp2/.conda/envs/cpdm3d/bin/python"
GPT_ENV="/home/huynp2/.conda/envs/gpt-oss-env/bin/python"
DATASET="/home/huynp2/data/petct_paired_onefold_1100_140_136_int16_rescaled"

resume_cpdm() {
    echo ""
    echo "================================================================================"
    echo "RESUMING CPDM TRAINING PIPELINE"
    echo "================================================================================"
    cd "$CPDM_DIR"
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    
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

# Trap ensures CPDM is resumed on exit or failure
trap resume_cpdm EXIT

echo ""
echo "=== Step 1: Capturing Activations for Sound 2x2 Dataset (40 Real-Case Requests) ==="
cd /home/huynp2/sleeper_detection

# Explicitly ensure low_cpu_mem_usage to protect the 8 GB cgroup cap
"$GPT_ENV" -c "
import sys
sys.path.insert(0, '/home/huynp2/sleeper_detection/interp')
import gptoss as G
from pathlib import Path

data_dir = Path('/home/huynp2/sleeper_detection/interp/data/sound_2x2')
cache_dir = Path('/home/huynp2/sleeper_detection/interp/cache/sound_2x2')
cache_dir.mkdir(parents=True, exist_ok=True)

print('Capturing activations to:', cache_dir)
G.capture(data_dir, cache_dir, router=True, include_entries=True)
print('Sound dataset captured successfully.')
"

echo ""
echo "=== Step 2: Evaluating Pre-Fixed Window Probing & Interaction Endpoint ==="
"$GPT_ENV" interp/eval_sound_detector.py

echo ""
echo "=== Step 3: Verifying In-Place Causal Ablation Assertion & Tool-Call ASR ==="
"$GPT_ENV" interp/verify_causal_soundness.py

echo ""
echo "=== Step 4: Suite Completed Successfully. Triggering CPDM Resumption ==="
# Exit trap will run resume_cpdm now
