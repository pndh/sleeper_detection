#!/usr/bin/env bash
set -e

echo "================================================================================"
echo "STARTING SCALED-UP 2x2x10 EVALUATION SUITE ON A100 GPU (gpt-oss-20b MoE)"
echo "Host: $(hostname) | Date: $(date)"
echo "================================================================================"

CPDM_DIR="/home/huynp2/cpdm3d_attention3d_portable"
CPDM_ENV="/home/huynp2/.conda/envs/cpdm3d/bin/python"
GPT_ENV="/home/huynp2/.conda/envs/gpt-oss-env/bin/python"
DATASET="/home/huynp2/data/petct_paired_onefold_1100_140_136_int16_rescaled"

resume_cpdm() {
    echo ""
    echo "================================================================================"
    echo "STEP 3: RESUMING CPDM TRAINING PIPELINE"
    echo "================================================================================"
    cd "$CPDM_DIR"
    # Export PyTorch allocator config to prevent memory fragmentation OOM
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    
    # Launch pipeline in background
    nohup "$CPDM_ENV" "$CPDM_DIR/scripts/run_pipeline.py" "$DATASET" --kind onefold >> "$CPDM_DIR/run_onefold.log" 2>&1 &
    NEW_PID=$!
    echo "CPDM run_pipeline.py launched with PID: $NEW_PID (with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)"
    sleep 3
    
    # Launch fast memory watchdog
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

echo ""
echo "=== Step 1: Capturing Activations for 40 Requests (2x2x10) on A100 GPU ==="
cd /home/huynp2/sleeper_detection

# Explicitly ensure GPU is targeted and system host RAM is preserved (< 8 GB)
"$GPT_ENV" -c "
import sys
sys.path.insert(0, '/home/huynp2/sleeper_detection/interp')
import gptoss as G
from pathlib import Path

data_dir = Path('/home/huynp2/sleeper_detection/interp/data/scaleup_2x2x10')
cache_dir = Path('/home/huynp2/sleeper_detection/interp/cache/scaleup_2x2x10')
cache_dir.mkdir(parents=True, exist_ok=True)

print('Capturing activations to:', cache_dir)
G.capture(data_dir, cache_dir, router=True, include_entries=True)
print('All 40 requests captured successfully.')
"

echo ""
echo "=== Step 2: Running Statistical Evaluation & LOCO-CV on 2x2x10 Dataset ==="
"$GPT_ENV" interp/eval_scaleup_2x2x10.py

echo ""
echo "=== Step 3: Experiments Complete. Resuming CPDM as requested ==="
resume_cpdm

echo ""
echo "================================================================================"
echo "ALL TASKS COMPLETED SUCCESSFULLY!"
echo "================================================================================"
