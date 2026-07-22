#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/GS-LoRA
RUNNER=$ROOT/run_llama31_metamathqa_gslora_sqrtrank_4gpu.sh

for seed in 2025 1319 1; do
  exp_name="llama31_metamathqa_gslora_sqrtrank_seed${seed}_gpu0123"
  echo "[$(date '+%F %T')] starting seed=$seed exp=$exp_name"
  SEED="$seed" \
  EXP_NAME="$exp_name" \
  WANDB_NAME="gslora-llama31-metamathqa-sqrtrank-gpu0123-seed${seed}" \
    "$RUNNER"
  echo "[$(date '+%F %T')] completed seed=$seed exp=$exp_name"
done
