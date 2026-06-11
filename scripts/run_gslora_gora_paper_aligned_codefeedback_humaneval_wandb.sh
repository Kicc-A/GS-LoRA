#!/usr/bin/env bash
set -euo pipefail

source /root/anaconda3/etc/profile.d/conda.sh
conda activate gslora

cd /workspace/GS-LoRA
export PYTHONPATH=/workspace/GS-LoRA:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-online}

SEED=${SEED:-42}
GPUS=${GPUS:-2,3}
MASTER_PORT=${MASTER_PORT:-16674}
LOG_DIR="/workspace/GS-LoRA/logs"
RUN_NAME="gora-paper-aligned-codefeedback100k-localgora-strictprompt-humaneval-seed${SEED}-gpus${GPUS//,/}"
OUTPUT_DIR="/workspace/GS-LoRA/outputs_gslora_gora_paper_aligned_codefeedback100k_localgora_strictprompt_humaneval_seed${SEED}_gpus${GPUS//,/}"
RANK_LOG_DIR="/workspace/GS-LoRA/logs_gslora_gora_paper_aligned_codefeedback100k_localgora_strictprompt_humaneval_seed${SEED}_gpus${GPUS//,/}_ranks"

IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}
if [[ "$NUM_GPUS" -eq 8 ]]; then
  PER_DEVICE_TRAIN_BATCH_SIZE=4
  GRADIENT_ACCUMULATION_STEPS=2
elif [[ "$NUM_GPUS" -eq 2 ]]; then
  PER_DEVICE_TRAIN_BATCH_SIZE=1
  GRADIENT_ACCUMULATION_STEPS=32
else
  PER_DEVICE_TRAIN_BATCH_SIZE=1
  GRADIENT_ACCUMULATION_STEPS=64
fi
EFFECTIVE_BATCH=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))

mkdir -p "$OUTPUT_DIR" "$RANK_LOG_DIR" "$LOG_DIR/wandb"

echo "Launching GS-LoRA GoRA-paper-aligned HumanEval"
echo "gpus=${GPUS} seed=${SEED} effective_batch=${EFFECTIVE_BATCH} train_samples=100000 calibration_steps=32 lora_plus=16"

if [[ "$EFFECTIVE_BATCH" -ne 64 ]]; then
  echo "[Warning] Effective batch is ${EFFECTIVE_BATCH}, expected GoRA paper batch 64."
fi

deepspeed --include localhost:${GPUS} --master_port ${MASTER_PORT} \
  --enable_each_rank_log "$RANK_LOG_DIR" \
  examples/NLG/run_code_humaneval_gslora.py \
  --model_name_or_path /workspace/Models/Llama-3.1-8B-Base \
  --output_dir "$OUTPUT_DIR" \
  --train_dataset_name /workspace/GS-LoRA/runtime_datasets/codefeedback_100k_gora_format \
  --train_split train \
  --train_question_column query \
  --train_answer_column answer \
  --eval_dataset_name openai/openai_humaneval \
  --eval_split test \
  --target_modules q_proj k_proj v_proj o_proj \
  --base_rank 8 \
  --tau 0.90 \
  --r_min 4 \
  --r_max 32 \
  --lora_alpha 16 \
  --lora_dropout 0.0 \
  --use_lora_plus \
  --lora_plus_scaler 16 \
  --init_method svd_a_zero_b \
  --init_scale 1.0 \
  --scaling_mode rank \
  --calibration_steps 32 \
  --max_length 1024 \
  --max_new_tokens 512 \
  --max_train_samples 100000 \
  --num_train_epochs 1 \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate 5e-5 \
  --weight_decay 5e-4 \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --bf16 \
  --gradient_checkpointing \
  --seed "$SEED" \
  --num_workers 0 \
  --wandb \
  --wandb_project gslora-humaneval \
  --wandb_name "$RUN_NAME" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_dir "$LOG_DIR/wandb" \
  --wandb_log_interval 10 \
  --trust_remote_code
