#!/usr/bin/env bash
set -euo pipefail

source /root/anaconda3/etc/profile.d/conda.sh
conda activate gslora

cd /workspace/GS-LoRA
export PYTHONPATH=/workspace/GS-LoRA:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online

SEED=${SEED:-42}
GPU=${GPU:-3}
MASTER_PORT=${MASTER_PORT:-16673}
RUN_NAME="paper-hparams-codefeedback-humaneval-seed${SEED}-gpu${GPU}"
OUTPUT_DIR="/workspace/GS-LoRA/outputs_gslora_paper_hparams_codefeedback_humaneval_seed${SEED}_gpu${GPU}"
RANK_LOG_DIR="/workspace/GS-LoRA/logs_gslora_paper_hparams_codefeedback_humaneval_seed${SEED}_gpu${GPU}_ranks"
LOG_DIR="/workspace/GS-LoRA/logs"

mkdir -p "$OUTPUT_DIR" "$RANK_LOG_DIR" "$LOG_DIR/wandb"

# Paper code-task effective batch: 8 GPUs * batch 4 * accum 2 = 64.
# Single 3090 setting: 1 GPU * batch 1 * accum 64 = 64.
deets="gpu=${GPU} seed=${SEED} effective_batch=64 warmup=0.03"
echo "Launching GSLoRA HumanEval (${deets})"

deepspeed --include localhost:${GPU} --master_port ${MASTER_PORT} \
  --enable_each_rank_log "$RANK_LOG_DIR" \
  examples/NLG/run_code_humaneval_gslora.py \
  --model_name_or_path /workspace/Models/Llama-3.1-8B-Base \
  --output_dir "$OUTPUT_DIR" \
  --train_dataset_name m-a-p/CodeFeedback-Filtered-Instruction \
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
  --init_method svd_a_zero_b \
  --init_scale 1.0 \
  --scaling_mode rank \
  --calibration_steps 64 \
  --max_length 1024 \
  --max_new_tokens 512 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 64 \
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
  --wandb_mode online \
  --wandb_dir "$LOG_DIR/wandb" \
  --wandb_log_interval 10 \
  --trust_remote_code
