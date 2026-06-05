#!/usr/bin/env bash
set -euo pipefail

source /root/anaconda3/etc/profile.d/conda.sh
conda activate gslora

cd /workspace/GS-LoRA
export PYTHONPATH=/workspace/GS-LoRA:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR=/workspace/GS-LoRA/outputs_gslora_paper_hparams_base_metamathqa_gsm8k_scaling1
RANK_LOG_DIR=/workspace/GS-LoRA/logs_gslora_paper_hparams_base_metamathqa_gsm8k_scaling1_ranks

mkdir -p "$OUTPUT_DIR" "$RANK_LOG_DIR"

deepspeed --include localhost:0,2 --master_port 16669 \
  --enable_each_rank_log "$RANK_LOG_DIR" \
  examples/NLG/run_math_gslora.py \
  --model_name_or_path /workspace/Models/Llama-3.1-8B-Base \
  --output_dir "$OUTPUT_DIR" \
  --train_dataset_name /workspace/GoRA/runtime_datasets/metamathqa_100k \
  --train_split train \
  --train_question_column input \
  --train_answer_column output \
  --eval_dataset_name /workspace/GoRA/runtime_datasets/gsm8k \
  --eval_dataset_config "" \
  --eval_split test \
  --eval_question_column input \
  --eval_answer_column output \
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
  --calibration_steps 32 \
  --max_length 1024 \
  --max_new_tokens 256 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 5e-5 \
  --weight_decay 5e-4 \
  --warmup_ratio 0.3 \
  --max_grad_norm 1.0 \
  --bf16 \
  --gradient_checkpointing \
  --seed 42 \
  --num_workers 0 \
  --trust_remote_code
