#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/GS-LoRA
SEED=${SEED:-42}
EXP_NAME=${EXP_NAME:-llama31_metamathqa_gslora_sqrtrank_gpu0123}
WANDB_NAME=${WANDB_NAME:-gslora-llama31-metamathqa-sqrtrank-gpu0123-seed${SEED}}
EXP=$ROOT/experiments/$EXP_NAME
LOG=$EXP/logs/train.log

mkdir -p "$EXP/logs"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

exec /root/anaconda3/envs/gslora/bin/deepspeed --num_gpus 4 \
  examples/NLG/run_math_gslora.py \
  --model_name_or_path /workspace/Models/Llama-3.1-8B-Base \
  --output_dir "$EXP/output" \
  --train_dataset_name /workspace/MyTransformers/data/metamathqa_lorapro/metamathqa_lorapro_gsm_tok512_train100k.input_output.jsonl \
  --train_question_column input \
  --train_answer_column output \
  --eval_dataset_name /workspace/MyTransformers/data/metamathqa_lorapro/metamathqa_lorapro_gsm_tok512_eval10k.input_output.jsonl \
  --eval_dataset_config '' \
  --eval_question_column input \
  --eval_answer_column output \
  --dataset_prefix '' \
  --dataset_postfix '' \
  --target_modules q_proj k_proj v_proj o_proj \
  --base_rank 8 \
  --r_min 4 \
  --r_max 32 \
  --rank_budget_mode param \
  --lora_alpha 16 \
  --lora_dropout 0.0 \
  --use_lora_plus \
  --lora_plus_scaler 16 \
  --init_method svd_a_energy_small_b \
  --init_scale auto \
  --init_auto_target_ratio 0.08 \
  --init_energy_beta 0.5 \
  --init_small_b_scale 1e-4 \
  --scaling_mode sqrt_rank \
  --calibration_steps 128 \
  --max_length 1024 \
  --max_src_len 1024 \
  --max_new_tokens 1024 \
  --no_pad_to_max_length \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --weight_decay 5e-4 \
  --warmup_ratio 0.03 \
  --seed "$SEED" \
  --bf16 \
  --gradient_checkpointing \
  --trust_remote_code \
  --wandb \
  --wandb_project GSLoRA-MetaMathQA \
  --wandb_name "$WANDB_NAME" \
  --wandb_mode online \
  --wandb_dir "$ROOT/wandb" \
  2>&1 | tee "$LOG"
