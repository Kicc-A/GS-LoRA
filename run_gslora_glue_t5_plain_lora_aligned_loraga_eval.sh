#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/workspace/Models/T5-Base}"
GLUE_ROOT="${GLUE_ROOT:-/workspace/datasets/glue}"
OUT_ROOT="${OUT_ROOT:-outputs/glue_t5_plain_lora_aligned_loraga_eval_scale0.01}"
TASKS="${TASKS:-mnli sst2 cola qnli mrpc}"
SEEDS="${SEEDS:-42}"

mkdir -p "${OUT_ROOT}"

for SEED in ${SEEDS}; do
  for TASK in ${TASKS}; do
    echo "=============================="
    echo "TASK=${TASK}  SEED=${SEED}"
    echo "=============================="

    python run_gslora_glue_t5.py \
      --model_name_or_path "${MODEL}" \
      --glue_root "${GLUE_ROOT}" \
      --task_name "${TASK}" \
      --output_dir "${OUT_ROOT}/${TASK}_seed${SEED}" \
      --num_train_epochs 1 \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 32 \
      --learning_rate 1e-4 \
      --weight_decay 0 \
      --warmup_ratio 0.03 \
      --lr_scheduler_type cosine \
      --max_source_length 128 \
      --max_target_length 32 \
      --generation_max_length 32 \
      --glue_format loraga \
      --metric_style loraga_logits \
      --no_mask_label_padding \
      --target_modules q k v o wi wo \
      --tau 0.90 \
      --r_min 4 \
      --r_max 32 \
      --base_rank 8 \
      --lora_alpha 16 \
      --lora_dropout 0.05 \
      --calibration_steps 64 \
      --init_method svd_a_zero_b \
      --init_scale 0.01 \
      --scaling_mode rank \
      --no_gslora \
      --seed "${SEED}"
  done
done
