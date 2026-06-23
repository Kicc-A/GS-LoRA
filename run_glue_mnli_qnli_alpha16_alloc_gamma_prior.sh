#!/usr/bin/env bash
set -euo pipefail

cd /workspace/GS-LoRA

GPU="${GPU:-0}"
SEEDS="${SEEDS:-42}"
TASKS="${TASKS:-mnli qnli}"

ALLOC_GAMMA="${ALLOC_GAMMA:-1.5}"
ATTN_PRIOR="${ATTN_PRIOR:-1.2}"

for SEED in $SEEDS; do
  for TASK in $TASKS; do
    SCALE=0.9
    BETA=0.5
    DROPOUT=0.05

    CUDA_VISIBLE_DEVICES="$GPU" python run_gslora_glue_t5.py \
      --task_name "$TASK" \
      --model_name_or_path /workspace/Models/T5-Base \
      --glue_root /workspace/datasets/glue \
      --output_dir "outputs/${TASK}_smallb_alpha16_allocgamma${ALLOC_GAMMA}_attnprior${ATTN_PRIOR}_scale${SCALE}_beta${BETA}_rmin4_rmax32_param_loraplus_seed${SEED}" \
      --num_train_epochs 1 \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 32 \
      --eval_accumulation_steps 32 \
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
      --lora_dropout "$DROPOUT" \
      --calibration_steps 64 \
      --init_method svd_a_energy_small_b \
      --init_scale "$SCALE" \
      --init_energy_beta "$BETA" \
      --init_small_b_scale 1e-4 \
      --scaling_mode avg_rank \
      --rank_budget_mode param \
      --param_budget 3244032 \
      --rank_score_gamma "$ALLOC_GAMMA" \
      --attention_rank_prior "$ATTN_PRIOR" \
      --use_loraplus \
      --loraplus_lr_ratio 16 \
      --seed "$SEED"
  done
done
