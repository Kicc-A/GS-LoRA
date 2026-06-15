#!/usr/bin/env bash
set -e

export PYTHONPATH=/workspace/GS-LoRA:${PYTHONPATH:-}

CUDA_VISIBLE_DEVICES=2 python /workspace/GS-LoRA/examples/vision/gs_lora_clip_classification.py \
  --model_name_or_path /workspace/Models/clip-vit-base-patch16-safetensors \
  --zhou_root /workspace/KeepLoRA/MTIL/data/DTD/images \
  --zhou_split_file /workspace/KeepLoRA/MTIL/data/DTD/split_zhou_DescribableTextures.json \
  --train_split train \
  --test_split test \
  --output_dir /workspace/GS-LoRA/outputs/vision/dtd_gslora_gora_pinv_ep1_seed42 \
  --prompt_template "a photo of a {}." \
  --target_modules q_proj v_proj \
  --target_prefix vision_model \
  --base_rank 8 \
  --tau 0.90 \
  --r_min 4 \
  --r_max 32 \
  --lora_alpha 16 \
  --lora_dropout 0.0 \
  --init_method gora_pinv \
  --init_scale 5e-2 \
  --gora_stable_gamma 5e-2 \
  --scaling_mode rank \
  --calibration_steps 64 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --learning_rate 1e-4 \
  --loraplus_lr_ratio 16 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --seed 42 \
  --num_workers 4
