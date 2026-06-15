#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=2 python /workspace/GS-LoRA/examples/vision/openai_clip_sun397_lora.py \
  --method gs_lora \
  --dataset dtd \
  --dtd-root /workspace/KeepLoRA/MTIL/data/DTD \
  --dtd-split-file /workspace/KeepLoRA/MTIL/data/DTD/split_zhou_DescribableTextures.json \
  --dtd-train-split train \
  --dtd-test-split test \
  --clip-cache-root /root/.cache/clip \
  --output-dir /workspace/GS-LoRA/outputs/vision/dtd_gslora_svda_zero_b_ep1_seed42 \
  --epochs 1 \
  --batch-size 64 \
  --eval-batch-size 64 \
  --lr 1e-4 \
  --weight-decay 0 \
  --warmup-ratio 0.03 \
  --base-rank 8 \
  --alpha 16 \
  --dropout 0.0 \
  --tau 0.90 \
  --r-min 4 \
  --r-max 32 \
  --rank-budget-mode param \
  --calibration-steps 16 \
  --init-method svd_a_zero_b \
  --svd-scale 5e-2 \
  --loraplus-lr-ratio 16.0 \
  --seed 42
