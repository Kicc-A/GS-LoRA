#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=2 python examples/vision/openai_clip_sun397_lora.py \
  --method lora_official \
  --dataset svhn \
  --svhn-root /workspace/datasets/svhn \
  --svhn-single-digit-only \
  --clip-cache-root /root/.cache/clip \
  --output-dir outputs/vision/svhn_openai_clip_lora_r8_seed42_ep1 \
  --epochs 1 \
  --batch-size 64 \
  --eval-batch-size 64 \
  --lr 1e-4 \
  --weight-decay 0 \
  --warmup-ratio 0.03 \
  --base-rank 8 \
  --alpha 16 \
  --dropout 0.0 \
  --loraplus-lr-ratio 1.0 \
  --seed 42
