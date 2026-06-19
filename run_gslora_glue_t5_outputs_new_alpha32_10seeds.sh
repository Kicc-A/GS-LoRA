#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/workspace/Models/T5-Base}"
GLUE_ROOT="${GLUE_ROOT:-/workspace/datasets/glue}"
OUT_ROOT="${OUT_ROOT:-outputs_new}"
ALPHA="${ALPHA:-32}"
SEEDS="${SEEDS:-0 13 21 31 42 87 100 123 2024 3407}"
TASKS="${TASKS:-mnli sst2 cola qnli mrpc}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUT_ROOT}"

run_task() {
  local task="$1"
  local seed="$2"

  local init_method="svd_a_energy_small_b"
  local init_scale="0.7"
  local beta="0.25"
  local small_b="1e-4"
  local dropout="0.05"
  local lr="1e-4"
  local r_min="4"
  local r_max="32"
  local base_rank="8"
  local tau="0.90"
  local rank_budget_mode="param"
  local param_budget="3244032"
  local scaling_mode="avg_rank"
  local calibration_steps="64"
  local glue_format="loraga"
  local metric_style="loraga_logits"
  local mask_flag="--no_mask_label_padding"
  local use_loraplus="true"
  local loraplus_ratio="16"

  case "${task}" in
    cola)
      init_method="svd_a_energy_small_b"
      init_scale="1.2"
      beta="0.6"
      dropout="0.0"
      use_loraplus="true"
      ;;
    sst2)
      init_method="svd_a_energy_small_b"
      init_scale="0.5"
      beta="0.25"
      dropout="0.0"
      use_loraplus="true"
      ;;
    mrpc)
      init_method="svd_a_energy_small_b"
      init_scale="0.7"
      beta="0.25"
      dropout="0.05"
      use_loraplus="true"
      ;;
    mnli)
      init_method="svd_a_energy_small_b"
      init_scale="0.7"
      beta="0.25"
      dropout="0.05"
      use_loraplus="true"
      ;;
    qnli)
      init_method="svd_a_zero_b"
      init_scale="1.0"
      beta="0.5"
      dropout="0.05"
      r_min="2"
      r_max="16"
      rank_budget_mode="independent"
      param_budget=""
      scaling_mode="rank"
      glue_format="gs"
      metric_style="official_generate"
      mask_flag=""
      use_loraplus="true"
      ;;
    *)
      echo "Unsupported task: ${task}" >&2
      return 1
      ;;
  esac

  local out_dir="${OUT_ROOT}/${task}_gslora_best_alpha${ALPHA}_seed${seed}"

  echo "=============================="
  echo "TASK=${task} SEED=${seed} ALPHA=${ALPHA}"
  echo "OUT=${out_dir}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "=============================="

  local cmd=(
    python run_gslora_glue_t5.py
    --task_name "${task}"
    --model_name_or_path "${MODEL}"
    --glue_root "${GLUE_ROOT}"
    --output_dir "${out_dir}"
    --num_train_epochs 1
    --per_device_train_batch_size 32
    --per_device_eval_batch_size 32
    --eval_accumulation_steps 32
    --learning_rate "${lr}"
    --weight_decay 0
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --max_source_length 128
    --max_target_length 32
    --generation_max_length 32
    --glue_format "${glue_format}"
    --metric_style "${metric_style}"
    --target_modules q k v o wi wo
    --tau "${tau}"
    --r_min "${r_min}"
    --r_max "${r_max}"
    --base_rank "${base_rank}"
    --lora_alpha "${ALPHA}"
    --lora_dropout "${dropout}"
    --calibration_steps "${calibration_steps}"
    --init_method "${init_method}"
    --init_scale "${init_scale}"
    --init_energy_beta "${beta}"
    --init_small_b_scale "${small_b}"
    --scaling_mode "${scaling_mode}"
    --rank_budget_mode "${rank_budget_mode}"
    --seed "${seed}"
  )

  if [[ -n "${mask_flag}" ]]; then
    cmd+=("${mask_flag}")
  fi
  if [[ -n "${param_budget}" ]]; then
    cmd+=(--param_budget "${param_budget}")
  fi
  if [[ "${use_loraplus}" == "true" ]]; then
    cmd+=(--use_loraplus --loraplus_lr_ratio "${loraplus_ratio}")
  fi

  "${cmd[@]}"
}

for seed in ${SEEDS}; do
  for task in ${TASKS}; do
    run_task "${task}" "${seed}"
  done
done
