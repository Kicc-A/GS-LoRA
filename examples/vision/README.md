# GS-LoRA CLIP Image Classification

This example fine-tunes CLIP on image classification datasets such as DTD and
Food101 with gradient-SVD adaptive-rank LoRA.

The workflow is:

1. Load a CLIP model and a classification dataset.
2. Build class prompts such as `a photo of {}.`.
3. Run a small calibration backward pass on the vision encoder target weights.
4. Compute a per-layer `rank_pattern` from gradient singular-value energy.
5. Inject PEFT LoRA with that `rank_pattern` into vision encoder layers only.
6. Train only LoRA parameters and evaluate top-1 accuracy.

## Install

```bash
pip install torch transformers datasets peft accelerate tqdm pillow
pip install -e ../..
```

## DTD

```bash
python examples/vision/gs_lora_clip_classification.py \
  --model_name_or_path /path/to/clip-vit-base-patch16 \
  --dataset_name dtd \
  --output_dir outputs/gs_lora_dtd \
  --prompt_template "a photo of a {} texture." \
  --target_modules q_proj v_proj out_proj \
  --tau 0.90 \
  --r_min 2 \
  --r_max 16 \
  --base_rank 8 \
  --lora_alpha 16 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 32
```

## Food101

```bash
python examples/vision/gs_lora_clip_classification.py \
  --model_name_or_path /path/to/clip-vit-base-patch16 \
  --dataset_name food101 \
  --output_dir outputs/gs_lora_food101 \
  --prompt_template "a photo of {}." \
  --target_modules q_proj v_proj out_proj \
  --tau 0.90 \
  --r_min 2 \
  --r_max 16 \
  --base_rank 8 \
  --lora_alpha 16 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 32
```

For local datasets arranged as `imagefolder`, use:

```bash
python examples/vision/gs_lora_clip_classification.py \
  --model_name_or_path /path/to/clip-vit-base-patch16 \
  --dataset_dir /path/to/Food101 \
  --output_dir outputs/gs_lora_food101_local
```

To run fixed-r LoRA as a baseline, add `--skip_adaptive_rank`. The script will
then use `--base_rank` for every target layer.

Outputs include:

- `rank_pattern.json`: PEFT-compatible module-name to rank mapping.
- `rank_stats.json`: retained gradient-energy statistics.
- `run_config.json`: resolved class prompts and target module names.
- `best_adapter/`: best PEFT adapter by validation accuracy.
- `last_adapter/`: final PEFT adapter.
- `metrics.json`: validation accuracy and loss.
