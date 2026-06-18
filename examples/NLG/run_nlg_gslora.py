import argparse
import json
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from gs_lora import GSLoraConfig, prepare_gslora_model


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal CausalLM GS-LoRA reuse template.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--target_prefix", type=str, default=None)
    parser.add_argument("--base_rank", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--r_min", type=int, default=2)
    parser.add_argument("--r_max", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--init_method", choices=["none", "svd_sqrt", "svd_sigma", "svd_a_zero_b", "svd_a_energy_zero_b", "svd_a_energy_small_b"], default="svd_a_zero_b")
    parser.add_argument("--init_scale", type=float, default=1e-3)
    parser.add_argument("--init_energy_beta", type=float, default=0.5)
    parser.add_argument("--init_energy_eps", type=float, default=1e-8)
    parser.add_argument("--init_small_b_scale", type=float, default=1e-4)
    parser.add_argument("--scaling_mode", choices=["rank", "sqrt_rank", "avg_rank"], default="rank")
    parser.add_argument("--calibration_steps", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=128)
    parser.add_argument("--skip_adaptive_rank", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.train_split)
    if args.max_train_samples is not None and args.max_train_samples < len(dataset):
        dataset = dataset.select(range(args.max_train_samples))

    def tokenize(examples):
        return tokenizer(
            examples[args.text_column],
            truncation=True,
            max_length=args.block_size,
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    config = GSLoraConfig(
        target_modules=args.target_modules,
        target_prefix=args.target_prefix,
        base_rank=args.base_rank,
        tau=args.tau,
        r_min=args.r_min,
        r_max=args.r_max,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        calibration_steps=args.calibration_steps,
        adaptive_rank=not args.skip_adaptive_rank,
        init_method=args.init_method,
        init_scale=args.init_scale,
        init_energy_beta=args.init_energy_beta,
        init_energy_eps=args.init_energy_eps,
        init_small_b_scale=args.init_small_b_scale,
        scaling_mode=args.scaling_mode,
    )

    def loss_fn(current_model, batch):
        batch = {key: value.to(device) for key, value in batch.items()}
        return current_model(**batch).loss

    model, gs_report = prepare_gslora_model(
        model=model,
        dataloader=train_loader,
        loss_fn=loss_fn,
        config=config,
        device=device,
    )

    with open(os.path.join(args.output_dir, "gs_lora_report.json"), "w", encoding="utf-8") as f:
        json.dump(gs_report, f, indent=2)
    with open(os.path.join(args.output_dir, "rank_pattern.json"), "w", encoding="utf-8") as f:
        json.dump(gs_report["rank_pattern"], f, indent=2)
    with open(os.path.join(args.output_dir, "rank_stats.json"), "w", encoding="utf-8") as f:
        json.dump(gs_report["rank_stats"], f, indent=2)

    model.save_pretrained(os.path.join(args.output_dir, "adapter"))


if __name__ == "__main__":
    main()
