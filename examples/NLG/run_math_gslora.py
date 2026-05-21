import argparse
import json
import os
import random
import re
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from gs_lora import GSLoraConfig, prepare_gslora_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a CausalLM with GS-LoRA on MetaMathQA and evaluate on GSM8K."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_dataset_name", type=str, default="meta-math/MetaMathQA")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--train_question_column", type=str, default="query")
    parser.add_argument("--train_answer_column", type=str, default="response")
    parser.add_argument("--eval_dataset_name", type=str, default="gsm8k")
    parser.add_argument("--eval_dataset_config", type=str, default="main")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--eval_question_column", type=str, default="question")
    parser.add_argument("--eval_answer_column", type=str, default="answer")
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--target_prefix", type=str, default=None)
    parser.add_argument("--base_rank", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--r_min", type=int, default=2)
    parser.add_argument("--r_max", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--init_method", choices=["none", "svd_sqrt", "svd_sigma"], default="none")
    parser.add_argument("--init_scale", type=float, default=1e-3)
    parser.add_argument("--no_compensate_scaling", action="store_true")
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--skip_adaptive_rank", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_prompt(question: str) -> str:
    return f"Question:\n{question.strip()}\n\nAnswer:\n"


def maybe_select(dataset, max_samples: Optional[int], seed: int):
    if max_samples is not None and max_samples < len(dataset):
        return dataset.shuffle(seed=seed).select(range(max_samples))
    return dataset


def tokenize_sft_example(example, tokenizer, args):
    question = str(example[args.train_question_column])
    answer = str(example[args.train_answer_column])
    prompt = format_prompt(question)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]

    max_answer_length = min(len(answer_ids), args.max_length)
    max_prompt_length = max(args.max_length - max_answer_length, 0)
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:] if max_prompt_length > 0 else []

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    if len(input_ids) > args.max_length:
        input_ids = input_ids[-args.max_length :]
        labels = labels[-args.max_length :]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class SupervisedDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, List[int]]]):
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def extract_gsm8k_answer(text: str) -> str:
    if "####" in text:
        text = text.split("####")[-1]
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not matches:
        return text.strip()
    return matches[-1].replace(",", "").strip()


def train(model, train_loader, args, device: torch.device):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps_per_epoch = max(1, len(train_loader) // args.gradient_accumulation_steps)
    total_steps = args.num_train_epochs * update_steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16
    use_autocast = args.fp16 or args.bf16

    model.train()
    completed_steps = 0
    for epoch in range(args.num_train_epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}")
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, batch in enumerate(progress):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                loss = model(**batch).loss
                scaled_loss = loss / args.gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1

            running_loss += loss.item()
            progress.set_postfix(loss=running_loss / max(step + 1, 1), steps=completed_steps)


@torch.no_grad()
def evaluate_gsm8k(model, tokenizer, eval_dataset, args, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    predictions = []

    for start in tqdm(range(0, len(eval_dataset), args.per_device_eval_batch_size), desc="GSM8K eval"):
        batch = eval_dataset[start : start + args.per_device_eval_batch_size]
        questions = batch[args.eval_question_column]
        answers = batch[args.eval_answer_column]
        prompts = [format_prompt(question) for question in questions]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for index, output_ids in enumerate(generated):
            new_tokens = output_ids[int(prompt_lengths[index]) :]
            prediction_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            pred_answer = extract_gsm8k_answer(prediction_text)
            gold_answer = extract_gsm8k_answer(str(answers[index]))
            is_correct = pred_answer == gold_answer
            correct += int(is_correct)
            total += 1
            predictions.append(
                {
                    "question": questions[index],
                    "prediction": prediction_text,
                    "pred_answer": pred_answer,
                    "gold_answer": gold_answer,
                    "correct": is_correct,
                }
            )

    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype).to(device)

    train_dataset = load_dataset(args.train_dataset_name, split=args.train_split)
    eval_dataset = load_dataset(args.eval_dataset_name, args.eval_dataset_config, split=args.eval_split)
    train_dataset = maybe_select(train_dataset, args.max_train_samples, args.seed)
    eval_dataset = maybe_select(eval_dataset, args.max_eval_samples, args.seed)

    tokenized_train = train_dataset.map(
        lambda example: tokenize_sft_example(example, tokenizer, args),
        remove_columns=train_dataset.column_names,
    )
    collator = SupervisedDataCollator(tokenizer)
    train_loader = DataLoader(
        tokenized_train,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
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
        compensate_scaling=not args.no_compensate_scaling,
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
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    with open(os.path.join(args.output_dir, "gs_lora_report.json"), "w", encoding="utf-8") as f:
        json.dump(gs_report, f, indent=2)
    with open(os.path.join(args.output_dir, "rank_pattern.json"), "w", encoding="utf-8") as f:
        json.dump(gs_report["rank_pattern"], f, indent=2)
    with open(os.path.join(args.output_dir, "rank_stats.json"), "w", encoding="utf-8") as f:
        json.dump(gs_report["rank_stats"], f, indent=2)
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train(model, train_loader, args, device)
    model.save_pretrained(os.path.join(args.output_dir, "last_adapter"))
    tokenizer.save_pretrained(args.output_dir)

    eval_metrics = evaluate_gsm8k(model, tokenizer, eval_dataset, args, device)
    predictions = eval_metrics.pop("predictions")
    with open(os.path.join(args.output_dir, "gsm8k_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
    with open(os.path.join(args.output_dir, "gsm8k_predictions.jsonl"), "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
