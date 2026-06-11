#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import evaluate
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    set_seed,
)

_GSLORA_AVAILABLE = False
_GSLORA_PREPARE = None
try:
    from gs_lora import prepare_gslora_model, GSLoraConfig  # type: ignore
    _GSLORA_AVAILABLE = True
    _GSLORA_PREPARE = prepare_gslora_model
except Exception:
    try:
        from gs_lora.api import prepare_gslora_model  # type: ignore
        from gs_lora.config import GSLoraConfig  # type: ignore
        _GSLORA_AVAILABLE = True
        _GSLORA_PREPARE = prepare_gslora_model
    except Exception:
        _GSLORA_AVAILABLE = False
        _GSLORA_PREPARE = None
        GSLoraConfig = None

try:
    from peft import LoraConfig, TaskType, get_peft_model
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False

# GoRA/LoRA-GA T5 GLUE setting.
TASK_TO_KEYS: Dict[str, Tuple[str, Optional[str]]] = {
    "mnli": ("premise", "hypothesis"),
    "sst2": ("sentence", None),
    "cola": ("sentence", None),
    "qnli": ("question", "sentence"),
    "mrpc": ("sentence1", "sentence2"),
}

LABEL_TEXT = {
    "sst2": {0: "negative", 1: "positive"},
    "cola": {0: "unacceptable", 1: "acceptable"},
    "qnli": {0: "entailment", 1: "not_entailment"},
    "mrpc": {0: "not_equivalent", 1: "equivalent"},
    "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
}


TASK_INSTRUCTION = {
    "mnli": "classify the semantic similarity of the text: ",
    "sst2": "classify the sentiment of the text: ",
    "cola": "classify the grammaticality of the text: ",
    "qnli": "classify the semantic similarity of the question and the sentence: ",
    "mrpc": "classify the semantic similarity of the text: ",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="/workspace/Models/T5-Base")
    p.add_argument("--task_name", type=str, required=True, choices=sorted(TASK_TO_KEYS.keys()))
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--glue_root", type=str, default="/workspace/datasets/glue")
    p.add_argument("--max_source_length", type=int, default=128)
    p.add_argument("--max_target_length", type=int, default=32)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=32)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--evaluation_strategy", type=str, default="epoch")
    p.add_argument("--save_strategy", type=str, default="epoch")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--generation_max_length", type=int, default=32)

    # GS-LoRA / LoRA args. Keep GS-LoRA's adaptive-rank and SVD-init path as the method under test.
    p.add_argument("--target_modules", nargs="+", default=["q", "k", "v", "o", "wi", "wo"])
    p.add_argument("--tau", type=float, default=0.90)
    p.add_argument("--r_min", type=int, default=2)
    p.add_argument("--r_max", type=int, default=16)
    p.add_argument("--base_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--calibration_steps", type=int, default=64)
    p.add_argument("--init_method", type=str, default="svd_a_zero_b", choices=["none", "svd_sqrt", "svd_sigma", "svd_a_zero_b"])
    p.add_argument("--rank_budget_mode", type=str, default="independent", choices=["independent", "param"])
    p.add_argument("--param_budget", type=int, default=None)
    p.add_argument("--init_scale", type=float, default=1.0)
    p.add_argument("--scaling_mode", type=str, default="rank", choices=["rank", "sqrt_rank", "avg_rank"])
    p.add_argument("--skip_adaptive_rank", action="store_true")
    p.add_argument("--no_gslora", action="store_true")
    p.add_argument("--plain_lora_init", type=str, default="peft", choices=["peft", "gora"])
    p.add_argument("--use_loraplus", action="store_true")
    p.add_argument("--loraplus_lr_ratio", type=float, default=16.0)
    return p.parse_args()


def build_prompt(task_name: str, ex: dict) -> str:
    a, b = TASK_TO_KEYS[task_name]
    prompt = f"{TASK_INSTRUCTION[task_name]}{ex[a]}"
    if b is not None:
        prompt += f"\n{ex[b]}"
    return prompt + "\nresult: "


def preprocess_datasets(task_name: str, tokenizer, max_source_length: int, max_target_length: int, glue_root: str):
    local_task_dir = Path(glue_root) / task_name
    if local_task_dir.exists():
        print(f"[INFO] Loading local GLUE dataset from {local_task_dir}")
        raw_all = load_from_disk(str(local_task_dir))
    else:
        print(f"[INFO] Local GLUE dataset not found at {local_task_dir}, fallback to Hugging Face")
        raw_all = load_dataset("glue", task_name)

    def preprocess(examples):
        inputs, targets = [], []
        for i in range(len(examples["label"])):
            ex = {k: examples[k][i] for k in examples.keys()}
            inputs.append(build_prompt(task_name, ex))
            label_id = int(examples["label"][i])

            # GLUE test split has label = -1. We do not train/eval on test,
            # but keep preprocessing robust.
            if label_id == -1:
                targets.append(list(LABEL_TEXT[task_name].values())[0])
            else:
                targets.append(LABEL_TEXT[task_name][label_id])

        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
            padding="max_length",
        )
        try:
            labels = tokenizer(
                text_target=targets,
                max_length=max_target_length,
                truncation=True,
                padding="max_length",
            )
        except TypeError:
            labels = tokenizer(
                targets,
                max_length=max_target_length,
                truncation=True,
                padding="max_length",
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    keep_splits = ["train"]
    if task_name == "mnli":
        keep_splits += ["validation_matched", "validation_mismatched"]
    else:
        keep_splits += ["validation"]

    raw = {k: raw_all[k] for k in keep_splits}
    tokenized = {}
    for split_name, ds in raw.items():
        tokenized[split_name] = ds.map(
            preprocess,
            batched=True,
            remove_columns=ds.column_names,
        )
    return raw, tokenized


def build_model(args, tokenizer, train_dataset):
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    rank_summary = None
    mode = "plain_lora"

    print("[DEBUG] _GSLORA_AVAILABLE =", _GSLORA_AVAILABLE)
    print("[DEBUG] _GSLORA_PREPARE =", _GSLORA_PREPARE)
    print("[DEBUG] no_gslora =", args.no_gslora)

    if not args.no_gslora and _GSLORA_AVAILABLE and _GSLORA_PREPARE is not None:
        print("[DEBUG] Trying GS-LoRA path ...")
        collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        )

        n = min(len(train_dataset), max(args.calibration_steps * args.per_device_train_batch_size, 64))
        subset = train_dataset.shuffle(seed=args.seed).select(range(n))
        print(f"[DEBUG] calibration samples = {n}")
        print(f"[DEBUG] target_modules = {args.target_modules}")
        print(f"[DEBUG] tau={args.tau}, r_min={args.r_min}, r_max={args.r_max}, base_rank={args.base_rank}")
        batches, buf = [], []
        for item in subset:
            buf.append(item)
            if len(buf) >= args.per_device_train_batch_size:
                batches.append(collator(buf))
                buf = []
        if buf:
            batches.append(collator(buf))

        def loss_fn(m, batch):
            outputs = m(**batch)
            return outputs.loss

        try:
            gslora_config = GSLoraConfig(
                target_modules=args.target_modules,
                base_rank=args.base_rank,
                tau=args.tau,
                r_min=args.r_min,
                r_max=args.r_max,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                calibration_steps=args.calibration_steps,
                adaptive_rank=(not args.skip_adaptive_rank),
                bias="none",
                init_method=args.init_method,
                rank_budget_mode=args.rank_budget_mode,
                param_budget=args.param_budget,
                init_scale=args.init_scale,
                compensate_scaling=True,
                scaling_mode=args.scaling_mode,
            )
            print("[DEBUG] GSLoRAConfig =", gslora_config)

            prepared = _GSLORA_PREPARE(
                model=model,
                dataloader=batches,
                loss_fn=loss_fn,
                config=gslora_config,
                device=None,
            )

            if isinstance(prepared, tuple):
                model = prepared[0]
                report = prepared[1] if len(prepared) > 1 else {}
            else:
                model = prepared
                report = {}

            rank_summary = report.get("rank_summary", None)
            mode = "gslora"
            print("[DEBUG] GS-LoRA successfully prepared.")
            print("[DEBUG] report =", report)
            print("[DEBUG] rank_summary =", rank_summary)
            return model, rank_summary, mode
        except Exception as e:
            print("[ERROR] GS-LoRA prepare failed:", repr(e))
            raise

    if not _PEFT_AVAILABLE:
        raise RuntimeError("GS-LoRA API unavailable and PEFT is not installed.")

    print("[DEBUG] Falling back to plain PEFT LoRA.")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.base_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    if args.plain_lora_init == "gora":
        init_plain_lora_like_gora(model)
    return model, rank_summary, mode


def init_plain_lora_like_gora(model):
    a_numel = 0
    b_numel = 0
    for module in model.modules():
        if hasattr(module, "lora_A"):
            for adapter_layer in module.lora_A.values():
                weight = adapter_layer.weight
                torch.nn.init.normal_(weight, mean=0.0, std=1.0 / (weight.shape[1] ** 0.5))
                a_numel += weight.numel()
        if hasattr(module, "lora_B"):
            for adapter_layer in module.lora_B.values():
                torch.nn.init.zeros_(adapter_layer.weight)
                b_numel += adapter_layer.weight.numel()
    print("[DEBUG] plain_lora_init = gora")
    print("[DEBUG] GoRA-style plain LoRA A numel =", a_numel)
    print("[DEBUG] GoRA-style plain LoRA B numel =", b_numel)


def normalize_text(s: str) -> str:
    return s.strip().lower()


def compute_metrics_builder(task_name: str, tokenizer):
    metric = evaluate.load("glue", task_name)
    inv = {normalize_text(v): k for k, v in LABEL_TEXT[task_name].items()}

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        pred_ids, label_ids = [], []
        for pred_text, label_text in zip(decoded_preds, decoded_labels):
            pred_ids.append(inv.get(normalize_text(pred_text), -1))
            label_ids.append(inv[normalize_text(label_text)])

        return metric.compute(predictions=pred_ids, references=label_ids)

    return compute_metrics


def build_loraplus_optimizer(model, learning_rate: float, lr_ratio: float, weight_decay: float):
    lora_a_params = []
    lora_b_params = []
    other_params = []
    lora_a_numel = 0
    lora_b_numel = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_B" in name:
            lora_b_params.append(param)
            lora_b_numel += param.numel()
        elif "lora_A" in name:
            lora_a_params.append(param)
            lora_a_numel += param.numel()
        else:
            other_params.append(param)

    b_lr = learning_rate * lr_ratio
    print("[DEBUG] use_loraplus = True")
    print("[DEBUG] loraplus_lr_ratio =", lr_ratio)
    print("[DEBUG] LoRA A numel =", lora_a_numel)
    print("[DEBUG] LoRA B numel =", lora_b_numel)
    print("[DEBUG] LoRA B lr =", b_lr)

    param_groups = []
    if lora_a_params:
        param_groups.append({"params": lora_a_params, "lr": learning_rate, "weight_decay": weight_decay})
    if lora_b_params:
        param_groups.append({"params": lora_b_params, "lr": b_lr, "weight_decay": weight_decay})
    if other_params:
        param_groups.append({"params": other_params, "lr": learning_rate, "weight_decay": weight_decay})
    if not param_groups:
        raise RuntimeError("No trainable parameters found for LoRA+ optimizer.")

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    raw, tok = preprocess_datasets(
        args.task_name,
        tokenizer,
        args.max_source_length,
        args.max_target_length,
        args.glue_root,
    )

    eval_dataset = tok["validation_matched"] if args.task_name == "mnli" else tok["validation"]
    extra_eval = {"validation_mismatched": tok["validation_mismatched"]} if args.task_name == "mnli" else {}

    model, rank_summary, mode = build_model(args, tokenizer, tok["train"])
    print("[DEBUG] final mode =", mode)
    print("[DEBUG] final rank_summary =", rank_summary)
    print("[DEBUG] plain_lora_init =", args.plain_lora_init)
    print("[DEBUG] use_loraplus =", args.use_loraplus)
    print("[DEBUG] loraplus_lr_ratio =", args.loraplus_lr_ratio)
    print("[DEBUG] init_method =", args.init_method)
    print("[DEBUG] init_scale =", args.init_scale)
    print("[DEBUG] rank_budget_mode =", args.rank_budget_mode)
    print("[DEBUG] param_budget =", args.param_budget)
    avg_rank = (rank_summary or {}).get("avg_rank") if isinstance(rank_summary, dict) else None
    print("[DEBUG] avg_rank =", avg_rank)
    if args.use_loraplus and avg_rank is not None and float(avg_rank) < 7.0:
        raise RuntimeError(f"avg_rank={avg_rank} < 7.0; stop before training.")

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    import inspect

    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        report_to=[] if args.report_to == "none" else [args.report_to],
        load_best_model_at_end=(args.evaluation_strategy != "no"),
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
    )

    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = args.evaluation_strategy
    elif "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = args.evaluation_strategy

    if "overwrite_output_dir" in sig.parameters:
        kwargs["overwrite_output_dir"] = True

    train_args = Seq2SeqTrainingArguments(**kwargs)

    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=tok["train"],
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics_builder(args.task_name, tokenizer),
    )
    if args.use_loraplus:
        optimizer = build_loraplus_optimizer(
            model,
            learning_rate=args.learning_rate,
            lr_ratio=args.loraplus_lr_ratio,
            weight_decay=args.weight_decay,
        )
        trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
        if "optimizers" not in trainer_sig.parameters:
            raise RuntimeError("This Seq2SeqTrainer does not support optimizers=(optimizer, scheduler).")
        trainer_kwargs["optimizers"] = (optimizer, None)

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    train_result = trainer.train()
    trainer.save_model()

    metrics = {
        "train": train_result.metrics,
        "eval": trainer.evaluate(eval_dataset=eval_dataset),
    }
    for name, ds in extra_eval.items():
        metrics[name] = trainer.evaluate(eval_dataset=ds)

    Path(args.output_dir, "glue_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    Path(args.output_dir, "glue_run_config.json").write_text(
        json.dumps(
            {"mode": mode, "rank_summary": rank_summary, "args": vars(args)},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
