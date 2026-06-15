"""
Fine-tune HF CLIP on SUN397 with GS-LoRA (gradient-SVD adaptive-rank LoRA).

Uses the official SUN397 partition format (Training_01.txt / Testing_01.txt),
matching the data loading of the OpenAI CLIP script exactly.

Supports two loss modes:
  --use_cls_head        [CLS] token + classifier head  (GoRA paper protocol)
  (default)             standard CLIP contrastive loss with text prompts
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, get_scheduler

from gs_lora import GSLoraConfig, find_target_module_names, prepare_gslora_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune HF CLIP on SUN397 with gradient-SVD adaptive-rank LoRA."
    )
    # --- Model & Data ---
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--sun397_root", type=str, required=True,
                        help="SUN397 root dir containing ClassName.txt and image subdirs")
    parser.add_argument("--sun397_train_list", type=str, required=True,
                        help="Path to Training_01.txt (official split)")
    parser.add_argument("--sun397_test_list", type=str, required=True,
                        help="Path to Testing_01.txt (official split)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt_template", type=str, default="a photo of a {}.")
    parser.add_argument("--use_cls_head", action="store_true",
                        help="Use [CLS] token + classifier head (GoRA paper protocol)")

    # --- GS-LoRA ---
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
    parser.add_argument("--target_prefix", type=str, default="vision_model")
    parser.add_argument("--base_rank", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--r_min", type=int, default=4)
    parser.add_argument("--r_max", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--init_method",
                        choices=["none", "svd_sqrt", "svd_sigma", "svd_a_zero_b", "gora_pinv"],
                        default="svd_a_zero_b")
    parser.add_argument("--init_scale", type=float, default=0.05)
    parser.add_argument("--gora_stable_gamma", type=float, default=5e-2)
    parser.add_argument("--gora_scale_by_lr", action="store_true")
    parser.add_argument("--gora_lr", type=float, default=5e-2)
    parser.add_argument("--no_compensate_scaling", action="store_true")
    parser.add_argument("--scaling_mode", choices=["rank", "sqrt_rank", "avg_rank"], default="rank")
    parser.add_argument("--rank_budget_mode", choices=["independent", "param"], default="param")
    parser.add_argument("--param_budget", type=int, default=None)
    parser.add_argument("--calibration_steps", type=int, default=64)
    parser.add_argument("--skip_adaptive_rank", action="store_true")

    # --- Training ---
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--loraplus_lr_ratio", type=float, default=16.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--train_logit_scale", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            if not getattr(stream, "closed", False):
                stream.flush()


def setup_file_logging(output_dir: str):
    log_path = os.path.join(output_dir, "train.log")
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeLogger(original_stdout, log_file)
    sys.stderr = TeeLogger(original_stderr, log_file)
    print(f"Logging to {log_path}", flush=True)
    return log_file, original_stdout, original_stderr


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# SUN397 official partition format
# ---------------------------------------------------------------------------

def load_sun397_partition_class_names(root: str) -> List[str]:
    """Load SUN397 class names from ClassName.txt (same format as OpenAI CLIP script)."""
    with open(Path(root) / "ClassName.txt", encoding="utf-8") as f:
        return [line.strip().lstrip("/") for line in f if line.strip()]


def load_sun397_partition_samples(root: str, list_path: str) -> List[tuple]:
    """Load samples from a SUN397 partition file (Training_01.txt / Testing_01.txt).

    Each line: a/abbey/sun_xxxx.jpg
    Returns: [(full_path, label_int), ...]
    """
    root = Path(root)
    class_names = load_sun397_partition_class_names(str(root))
    class_to_label = {name: idx for idx, name in enumerate(class_names)}
    samples = []
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            rel = line.strip().lstrip("/")
            if not rel:
                continue
            class_name = str(Path(rel).parent)
            if class_name not in class_to_label:
                raise ValueError(f"Unknown SUN397 class in {list_path}: {class_name}")
            path = root / rel
            if not path.is_file():
                raise FileNotFoundError(path)
            samples.append((str(path), class_to_label[class_name]))
    return samples


class Sun397PartitionDataset(Dataset):
    """SUN397 dataset reading from partition files (Training_01.txt / Testing_01.txt)."""
    def __init__(self, samples: List[tuple], processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[int(idx)]
        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(images=[image], return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pixel_values, "labels": label}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def collate_fn(features):
    pixel_values = []
    for feature in features:
        value = feature["pixel_values"]
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        pixel_values.append(value)
    pixel_values = torch.stack(pixel_values)
    labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


def maybe_subset(dataset, max_samples: Optional[int], seed: int):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return torch.utils.data.Subset(dataset, indices)


# ---------------------------------------------------------------------------
# CLIP helpers
# ---------------------------------------------------------------------------

def get_clip_backbone(model):
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def to_float_tensor(output):
    if torch.is_tensor(output):
        return output.float()
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output.float()
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :].float()
    raise TypeError(f"Cannot convert {type(output)!r} to a tensor.")


@torch.no_grad()
def build_text_features(model, processor, class_names: List[str], prompt_template: str,
                        device: torch.device, batch_size: int = 128) -> torch.Tensor:
    model.eval()
    prompts = []
    for name in class_names:
        # Strip SUN397 subdirectory prefix (e.g., 'a/abbey' -> 'abbey')
        parts = name.split("/")
        clean = "/".join(parts[1:]) if len(parts) > 1 else name
        clean = clean.replace("_", " ").replace("-", " ").replace("/", " ")
        prompts.append(prompt_template.format(clean))
    features = []
    clip_model = get_clip_backbone(model)

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start: start + batch_size]
        inputs = processor(text=batch_prompts, padding=True, return_tensors="pt").to(device)
        text_features = to_float_tensor(clip_model.get_text_features(**inputs))
        text_features = F.normalize(text_features, dim=-1)
        features.append(text_features.cpu())

    return torch.cat(features, dim=0)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def clip_classification_loss(model, pixel_values, labels, text_features):
    """Standard CLIP contrastive loss: image features @ text features."""
    clip_model = get_clip_backbone(model)
    image_features = to_float_tensor(clip_model.get_image_features(pixel_values=pixel_values))
    image_features = F.normalize(image_features, dim=-1)
    logits = image_features @ text_features.t()
    logits = logits * clip_model.logit_scale.exp()
    loss = F.cross_entropy(logits, labels)
    return loss, logits


def cls_classification_loss(model, pixel_values, labels):
    """[CLS] token + classifier head — GoRA paper protocol."""
    clip_model = get_clip_backbone(model)
    vision_outputs = clip_model.vision_model(pixel_values=pixel_values)
    pooled = vision_outputs.pooler_output
    image_features = clip_model.visual_projection(pooled)
    image_features = F.normalize(image_features.float(), dim=-1)
    logits = model.classifier(image_features)
    loss = F.cross_entropy(logits, labels)
    return loss, logits


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, text_features, device: torch.device,
             use_cls_head: bool = False) -> Dict[str, float]:
    model.eval()
    if not use_cls_head:
        text_features = text_features.to(device)
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            if use_cls_head:
                loss, logits = cls_classification_loss(model, pixel_values, labels)
            else:
                loss, logits = clip_classification_loss(model, pixel_values, labels, text_features)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            total_loss += loss.item() * labels.numel()

    return {
        "accuracy": correct / max(total, 1),
        "loss": total_loss / max(total, 1),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_loraplus_param_groups(model, base_lr: float, ratio: float, weight_decay: float):
    lora_a_params = []
    lora_b_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            lora_a_params.append(param)
        elif "lora_B" in name:
            lora_b_params.append(param)
        else:
            other_params.append(param)

    groups = []
    if lora_a_params:
        groups.append({"params": lora_a_params, "lr": base_lr, "weight_decay": weight_decay})
    if lora_b_params:
        groups.append({"params": lora_b_params, "lr": base_lr * ratio, "weight_decay": weight_decay})
    if other_params:
        groups.append({"params": other_params, "lr": base_lr, "weight_decay": weight_decay})
    return groups


def train(model, train_loader, eval_loader, text_features, args, device: torch.device,
          use_cls_head: bool = False):
    param_groups = build_loraplus_param_groups(
        model, args.learning_rate, args.loraplus_lr_ratio, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps_per_epoch = max(
        1,
        (len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps,
    )
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
    if not use_cls_head:
        text_features = text_features.to(device)

    best_accuracy = -1.0
    metrics = {}
    for epoch in range(args.num_train_epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}")
        running_loss = 0.0
        seen = 0
        correct = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                if use_cls_head:
                    loss, logits = cls_classification_loss(model, pixel_values, labels)
                else:
                    loss, logits = clip_classification_loss(model, pixel_values, labels, text_features)

            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            should_step = ((step + 1) % args.gradient_accumulation_steps == 0
                           or (step + 1) == len(train_loader))
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * labels.numel()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            seen += labels.numel()
            progress.set_postfix(
                loss=running_loss / max(seen, 1),
                acc=correct / max(seen, 1),
                lr=scheduler.get_last_lr()[0],
            )

        eval_metrics = evaluate(model, eval_loader, text_features, device, use_cls_head)
        eval_metrics["epoch"] = epoch + 1
        eval_metrics["train_loss"] = running_loss / max(seen, 1)
        eval_metrics["train_acc"] = correct / max(seen, 1)
        print(json.dumps(eval_metrics, indent=2))

        if eval_metrics["accuracy"] > best_accuracy:
            best_accuracy = eval_metrics["accuracy"]
            metrics = eval_metrics
            model.save_pretrained(os.path.join(args.output_dir, "best_adapter"))

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    log_file, original_stdout, original_stderr = setup_file_logging(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    use_cls_head = args.use_cls_head

    # --- Data: SUN397 official partition format only ---
    class_names = load_sun397_partition_class_names(args.sun397_root)
    train_samples = load_sun397_partition_samples(args.sun397_root, args.sun397_train_list)
    eval_samples = load_sun397_partition_samples(args.sun397_root, args.sun397_test_list)

    prompt_template = args.prompt_template
    processor = CLIPProcessor.from_pretrained(args.model_name_or_path)
    model = CLIPModel.from_pretrained(args.model_name_or_path, torch_dtype=dtype).to(device)

    target_module_names = find_target_module_names(model, args.target_modules, args.target_prefix)

    train_data = Sun397PartitionDataset(train_samples, processor)
    eval_data = Sun397PartitionDataset(eval_samples, processor)
    train_data = maybe_subset(train_data, args.max_train_samples, args.seed)
    eval_data = maybe_subset(eval_data, args.max_eval_samples, args.seed)

    print(f"SUN397: {len(class_names)} classes, {len(train_data)} train, {len(eval_data)} test", flush=True)

    train_loader = DataLoader(train_data, batch_size=args.per_device_train_batch_size,
                              shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_data, batch_size=args.per_device_eval_batch_size,
                             shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # --- Build text features (shared across training) ---
    text_features = build_text_features(model, processor, class_names, prompt_template, device)

    # --- GS-LoRA config ---
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
        gora_stable_gamma=args.gora_stable_gamma,
        gora_scale_by_lr=args.gora_scale_by_lr,
        gora_lr=args.gora_lr,
        compensate_scaling=not args.no_compensate_scaling,
        scaling_mode=args.scaling_mode,
        rank_budget_mode=args.rank_budget_mode,
        param_budget=args.param_budget,
    )

    def loss_fn(current_model, batch):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        loss, _ = clip_classification_loss(current_model, pixel_values, labels, text_features.to(device))
        return loss

    model, gs_report = prepare_gslora_model(
        model=model,
        dataloader=train_loader,
        loss_fn=loss_fn,
        config=config,
        device=device,
    )

    # --- Classifier head (GoRA paper protocol) ---
    if use_cls_head:
        proj_dim = model.config.projection_dim
        model.classifier = nn.Linear(proj_dim, len(class_names), bias=False).to(device)
        with torch.no_grad():
            text_feats = build_text_features(model, processor, class_names, prompt_template, device)
            model.classifier.weight.copy_(text_feats.to(device))

    if args.train_logit_scale:
        for name, param in model.named_parameters():
            if name.endswith("logit_scale"):
                param.requires_grad = True

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    # --- Save GS-LoRA report ---
    rank_pattern = gs_report["rank_pattern"]
    rank_stats = gs_report["rank_stats"]

    with open(os.path.join(args.output_dir, "rank_pattern.json"), "w", encoding="utf-8") as f:
        json.dump(rank_pattern, f, indent=2)
    with open(os.path.join(args.output_dir, "rank_stats.json"), "w", encoding="utf-8") as f:
        json.dump(rank_stats, f, indent=2)
    with open(os.path.join(args.output_dir, "gs_lora_report.json"), "w", encoding="utf-8") as f:
        json.dump(gs_report, f, indent=2)
    with open(os.path.join(args.output_dir, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    # --- Save run config ---
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_name_or_path": args.model_name_or_path,
            "sun397_root": args.sun397_root,
            "sun397_train_list": args.sun397_train_list,
            "sun397_test_list": args.sun397_test_list,
            "prompt_template": prompt_template,
            "use_cls_head": use_cls_head,
            "target_module_names": target_module_names,
            "target_modules": args.target_modules,
            "target_prefix": args.target_prefix,
            "base_rank": args.base_rank,
            "tau": args.tau,
            "r_min": args.r_min,
            "r_max": args.r_max,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "learning_rate": args.learning_rate,
            "loraplus_lr_ratio": args.loraplus_lr_ratio,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "calibration_steps": args.calibration_steps,
            "seed": args.seed,
            "init_method": args.init_method,
            "init_scale": args.init_scale,
            "gora_stable_gamma": args.gora_stable_gamma,
            "gora_scale_by_lr": args.gora_scale_by_lr,
            "gora_lr": args.gora_lr,
            "compensate_scaling": not args.no_compensate_scaling,
            "scaling_mode": args.scaling_mode,
            "rank_budget_mode": args.rank_budget_mode,
            "param_budget": args.param_budget,
            "skip_adaptive_rank": args.skip_adaptive_rank,
            "fp16": args.fp16,
            "bf16": args.bf16,
        }, f, indent=2)

    # --- Train ---
    metrics = train(model, train_loader, eval_loader, text_features, args, device,
                    use_cls_head=use_cls_head)
    final_metrics = evaluate(model, eval_loader, text_features, device, use_cls_head=use_cls_head)
    final_metrics = {f"final_{key}": value for key, value in final_metrics.items()}
    metrics.update(final_metrics)

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    processor.save_pretrained(args.output_dir)
    model.save_pretrained(os.path.join(args.output_dir, "last_adapter"))

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


if __name__ == "__main__":
    main()
