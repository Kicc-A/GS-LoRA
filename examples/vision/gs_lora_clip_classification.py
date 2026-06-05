import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, get_scheduler

from gs_lora import GSLoraConfig, find_target_module_names, prepare_gslora_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune CLIP for image classification with gradient-SVD adaptive-rank LoRA."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--zhou_root", type=str, default=None)
    parser.add_argument("--zhou_split_file", type=str, default=None)
    parser.add_argument("--gtsrb_root", type=str, default=None)
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--validation_split", type=str, default="validation")
    parser.add_argument("--test_split", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj", "out_proj"])
    parser.add_argument("--target_prefix", type=str, default="vision_model")
    parser.add_argument("--base_rank", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--r_min", type=int, default=2)
    parser.add_argument("--r_max", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--init_method",
        choices=["none", "svd_sqrt", "svd_sigma", "svd_a_zero_b", "gora_pinv"],
        default="none",
    )
    parser.add_argument("--init_scale", type=float, default=1e-3)
    parser.add_argument("--gora_stable_gamma", type=float, default=5e-2)
    parser.add_argument("--gora_scale_by_lr", action="store_true")
    parser.add_argument("--gora_lr", type=float, default=5e-2)
    parser.add_argument("--no_compensate_scaling", action="store_true")
    parser.add_argument("--scaling_mode", choices=["rank", "sqrt_rank", "avg_rank"], default="rank")
    parser.add_argument("--rank_budget_mode", choices=["independent", "param"], default="independent")
    parser.add_argument("--param_budget", type=int, default=None)
    parser.add_argument("--calibration_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--train_logit_scale", action="store_true")
    parser.add_argument("--skip_adaptive_rank", action="store_true")
    parser.add_argument("--svhn_single_digit", action="store_true")
    return parser.parse_args()


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


class SvhnSingleDigitDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = [idx for idx, label in enumerate(dataset["label"]) if len(label["digit"]) == 1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.dataset[int(self.indices[int(idx)])]
        digit = int(item["label"]["digit"][0])
        return {
            "image": item["image"].convert("RGB"),
            "label": 0 if digit == 10 else digit,
        }


class ZhouSplitDataset(Dataset):
    def __init__(self, root: str, split_file: str, split: str):
        self.root = Path(root)
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = data[split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label, _ = self.samples[int(idx)]
        image = Image.open(self.root / rel_path).convert("RGB")
        return {"image": image, "label": int(label)}


def get_zhou_class_names(split_file: str) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = {}
    for samples in data.values():
        for _, label, class_name in samples:
            names[int(label)] = str(class_name)
    return [names[idx] for idx in sorted(names)]


class GTSRBDataset(Dataset):
    def __init__(self, root: str, split: str):
        self.root = Path(root)
        self.samples = []
        if split == "train":
            with open(self.root / "Train.csv", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.samples.append((row["Path"], int(row["ClassId"])))
        elif split == "test":
            with open(self.root / "Test.csv", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.samples.append((row["Path"], int(row["ClassId"])))
        else:
            raise ValueError(f"GTSRB supports train/test splits, got {split!r}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[int(idx)]
        image = Image.open(self.root / rel_path).convert("RGB")
        return {"image": image, "label": label}


GTSRB_CLASS_NAMES = [
    "speed limit 20 km/h",
    "speed limit 30 km/h",
    "speed limit 50 km/h",
    "speed limit 60 km/h",
    "speed limit 70 km/h",
    "speed limit 80 km/h",
    "end of speed limit 80 km/h",
    "speed limit 100 km/h",
    "speed limit 120 km/h",
    "no passing",
    "no passing for vehicles over 3.5 metric tons",
    "right-of-way at the next intersection",
    "priority road",
    "yield",
    "stop",
    "no vehicles",
    "vehicles over 3.5 metric tons prohibited",
    "no entry",
    "general caution",
    "dangerous curve to the left",
    "dangerous curve to the right",
    "double curve",
    "bumpy road",
    "slippery road",
    "road narrows on the right",
    "road work",
    "traffic signals",
    "pedestrians",
    "children crossing",
    "bicycles crossing",
    "beware of ice or snow",
    "wild animals crossing",
    "end of all speed and passing limits",
    "turn right ahead",
    "turn left ahead",
    "ahead only",
    "go straight or right",
    "go straight or left",
    "keep right",
    "keep left",
    "roundabout mandatory",
    "end of no passing",
    "end of no passing by vehicles over 3.5 metric tons",
]


class VisionTransformDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[int(idx)]
        pixel_values = self.processor(images=[item["image"]], return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pixel_values, "labels": item["label"]}


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


def load_classification_dataset(args):
    if args.zhou_root or args.zhou_split_file or args.gtsrb_root:
        return None

    if args.dataset_dir:
        dataset = load_dataset("imagefolder", data_dir=args.dataset_dir)
    elif args.dataset_name:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        raise ValueError("Pass either --dataset_name or --dataset_dir.")

    if args.validation_split not in dataset:
        if "test" in dataset:
            args.validation_split = "test"
        else:
            split = dataset[args.train_split].train_test_split(test_size=0.1, seed=args.seed)
            dataset = {"train": split["train"], "validation": split["test"]}
            args.train_split = "train"
            args.validation_split = "validation"

    return dataset


def get_class_names(dataset, split: str, label_column: str) -> List[str]:
    feature = dataset[split].features[label_column]
    if hasattr(feature, "names") and feature.names is not None:
        return list(feature.names)

    labels = dataset[split][label_column]
    unique_labels = sorted(set(labels))
    return [str(label) for label in unique_labels]


def infer_prompt_template(dataset_name: Optional[str], prompt_template: Optional[str]) -> str:
    if prompt_template:
        return prompt_template
    name = (dataset_name or "").lower()
    if "dtd" in name:
        return "a photo of a {} texture."
    if "food" in name:
        return "a photo of {}."
    if "cars" in name or "stanfordcars" in name:
        return "a photo of the {}, a type of car."
    if "eurosat" in name:
        return "a centered satellite photo of {}."
    if "gtsrb" in name:
        return "a photo of a traffic sign: {}."
    return "a photo of a {}."


def preprocess_dataset(dataset, processor, image_column: str, label_column: str):
    def transform(examples):
        raw_images = examples[image_column]
        raw_labels = examples[label_column]
        is_single = not isinstance(raw_images, list)
        if is_single:
            raw_images = [raw_images]
            raw_labels = [raw_labels]

        images = [image.convert("RGB") for image in raw_images]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        if is_single:
            pixel_values = pixel_values[0]
            raw_labels = raw_labels[0]
        return {
            "pixel_values": pixel_values,
            "labels": raw_labels,
        }

    dataset.set_transform(transform)
    return dataset


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
def build_text_features(
    model,
    processor,
    class_names: List[str],
    prompt_template: str,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    model.eval()
    prompts = [prompt_template.format(name.replace("_", " ")) for name in class_names]
    features = []
    clip_model = get_clip_backbone(model)

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        inputs = processor(text=batch_prompts, padding=True, return_tensors="pt").to(device)
        text_features = to_float_tensor(clip_model.get_text_features(**inputs))
        text_features = F.normalize(text_features, dim=-1)
        features.append(text_features.cpu())

    return torch.cat(features, dim=0)


def clip_classification_loss(model, pixel_values, labels, text_features):
    clip_model = get_clip_backbone(model)
    image_features = to_float_tensor(clip_model.get_image_features(pixel_values=pixel_values))
    image_features = F.normalize(image_features, dim=-1)
    logits = image_features @ text_features.t()
    logits = logits * clip_model.logit_scale.exp()
    loss = F.cross_entropy(logits, labels)
    return loss, logits


def evaluate(model, dataloader, text_features, device: torch.device) -> Dict[str, float]:
    model.eval()
    text_features = text_features.to(device)
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            loss, logits = clip_classification_loss(model, pixel_values, labels, text_features)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            total_loss += loss.item() * labels.numel()

    return {
        "accuracy": correct / max(total, 1),
        "loss": total_loss / max(total, 1),
    }


def train(model, train_loader, eval_loader, text_features, args, device: torch.device):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
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
                loss, logits = clip_classification_loss(model, pixel_values, labels, text_features)

            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            should_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader)
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

        eval_metrics = evaluate(model, eval_loader, text_features, device)
        eval_metrics["epoch"] = epoch + 1
        eval_metrics["train_loss"] = running_loss / max(seen, 1)
        eval_metrics["train_acc"] = correct / max(seen, 1)
        print(json.dumps(eval_metrics, indent=2))

        if eval_metrics["accuracy"] > best_accuracy:
            best_accuracy = eval_metrics["accuracy"]
            metrics = eval_metrics
            model.save_pretrained(os.path.join(args.output_dir, "best_adapter"))

    return metrics


def maybe_select(dataset, split: str, max_samples: Optional[int], seed: int):
    data = dataset[split]
    if max_samples is not None and max_samples < len(data):
        data = data.shuffle(seed=seed).select(range(max_samples))
    return data


def maybe_subset(dataset, max_samples: Optional[int], seed: int):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    log_file, original_stdout, original_stderr = setup_file_logging(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32

    dataset = load_classification_dataset(args)
    if args.zhou_root or args.zhou_split_file:
        if not args.zhou_root or not args.zhou_split_file:
            raise ValueError("Pass both --zhou_root and --zhou_split_file.")
        class_names = get_zhou_class_names(args.zhou_split_file)
    elif args.gtsrb_root:
        class_names = GTSRB_CLASS_NAMES
    elif args.svhn_single_digit:
        class_names = [str(i) for i in range(10)]
    else:
        class_names = get_class_names(dataset, args.train_split, args.label_column)

    dataset_hint = args.dataset_name or args.dataset_dir or args.zhou_root or args.gtsrb_root
    prompt_template = infer_prompt_template(dataset_hint, args.prompt_template)

    processor = CLIPProcessor.from_pretrained(args.model_name_or_path)
    model = CLIPModel.from_pretrained(args.model_name_or_path, torch_dtype=dtype).to(device)
    target_module_names = find_target_module_names(model, args.target_modules, args.target_prefix)

    eval_split = args.test_split or args.validation_split
    if args.zhou_root:
        train_data = ZhouSplitDataset(args.zhou_root, args.zhou_split_file, args.train_split)
        eval_data = ZhouSplitDataset(args.zhou_root, args.zhou_split_file, eval_split)
        train_data = VisionTransformDataset(maybe_subset(train_data, args.max_train_samples, args.seed), processor)
        eval_data = VisionTransformDataset(maybe_subset(eval_data, args.max_eval_samples, args.seed), processor)
    elif args.gtsrb_root:
        train_data = GTSRBDataset(args.gtsrb_root, args.train_split)
        eval_data = GTSRBDataset(args.gtsrb_root, eval_split)
        train_data = VisionTransformDataset(maybe_subset(train_data, args.max_train_samples, args.seed), processor)
        eval_data = VisionTransformDataset(maybe_subset(eval_data, args.max_eval_samples, args.seed), processor)
    elif args.svhn_single_digit:
        train_data = SvhnSingleDigitDataset(dataset[args.train_split])
        eval_data = SvhnSingleDigitDataset(dataset[eval_split])
        train_data = VisionTransformDataset(maybe_subset(train_data, args.max_train_samples, args.seed), processor)
        eval_data = VisionTransformDataset(maybe_subset(eval_data, args.max_eval_samples, args.seed), processor)
    else:
        train_data = maybe_select(dataset, args.train_split, args.max_train_samples, args.seed)
        eval_data = maybe_select(dataset, eval_split, args.max_eval_samples, args.seed)
        train_data = preprocess_dataset(train_data, processor, args.image_column, args.label_column)
        eval_data = preprocess_dataset(eval_data, processor, args.image_column, args.label_column)

    train_loader = DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_data,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    text_features = build_text_features(model, processor, class_names, prompt_template, device)

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
        loss, _ = clip_classification_loss(
            current_model,
            pixel_values,
            labels,
            text_features.to(device),
        )
        return loss

    model, gs_report = prepare_gslora_model(
        model=model,
        dataloader=train_loader,
        loss_fn=loss_fn,
        config=config,
        device=device,
    )

    if args.train_logit_scale:
        for name, param in model.named_parameters():
            if name.endswith("logit_scale"):
                param.requires_grad = True

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

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
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name_or_path": args.model_name_or_path,
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "dataset_dir": args.dataset_dir,
                "zhou_root": args.zhou_root,
                "zhou_split_file": args.zhou_split_file,
                "gtsrb_root": args.gtsrb_root,
                "train_split": args.train_split,
                "validation_split": args.validation_split,
                "test_split": args.test_split,
                "prompt_template": prompt_template,
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
                "svhn_single_digit": args.svhn_single_digit,
            },
            f,
            indent=2,
        )

    metrics = train(model, train_loader, eval_loader, text_features, args, device)
    final_metrics = evaluate(model, eval_loader, text_features, device)
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
