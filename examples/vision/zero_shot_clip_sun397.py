"""
Zero-shot CLIP evaluation on SUN397.

Uses HF CLIP (openai/clip-vit-base-patch16) with prompt template "a photo of a {}."
No fine-tuning — just compute image features, compare against text features, measure accuracy.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot CLIP evaluation on SUN397")
    parser.add_argument("--model_name_or_path", type=str,
                        default="/workspace/Models/clip-vit-base-patch16-safetensors")
    parser.add_argument("--zhou_root", type=str,
                        default="/workspace/KeepLoRA/MTIL/data/SUN397")
    parser.add_argument("--zhou_split_file", type=str,
                        default="/workspace/KeepLoRA/MTIL/data/SUN397/split_zhou_SUN397.json")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--prompt_template", type=str, default="a photo of a {}.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


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


def get_class_names(split_file: str) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = {}
    for samples in data.values():
        for _, label, class_name in samples:
            names[int(label)] = str(class_name)
    return [names[idx] for idx in sorted(names)]


@torch.no_grad()
def build_text_features(model, processor, class_names, prompt_template, device, batch_size=128):
    model.eval()
    prompts = []
    for name in class_names:
        parts = name.split("/")
        clean = "/".join(parts[1:]) if len(parts) > 1 else name
        clean = clean.replace("_", " ").replace("-", " ").replace("/", " ")
        prompts.append(prompt_template.format(clean))

    features = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start: start + batch_size]
        inputs = processor(text=batch, padding=True, return_tensors="pt").to(device)
        out = model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else None
        if feats is None and hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            feats = out.last_hidden_state[:, 0, :]
        if not torch.is_tensor(feats):
            feats = out
        feats = F.normalize(feats.float(), dim=-1)
        features.append(feats.cpu())
    return torch.cat(features, dim=0)


def collate_fn(features):
    pixel_values = []
    labels = []
    for f in features:
        pixel_values.append(f["image"])
        labels.append(f["label"])
    return {"images": pixel_values, "labels": torch.tensor(labels, dtype=torch.long)}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model & processor ---
    print(f"Loading model from {args.model_name_or_path} ...")
    model = CLIPModel.from_pretrained(args.model_name_or_path).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name_or_path)
    model.eval()

    # --- Load dataset ---
    print(f"Loading {args.split} split ...")
    dataset = ZhouSplitDataset(args.zhou_root, args.zhou_split_file, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)
    print(f"  Samples: {len(dataset)}")

    # --- Build text features ---
    class_names = get_class_names(args.zhou_split_file)
    print(f"  Classes: {len(class_names)}")
    text_features = build_text_features(model, processor, class_names,
                                        args.prompt_template, device)
    text_features = text_features.to(device)
    print(f"  Text features: {text_features.shape}")

    # --- Evaluate ---
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Zero-shot evaluating"):
        images = batch["images"]
        labels = batch["labels"].to(device)

        # Process images with processor (batch)
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)

        img_out = model.get_image_features(pixel_values=pixel_values)
        image_features = img_out.pooler_output if hasattr(img_out, "pooler_output") and img_out.pooler_output is not None else None
        if image_features is None and hasattr(img_out, "last_hidden_state") and img_out.last_hidden_state is not None:
            image_features = img_out.last_hidden_state[:, 0, :]
        if not torch.is_tensor(image_features):
            image_features = img_out
        image_features = F.normalize(image_features.float(), dim=-1)

        logits = image_features @ text_features.t()
        logits = logits * model.logit_scale.exp()

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    accuracy = correct / total
    print(f"\nZero-shot accuracy on {args.split}: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # --- Per-class accuracy ---
    per_class_correct = torch.zeros(len(class_names), dtype=torch.long)
    per_class_total = torch.zeros(len(class_names), dtype=torch.long)
    for batch in tqdm(loader, desc="Per-class"):
        images = batch["images"]
        labels = batch["labels"]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        img_out = model.get_image_features(pixel_values=pixel_values)
        image_features = img_out.pooler_output if hasattr(img_out, "pooler_output") and img_out.pooler_output is not None else None
        if image_features is None and hasattr(img_out, "last_hidden_state") and img_out.last_hidden_state is not None:
            image_features = img_out.last_hidden_state[:, 0, :]
        if not torch.is_tensor(image_features):
            image_features = img_out
        image_features = F.normalize(image_features.float(), dim=-1)
        logits = image_features @ text_features.t()
        logits = logits * model.logit_scale.exp()
        preds = logits.argmax(dim=-1).cpu()
        for i, lbl in enumerate(labels):
            per_class_total[lbl] += 1
            if preds[i] == lbl:
                per_class_correct[lbl] += 1

    per_class_acc = {}
    for i, name in enumerate(class_names):
        if per_class_total[i] > 0:
            per_class_acc[name] = (per_class_correct[i].item() / per_class_total[i].item())

    # --- Save ---
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        result = {
            "model": args.model_name_or_path,
            "split": args.split,
            "dataset": "SUN397",
            "num_classes": len(class_names),
            "num_samples": total,
            "accuracy": accuracy,
            "prompt_template": args.prompt_template,
        }
        with open(os.path.join(args.output_dir, "zero_shot_result.json"), "w") as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(args.output_dir, "per_class_accuracy.json"), "w") as f:
            json.dump(per_class_acc, f, indent=2)
        print(f"Results saved to {args.output_dir}")

    return accuracy


if __name__ == "__main__":
    main()
