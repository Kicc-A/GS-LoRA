"""
Zero-shot CLIP evaluation on StanfordCars using OpenAI CLIP.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot CLIP evaluation on StanfordCars")
    parser.add_argument("--model_name", type=str, default="ViT-B/16")
    parser.add_argument("--cars_root", type=str,
                        default="/workspace/KeepLoRA/MTIL/data/StanfordCars")
    parser.add_argument("--cars_split_file", type=str,
                        default="/workspace/KeepLoRA/MTIL/data/StanfordCars/split_zhou_StanfordCars.json")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--prompt_template", type=str, default="a photo of a {}.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


class ZhouSplitDataset(Dataset):
    """Dataset for StanfordCars Zhou split format: [rel_path, label, class_name]."""

    def __init__(self, root: str, split_file: str, split: str, preprocess):
        self.root = Path(root)
        self.preprocess = preprocess
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = data[split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label, _ = self.samples[int(idx)]
        image = self.preprocess(Image.open(self.root / rel_path).convert("RGB"))
        return image, int(label)


def get_class_names(split_file: str) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = {}
    for samples in data.values():
        for _, label, class_name in samples:
            names[int(label)] = str(class_name)
    return [names[idx] for idx in sorted(names)]


@torch.no_grad()
def build_text_features(model, class_names: List[str], prompt_template: str, device: torch.device):
    """Build text features using CLIP's text encoder."""
    prompts = [prompt_template.format(name) for name in class_names]
    text_tokens = clip.tokenize(prompts).to(device)
    text_features = model.encode_text(text_tokens)
    return F.normalize(text_features.float(), dim=-1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model & preprocess ---
    print(f"Loading CLIP model: {args.model_name} ...")
    model, preprocess = clip.load(args.model_name, device=device)
    model.eval()

    # --- Load dataset ---
    print(f"Loading {args.split} split ...")
    dataset = ZhouSplitDataset(args.cars_root, args.cars_split_file, args.split, preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    print(f"  Samples: {len(dataset)}")

    # --- Build text features ---
    class_names = get_class_names(args.cars_split_file)
    print(f"  Classes: {len(class_names)}")
    print(f"  Prompt: \"{args.prompt_template.format(class_names[0])}\"")
    text_features = build_text_features(model, class_names, args.prompt_template, device)
    print(f"  Text features: {text_features.shape}")

    # --- Evaluate ---
    correct = 0
    total = 0
    per_class_correct = torch.zeros(len(class_names), dtype=torch.long)
    per_class_total = torch.zeros(len(class_names), dtype=torch.long)

    for images, labels in tqdm(loader, desc="Zero-shot evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images)
        image_features = F.normalize(image_features.float(), dim=-1)

        logits = image_features @ text_features.t()
        logits = logits * model.logit_scale.exp()

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

        for i, lbl in enumerate(labels.cpu()):
            per_class_total[lbl] += 1
            if preds[i].cpu() == lbl:
                per_class_correct[lbl] += 1

    accuracy = correct / total
    print(f"\nZero-shot accuracy on {args.split}: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # --- Per-class accuracy ---
    per_class_acc = {}
    for i, name in enumerate(class_names):
        if per_class_total[i] > 0:
            per_class_acc[name] = per_class_correct[i].item() / per_class_total[i].item()

    # --- Save ---
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        result = {
            "model": args.model_name,
            "split": args.split,
            "dataset": "StanfordCars",
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
