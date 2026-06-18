"""
Zero-shot CLIP evaluation on SUN397 using OpenAI native CLIP.
CLIP-ViT-B/16, prompt "a photo of a {}."
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, "/workspace/CLIP_official")
import clip


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot CLIP evaluation on SUN397 (OpenAI native CLIP)")
    parser.add_argument("--clip_cache_root", type=str, default="/root/.cache/clip")
    parser.add_argument("--zhou_root", type=str, default="/workspace/KeepLoRA/MTIL/data/SUN397")
    parser.add_argument("--zhou_split_file", type=str,
                        default="/workspace/KeepLoRA/MTIL/data/SUN397/split_zhou_SUN397.json")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--prompt_template", type=str, default="a photo of a {}.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


class ZhouSplitDataset(Dataset):
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
        image = Image.open(self.root / rel_path).convert("RGB")
        return self.preprocess(image), int(label)


def get_class_names(split_file: str):
    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = {}
    for samples in data.values():
        for _, label, class_name in samples:
            names[int(label)] = str(class_name)
    return [names[idx] for idx in sorted(names)]


def format_class_name(name):
    parts = name.split("/")
    if len(parts) > 1:
        name = "/".join(parts[1:])
    return name.replace("_", " ").replace("-", " ").replace("/", " ")


@torch.no_grad()
def build_text_features(model, class_names, prompt_template, device):
    prompts = [prompt_template.format(format_class_name(name)) for name in class_names]
    tokens = clip.tokenize(prompts).to(device)
    text_features = model.encode_text(tokens).float()
    text_features = F.normalize(text_features, dim=-1)
    return text_features


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading OpenAI CLIP ViT-B/16 (cache: {args.clip_cache_root}) ...")
    model, preprocess = clip.load("ViT-B/16", device=device, jit=False, download_root=args.clip_cache_root)
    model.eval()
    model.float()

    print(f"Loading {args.split} split ...")
    dataset = ZhouSplitDataset(args.zhou_root, args.zhou_split_file, args.split, preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"  Samples: {len(dataset)}")

    class_names = get_class_names(args.zhou_split_file)
    print(f"  Classes: {len(class_names)}")
    print(f"  Prompt: \"{args.prompt_template.format(format_class_name(class_names[0]))}\"")
    text_features = build_text_features(model, class_names, args.prompt_template, device)
    print(f"  Text features: {text_features.shape}")

    correct = 0
    total = 0
    per_class_correct = torch.zeros(len(class_names), dtype=torch.long)
    per_class_total = torch.zeros(len(class_names), dtype=torch.long)

    for images, labels in tqdm(loader, desc="Zero-shot evaluating"):
        images = images.to(device)
        labels = torch.as_tensor(labels, device=device)

        image_features = model.encode_image(images).float()
        image_features = F.normalize(image_features, dim=-1)

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

    per_class_acc = {}
    for i, name in enumerate(class_names):
        if per_class_total[i] > 0:
            per_class_acc[name] = per_class_correct[i].item() / per_class_total[i].item()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "zero_shot_result.json"), "w") as f:
            json.dump({
                "model": f"OpenAI CLIP ViT-B/16",
                "split": args.split,
                "dataset": "SUN397",
                "num_classes": len(class_names),
                "num_samples": total,
                "accuracy": accuracy,
                "prompt_template": args.prompt_template,
            }, f, indent=2)
        with open(os.path.join(args.output_dir, "per_class_accuracy.json"), "w") as f:
            json.dump(per_class_acc, f, indent=2)
        print(f"Results saved to {args.output_dir}")

    return accuracy


if __name__ == "__main__":
    main()
