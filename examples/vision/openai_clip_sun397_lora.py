#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, "/workspace/CLIP_official")
sys.path.insert(0, "/workspace/GS-LoRA")

import clip
from gs_lora.rank_allocator import allocate_ranks


class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_file_logging(output_dir):
    log_path = output_dir / "train.log"
    log_file = log_path.open("a", buffering=1)
    sys.stdout = TeeLogger(sys.__stdout__, log_file)
    sys.stderr = TeeLogger(sys.__stderr__, log_file)
    print(f"Logging to {log_path}", flush=True)
    return log_file


class RankConfig:
    def __init__(self, tau, r_min, r_max, base_rank):
        self.tau = tau
        self.r_min = r_min
        self.r_max = r_max
        self.base_rank = base_rank


def allocate_param_budget(singular_values, rank_costs, r_min, r_max, param_budget):
    ranks = {}
    for name, values in singular_values.items():
        ranks[name] = min(r_min, int(values.numel()), r_max)

    def total_cost():
        return sum(ranks[name] * rank_costs[name] for name in ranks)

    max_budget = sum(min(int(values.numel()), r_max) * rank_costs[name] for name, values in singular_values.items())
    budget = max(total_cost(), min(param_budget, max_budget))
    while True:
        current = total_cost()
        best_name = None
        best_score = None
        for name, values in singular_values.items():
            next_rank = ranks[name]
            cost = rank_costs[name]
            if next_rank >= min(int(values.numel()), r_max) or current + cost > budget:
                continue
            score = values[next_rank].pow(2).item() / cost
            if best_score is None or score > best_score:
                best_name = name
                best_score = score
        if best_name is None:
            break
        ranks[best_name] += 1
    return ranks


def svd_lora_factors(grad, weight, rank, init_scale, effective_scaling, init_method="svd_sqrt", eps=1e-12):
    grad_matrix = grad.detach().float().reshape(grad.shape[0], -1)
    u, singular_values, vh = torch.linalg.svd(grad_matrix, full_matrices=False)
    rank = min(int(rank), singular_values.numel())
    if rank <= 0:
        return None
    u = u[:, :rank]
    singular_values = singular_values[:rank]
    vh = vh[:rank, :]
    if init_method == "svd_a_zero_b":
        return {
            "lora_A": (vh * init_scale).cpu(),
            "lora_B": torch.zeros(grad_matrix.shape[0], rank, dtype=vh.dtype).cpu(),
        }
    kept_grad_norm = singular_values.norm().item()
    if kept_grad_norm <= eps or init_scale == 0:
        return {
            "lora_A": torch.zeros(rank, grad_matrix.shape[1]),
            "lora_B": torch.zeros(grad_matrix.shape[0], rank),
        }
    root = torch.sqrt(singular_values.clamp_min(0.0))
    lora_b = -u * root.unsqueeze(0)
    lora_a = root.unsqueeze(1) * vh
    weight_norm = weight.detach().float().reshape(weight.shape[0], -1).norm().item()
    factor_scale = (init_scale * weight_norm) / kept_grad_norm
    factor_scale = factor_scale / effective_scaling if effective_scaling > 0 else factor_scale
    factor_scale = factor_scale ** 0.5
    return {
        "lora_A": (lora_a * factor_scale).cpu(),
        "lora_B": (lora_b * factor_scale).cpu(),
    }


def svd_target_delta(grad, weight, rank, init_scale, effective_scaling):
    factors = svd_lora_factors(grad, weight, rank, init_scale, effective_scaling, "svd_sqrt")
    if factors is None:
        return None
    return factors["lora_B"] @ factors["lora_A"]


def compensated_svd_factors(grad, weight, rank, init_scale, effective_scaling):
    target_delta = svd_target_delta(grad, weight, rank, init_scale, effective_scaling)
    if target_delta is None:
        return None
    lora_b = torch.empty(target_delta.shape[0], rank, dtype=target_delta.dtype)
    nn.init.kaiming_uniform_(lora_b, a=np.sqrt(5))
    lora_a = torch.linalg.pinv(lora_b.float()) @ target_delta.float()
    return {
        "lora_A": lora_a.to(target_delta.dtype).cpu(),
        "lora_B": lora_b.cpu(),
        "target_delta": target_delta.cpu(),
    }


def compensated_svd_direction_factors(grad, weight, rank, init_scale, effective_scaling):
    factors = svd_lora_factors(grad, weight, rank, init_scale, effective_scaling, "svd_sqrt")
    if factors is None:
        return None
    factors["target_delta"] = factors["lora_B"] @ factors["lora_A"]
    return factors


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_class_names(root):
    with (root / "ClassName.txt").open() as f:
        return [line.strip().lstrip("/") for line in f if line.strip()]


def build_partition_samples(root, partition_dir, list_name):
    class_names = load_class_names(root)
    class_to_label = {name: idx for idx, name in enumerate(class_names)}
    samples = []
    with (partition_dir / list_name).open() as f:
        for line in f:
            rel = line.strip().lstrip("/")
            if not rel:
                continue
            class_name = str(Path(rel).parent)
            if class_name not in class_to_label:
                raise ValueError(f"Unknown SUN397 class in {list_name}: {class_name}")
            path = root / rel
            if not path.is_file():
                raise FileNotFoundError(path)
            samples.append((str(path), class_to_label[class_name]))
    return class_names, samples


class Sun397Dataset(Dataset):
    def __init__(self, samples, preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.preprocess(image), label


class ImagePathDataset(Dataset):
    def __init__(self, samples, preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[int(idx)]
        image = Image.open(path).convert("RGB")
        return self.preprocess(image), label


class SvhnSingleDigitDataset(Dataset):
    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess
        self.indices = []
        for idx, label in enumerate(dataset["label"]):
            digits = label["digit"]
            if len(digits) == 1:
                self.indices.append(idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.dataset[int(self.indices[int(idx)])]
        digit = int(item["label"]["digit"][0])
        label = 0 if digit == 10 else digit
        return self.preprocess(item["image"].convert("RGB")), label


def build_dtd_split(root, split_file, train_split, test_split):
    with Path(split_file).open() as f:
        split = json.load(f)
    class_by_label = {}

    def read_split(name):
        samples = []
        for rel_path, label, class_name in split[name]:
            label = int(label)
            class_by_label[label] = class_name
            path = Path(root) / "images" / rel_path
            if not path.is_file():
                raise FileNotFoundError(path)
            samples.append((str(path), label))
        return samples

    train_samples = read_split(train_split)
    test_samples = read_split(test_split)
    class_names = [class_by_label[idx] for idx in sorted(class_by_label)]
    return class_names, train_samples, test_samples


def format_class_name(name):
    return name.replace("_", " ").replace("-", " ")


class LoRALinear(nn.Module):
    def __init__(self, source: nn.Linear, rank, alpha, dropout):
        super().__init__()
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.bias = nn.Parameter(source.bias.detach().clone()) if source.bias is not None else None
        self.rank = int(rank)
        self.scaling = alpha / self.rank if self.rank > 0 else 1.0
        self.lora_dropout = nn.Dropout(dropout)
        if self.rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.rank, source.in_features))
            self.lora_B = nn.Parameter(torch.zeros(source.out_features, self.rank))
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.rank > 0:
            out = out + F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B) * self.scaling
        return out


class LoRAMultiheadAttention(nn.Module):
    def __init__(self, source: nn.MultiheadAttention, rank_pattern, alpha, dropout):
        super().__init__()
        self.embed_dim = source.embed_dim
        self.num_heads = source.num_heads
        self.dropout = source.dropout
        self.batch_first = source.batch_first
        self.in_proj_weight = nn.Parameter(source.in_proj_weight.detach().clone())
        self.in_proj_bias = nn.Parameter(source.in_proj_bias.detach().clone()) if source.in_proj_bias is not None else None
        self.out_rank = int(rank_pattern.get("out", 0))
        self.out_proj = LoRALinear(source.out_proj, self.out_rank, alpha, dropout)
        self.lora_dropout = nn.Dropout(dropout)
        self.q_rank = int(rank_pattern.get("q", 0))
        self.k_rank = int(rank_pattern.get("k", 0))
        self.v_rank = int(rank_pattern.get("v", 0))
        self.q_A, self.q_B = self._make_lora(self.q_rank)
        self.k_A, self.k_B = self._make_lora(self.k_rank)
        self.v_A, self.v_B = self._make_lora(self.v_rank)
        self.q_scaling = alpha / self.q_rank if self.q_rank > 0 else 1.0
        self.k_scaling = alpha / self.k_rank if self.k_rank > 0 else 1.0
        self.v_scaling = alpha / self.v_rank if self.v_rank > 0 else 1.0

    def _make_lora(self, rank):
        if rank <= 0:
            return None, None
        a = nn.Parameter(torch.empty(rank, self.embed_dim))
        b = nn.Parameter(torch.zeros(self.embed_dim, rank))
        nn.init.kaiming_uniform_(a, a=np.sqrt(5))
        return a, b

    def forward(self, query, key, value, need_weights=False, attn_mask=None, **kwargs):
        if query is not key or key is not value:
            raise NotImplementedError("This wrapper supports CLIP self-attention only.")
        q_w, k_w, v_w = self.in_proj_weight.chunk(3, dim=0)
        q_b, k_b, v_b = self.in_proj_bias.chunk(3, dim=0) if self.in_proj_bias is not None else (None, None, None)
        q = F.linear(query, q_w, q_b)
        k = F.linear(query, k_w, k_b)
        v = F.linear(query, v_w, v_b)
        if self.q_rank > 0:
            q = q + F.linear(F.linear(self.lora_dropout(query), self.q_A), self.q_B) * self.q_scaling
        if self.k_rank > 0:
            k = k + F.linear(F.linear(self.lora_dropout(query), self.k_A), self.k_B) * self.k_scaling
        if self.v_rank > 0:
            v = v + F.linear(F.linear(self.lora_dropout(query), self.v_A), self.v_B) * self.v_scaling

        seq_len, batch, embed = q.shape
        head_dim = embed // self.num_heads
        q = q.view(seq_len, batch * self.num_heads, head_dim).transpose(0, 1)
        k = k.view(seq_len, batch * self.num_heads, head_dim).transpose(0, 1)
        v = v.view(seq_len, batch * self.num_heads, head_dim).transpose(0, 1)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        attn = attn.transpose(0, 1).contiguous().view(seq_len, batch, embed)
        return self.out_proj(attn), None


class OpenAIClipClassifier(nn.Module):
    def __init__(self, checkpoint_root):
        super().__init__()
        self.clip_model, self.preprocess = clip.load("ViT-B/16", device="cpu", jit=False, download_root=checkpoint_root)
        self.clip_model.float()
        self.visual = self.clip_model.visual
        self.logit_scale = self.clip_model.logit_scale

    def inject_lora(self, rank_pattern, alpha, dropout):
        for idx, block in enumerate(self.visual.transformer.resblocks):
            ranks = {
                "q": rank_pattern.get(f"visual.transformer.resblocks.{idx}.attn.q", 0),
                "k": rank_pattern.get(f"visual.transformer.resblocks.{idx}.attn.k", 0),
                "v": rank_pattern.get(f"visual.transformer.resblocks.{idx}.attn.v", 0),
                "out": rank_pattern.get(f"visual.transformer.resblocks.{idx}.attn.out_proj", 0),
            }
            block.attn = LoRAMultiheadAttention(block.attn, ranks, alpha, dropout)
            block.mlp.c_fc = LoRALinear(
                block.mlp.c_fc,
                rank_pattern.get(f"visual.transformer.resblocks.{idx}.mlp.c_fc", 0),
                alpha,
                dropout,
            )
            block.mlp.c_proj = LoRALinear(
                block.mlp.c_proj,
                rank_pattern.get(f"visual.transformer.resblocks.{idx}.mlp.c_proj", 0),
                alpha,
                dropout,
            )
        self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.clip_model.parameters():
            p.requires_grad = False
        for module in self.visual.modules():
            if isinstance(module, LoRAMultiheadAttention):
                if module.q_A is not None:
                    module.q_A.requires_grad = True
                    module.q_B.requires_grad = True
                if module.k_A is not None:
                    module.k_A.requires_grad = True
                    module.k_B.requires_grad = True
                if module.v_A is not None:
                    module.v_A.requires_grad = True
                    module.v_B.requires_grad = True
            elif isinstance(module, LoRALinear):
                if module.lora_A is not None:
                    module.lora_A.requires_grad = True
                    module.lora_B.requires_grad = True

    def forward(self, images):
        return self.visual(images)


@torch.no_grad()
def build_text_features(model, class_names, prompt_template, device, batch_size=128):
    prompts = [prompt_template.format(format_class_name(name)) for name in class_names]
    feats = []
    model.clip_model.eval()
    for start in range(0, len(prompts), batch_size):
        tokens = clip.tokenize(prompts[start:start + batch_size]).to(device)
        text_features = model.clip_model.encode_text(tokens).float()
        feats.append(F.normalize(text_features, dim=-1).cpu())
    return torch.cat(feats, dim=0)


def clip_loss(model, images, labels, text_features):
    image_features = F.normalize(model(images).float(), dim=-1)
    logits = image_features @ text_features.t()
    logits = logits * model.logit_scale.exp()
    return F.cross_entropy(logits, labels), logits


def set_base_attention_trainable(model, trainable):
    for module in model.visual.modules():
        if isinstance(module, LoRAMultiheadAttention):
            module.in_proj_weight.requires_grad = trainable
            if module.in_proj_bias is not None:
                module.in_proj_bias.requires_grad = False
            module.out_proj.weight.requires_grad = trainable
            if module.out_proj.bias is not None:
                module.out_proj.bias.requires_grad = False
            if module.q_A is not None:
                module.q_A.requires_grad = False
                module.q_B.requires_grad = False
            if module.k_A is not None:
                module.k_A.requires_grad = False
                module.k_B.requires_grad = False
            if module.v_A is not None:
                module.v_A.requires_grad = False
                module.v_B.requires_grad = False
            if module.out_proj.lora_A is not None:
                module.out_proj.lora_A.requires_grad = False
                module.out_proj.lora_B.requires_grad = False
        elif isinstance(module, LoRALinear):
            module.weight.requires_grad = trainable
            if module.bias is not None:
                module.bias.requires_grad = False
            if module.lora_A is not None:
                module.lora_A.requires_grad = False
                module.lora_B.requires_grad = False


def collect_grad_cache(model):
    grad_cache = {}
    for idx, block in enumerate(model.visual.transformer.resblocks):
        grad = block.attn.in_proj_weight.grad
        if grad is None:
            continue
        q_grad, k_grad, v_grad = grad.detach().float().chunk(3, dim=0)
        q_weight, k_weight, v_weight = block.attn.in_proj_weight.detach().float().chunk(3, dim=0)
        grad_cache[f"visual.transformer.resblocks.{idx}.attn.q"] = {"grad": q_grad, "weight": q_weight}
        grad_cache[f"visual.transformer.resblocks.{idx}.attn.k"] = {"grad": k_grad, "weight": k_weight}
        grad_cache[f"visual.transformer.resblocks.{idx}.attn.v"] = {"grad": v_grad, "weight": v_weight}
        out_grad = block.attn.out_proj.weight.grad
        if out_grad is not None:
            grad_cache[f"visual.transformer.resblocks.{idx}.attn.out_proj"] = {
                "grad": out_grad.detach().float(),
                "weight": block.attn.out_proj.weight.detach().float(),
            }
        fc_grad = block.mlp.c_fc.weight.grad
        if fc_grad is not None:
            grad_cache[f"visual.transformer.resblocks.{idx}.mlp.c_fc"] = {
                "grad": fc_grad.detach().float(),
                "weight": block.mlp.c_fc.weight.detach().float(),
            }
        proj_grad = block.mlp.c_proj.weight.grad
        if proj_grad is not None:
            grad_cache[f"visual.transformer.resblocks.{idx}.mlp.c_proj"] = {
                "grad": proj_grad.detach().float(),
                "weight": block.mlp.c_proj.weight.detach().float(),
            }
    return grad_cache


def _compensate_weight_(weight, lora_a, lora_b, scaling):
    delta = (lora_b @ lora_a).to(device=weight.device, dtype=weight.dtype) * scaling
    weight.sub_(delta)


def apply_svd_init(model, grad_cache, rank_pattern, init_scale, alpha, init_method):
    modules = {}
    for idx, block in enumerate(model.visual.transformer.resblocks):
        modules[f"visual.transformer.resblocks.{idx}.attn"] = block.attn
        modules[f"visual.transformer.resblocks.{idx}.attn.out_proj"] = block.attn.out_proj
        modules[f"visual.transformer.resblocks.{idx}.mlp.c_fc"] = block.mlp.c_fc
        modules[f"visual.transformer.resblocks.{idx}.mlp.c_proj"] = block.mlp.c_proj
    start = time.time()
    for name, item in grad_cache.items():
        rank = int(rank_pattern[name])
        scaling = alpha / rank if rank > 0 else 1.0
        if init_method == "svd_compensated":
            factors = compensated_svd_factors(item["grad"], item["weight"], rank, init_scale, scaling)
        elif init_method == "svd_compensated_svd":
            factors = compensated_svd_direction_factors(item["grad"], item["weight"], rank, init_scale, scaling)
        else:
            factors = svd_lora_factors(item["grad"], item["weight"], rank, init_scale, scaling, init_method)
        if factors is None:
            continue
        module = modules.get(name)
        with torch.no_grad():
            if isinstance(module, LoRALinear):
                module.lora_A.copy_(factors["lora_A"].to(module.lora_A.device, module.lora_A.dtype))
                module.lora_B.copy_(factors["lora_B"].to(module.lora_B.device, module.lora_B.dtype))
                if init_method.startswith("svd_compensated"):
                    _compensate_weight_(module.weight, module.lora_A, module.lora_B, scaling)
                continue
            module_name, which = name.rsplit(".", 1)
            module = modules[module_name]
            if which == "q":
                module.q_A.copy_(factors["lora_A"].to(module.q_A.device, module.q_A.dtype))
                module.q_B.copy_(factors["lora_B"].to(module.q_B.device, module.q_B.dtype))
                if init_method.startswith("svd_compensated"):
                    _compensate_weight_(module.in_proj_weight[: module.embed_dim], module.q_A, module.q_B, scaling)
            elif which == "k":
                module.k_A.copy_(factors["lora_A"].to(module.k_A.device, module.k_A.dtype))
                module.k_B.copy_(factors["lora_B"].to(module.k_B.device, module.k_B.dtype))
                if init_method.startswith("svd_compensated"):
                    _compensate_weight_(module.in_proj_weight[module.embed_dim : 2 * module.embed_dim], module.k_A, module.k_B, scaling)
            elif which == "v":
                module.v_A.copy_(factors["lora_A"].to(module.v_A.device, module.v_A.dtype))
                module.v_B.copy_(factors["lora_B"].to(module.v_B.device, module.v_B.dtype))
                if init_method.startswith("svd_compensated"):
                    _compensate_weight_(module.in_proj_weight[2 * module.embed_dim :], module.v_A, module.v_B, scaling)
    print(f"SVD init finished in {time.time() - start:.1f}s", flush=True)


def evaluate(model, loader, text_features, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    text_features = text_features.to(device)
    with torch.no_grad():
        progress = tqdm(loader, desc="Evaluating")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = torch.as_tensor(labels, device=device)
            loss, logits = clip_loss(model, images, labels, text_features)
            loss_sum += loss.item() * labels.numel()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.numel()
            progress.set_postfix(
                eval_loss=f"{loss_sum / total:.4f}",
                eval_acc=f"{correct / total:.4f}",
            )
    return {"loss": loss_sum / total, "accuracy": correct / total}


def save_lora_state(model, path, args, class_names, rank_pattern, rank_stats, metrics):
    lora_state = {}
    for name, param in model.named_parameters():
        if any(key in name for key in ("q_A", "q_B", "k_A", "k_B", "v_A", "v_B", "lora_A", "lora_B")):
            lora_state[name] = param.detach().cpu()
    torch.save(
        {
            "state_dict": lora_state,
            "args": vars(args),
            "class_names": class_names,
            "rank_pattern": rank_pattern,
            "rank_stats": rank_stats,
            "metrics": metrics,
        },
        path,
    )


def build_loraplus_param_groups(model, base_lr: float, ratio: float, weight_decay: float):
    lora_a_tokens = ("lora_A", "q_A", "k_A", "v_A")
    lora_b_tokens = ("lora_B", "q_B", "k_B", "v_B")
    lora_a_params = []
    lora_b_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(token in name for token in lora_a_tokens):
            lora_a_params.append(param)
        elif any(token in name for token in lora_b_tokens):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["gs_lora", "lora_official"], required=True)
    parser.add_argument("--dataset", choices=["sun397", "svhn", "dtd"], default="sun397")
    parser.add_argument("--data-root", default="/workspace/KeepLoRA/MTIL/data/Sun397")
    parser.add_argument("--partition-dir", default="/workspace/KeepLoRA/MTIL/data/Sun397/Partitions")
    parser.add_argument("--train-list", default="Training_01.txt")
    parser.add_argument("--test-list", default="Testing_01.txt")
    parser.add_argument("--svhn-root", default="/workspace/datasets/svhn")
    parser.add_argument("--svhn-single-digit-only", action="store_true")
    parser.add_argument("--dtd-root", default="/workspace/KeepLoRA/MTIL/data/DTD")
    parser.add_argument("--dtd-split-file", default="/workspace/KeepLoRA/MTIL/data/DTD/split_zhou_DescribableTextures.json")
    parser.add_argument("--dtd-train-split", default="train")
    parser.add_argument("--dtd-test-split", default="test")
    parser.add_argument("--clip-cache-root", default="/root/.cache/clip")
    parser.add_argument("--prompt-template", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loraplus-lr-ratio", type=float, default=16.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--base-rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--r-min", type=int, default=2)
    parser.add_argument("--r-max", type=int, default=16)
    parser.add_argument("--rank-budget-mode", choices=["independent", "param"], default="param")
    parser.add_argument("--param-budget", type=int, default=None)
    parser.add_argument("--calibration-steps", type=int, default=16)
    parser.add_argument("--init-method", choices=["svd_sqrt", "svd_a_zero_b", "svd_compensated", "svd_compensated_svd"], default="svd_sqrt")
    parser.add_argument("--svd-scale", "--init-scale", "--init_scale", dest="svd_scale", type=float, default=0.01)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.prompt_template is None:
        args.prompt_template = "a photo of a {}."

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = setup_file_logging(output_dir)
    with (output_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OpenAIClipClassifier(args.clip_cache_root)
    preprocess = model.preprocess

    if args.dataset == "sun397":
        root = Path(args.data_root)
        partition_dir = Path(args.partition_dir)
        class_names, train_samples = build_partition_samples(root, partition_dir, args.train_list)
        _, test_samples = build_partition_samples(root, partition_dir, args.test_list)
        if args.max_train_samples is not None:
            train_samples = train_samples[: args.max_train_samples]
        if args.max_eval_samples is not None:
            test_samples = test_samples[: args.max_eval_samples]
        train_dataset = Sun397Dataset(train_samples, preprocess)
        test_dataset = Sun397Dataset(test_samples, preprocess)
        print(f"SUN397: {len(class_names)} classes, {len(train_dataset)} train, {len(test_dataset)} test")
    else:
        if args.dataset == "svhn" and not args.svhn_single_digit_only:
            raise ValueError("SVHN currently expects --svhn-single-digit-only so labels map cleanly to 0-9.")
        if args.dataset == "svhn":
            data = load_dataset(args.svhn_root)
            class_names = [str(i) for i in range(10)]
            train_dataset = SvhnSingleDigitDataset(data["train"], preprocess)
            test_dataset = SvhnSingleDigitDataset(data["test"], preprocess)
            if args.max_train_samples is not None:
                train_dataset.indices = train_dataset.indices[: args.max_train_samples]
            if args.max_eval_samples is not None:
                test_dataset.indices = test_dataset.indices[: args.max_eval_samples]
            print(f"SVHN single digit: {len(class_names)} classes, {len(train_dataset)} train, {len(test_dataset)} test")
        else:
            class_names, train_samples, test_samples = build_dtd_split(
                args.dtd_root,
                args.dtd_split_file,
                args.dtd_train_split,
                args.dtd_test_split,
            )
            if args.max_train_samples is not None:
                train_samples = train_samples[: args.max_train_samples]
            if args.max_eval_samples is not None:
                test_samples = test_samples[: args.max_eval_samples]
            train_dataset = ImagePathDataset(train_samples, preprocess)
            test_dataset = ImagePathDataset(test_samples, preprocess)
            print(f"DTD: {len(class_names)} classes, {len(train_dataset)} train, {len(test_dataset)} test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    calib_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.method == "gs_lora":
        uniform = {}
        for idx in range(len(model.visual.transformer.resblocks)):
            uniform[f"visual.transformer.resblocks.{idx}.attn.q"] = args.base_rank
            uniform[f"visual.transformer.resblocks.{idx}.attn.k"] = args.base_rank
            uniform[f"visual.transformer.resblocks.{idx}.attn.v"] = args.base_rank
            uniform[f"visual.transformer.resblocks.{idx}.attn.out_proj"] = args.base_rank
            uniform[f"visual.transformer.resblocks.{idx}.mlp.c_fc"] = args.base_rank
            uniform[f"visual.transformer.resblocks.{idx}.mlp.c_proj"] = args.base_rank
        model.inject_lora(uniform, args.alpha, args.dropout)
        model.to(device)
        text_features = build_text_features(model, class_names, args.prompt_template, device).to(device)
        set_base_attention_trainable(model, True)
        model.train()
        for p in model.parameters():
            p.grad = None
        for step, (images, labels) in enumerate(tqdm(calib_loader, desc="Calibrating", total=args.calibration_steps)):
            if step >= args.calibration_steps:
                break
            images = images.to(device, non_blocking=True)
            labels = torch.as_tensor(labels, device=device)
            loss, _ = clip_loss(model, images, labels, text_features)
            loss = loss / args.calibration_steps
            loss.backward()
        grad_cache = collect_grad_cache(model)
        print(f"Collected gradients for {len(grad_cache)} LoRA matrices", flush=True)
        print("Allocating adaptive ranks...", flush=True)
        start = time.time()
        if args.rank_budget_mode == "param":
            rank_costs = {
                name: int(item["grad"].reshape(item["grad"].shape[0], -1).shape[0] + item["grad"].reshape(item["grad"].shape[0], -1).shape[1])
                for name, item in grad_cache.items()
            }
            singular_values = {
                name: torch.linalg.svdvals(item["grad"].detach().float().reshape(item["grad"].shape[0], -1))
                for name, item in grad_cache.items()
            }
            param_budget = args.param_budget
            if param_budget is None:
                param_budget = sum(
                    min(args.base_rank, int(values.numel()), args.r_max) * rank_costs[name]
                    for name, values in singular_values.items()
                )
            rank_pattern = allocate_param_budget(singular_values, rank_costs, args.r_min, args.r_max, param_budget)
            rank_stats = {}
            for name, values in singular_values.items():
                rank = rank_pattern[name]
                energy = values.pow(2)
                total = energy.sum().item()
                kept = energy[:rank].sum().item()
                rank_stats[name] = {
                    "rank": float(rank),
                    "energy_ratio": 0.0 if total == 0 else kept / total,
                    "total_energy": total,
                    "rank_cost": rank_costs[name],
                    "param_count": rank * rank_costs[name],
                }
        else:
            rank_pattern, rank_stats = allocate_ranks(grad_cache, RankConfig(args.tau, args.r_min, args.r_max, args.base_rank))
        print(f"Rank allocation finished in {time.time() - start:.1f}s", flush=True)
        print("Rebuilding model and applying SVD init...", flush=True)
        del model
        torch.cuda.empty_cache()
        model = OpenAIClipClassifier(args.clip_cache_root)
        model.inject_lora(rank_pattern, args.alpha, args.dropout)
        model.to(device)
        text_features = build_text_features(model, class_names, args.prompt_template, device).to(device)
        apply_svd_init(model, grad_cache, rank_pattern, args.svd_scale, args.alpha, args.init_method)
        model.freeze_backbone()
        with (output_dir / "rank_pattern.json").open("w") as f:
            json.dump(rank_pattern, f, indent=2)
        with (output_dir / "rank_stats.json").open("w") as f:
            json.dump(rank_stats, f, indent=2)
    else:
        rank_pattern = {}
        rank_stats = {}
        for idx in range(len(model.visual.transformer.resblocks)):
            rank_pattern[f"visual.transformer.resblocks.{idx}.attn.q"] = args.base_rank
            rank_pattern[f"visual.transformer.resblocks.{idx}.attn.k"] = args.base_rank
            rank_pattern[f"visual.transformer.resblocks.{idx}.attn.v"] = args.base_rank
            rank_pattern[f"visual.transformer.resblocks.{idx}.attn.out_proj"] = args.base_rank
            rank_pattern[f"visual.transformer.resblocks.{idx}.mlp.c_fc"] = args.base_rank
            rank_pattern[f"visual.transformer.resblocks.{idx}.mlp.c_proj"] = args.base_rank
        model.inject_lora(rank_pattern, args.alpha, args.dropout)
        model.to(device)
        text_features = build_text_features(model, class_names, args.prompt_template, device).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")

    optimizer = torch.optim.AdamW(
        build_loraplus_param_groups(model, args.lr, args.loraplus_lr_ratio, args.weight_decay),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    metrics = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = torch.as_tensor(labels, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss, _ = clip_loss(model, images, labels, text_features)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * labels.numel()
            seen += labels.numel()
            progress.set_postfix(
                batch_loss=f"{loss.item():.4f}",
                avg_loss=f"{loss_sum / seen:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
        metrics = evaluate(model, test_loader, text_features, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = loss_sum / seen
        print(json.dumps(metrics, indent=2))

    metrics["final_accuracy"] = metrics["accuracy"]
    metrics["final_loss"] = metrics["loss"]
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    save_lora_state(model, output_dir / "adapter.pt", args, class_names, rank_pattern, rank_stats, metrics)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()


if __name__ == "__main__":
    main()
