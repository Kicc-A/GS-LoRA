#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Callable, Dict, Iterable, Optional, Tuple, Union

from .layers import LoRALayer


RankPattern = Union[int, Iterable[int], Dict[Union[str, int], int], Callable[..., int]]


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def compute_adaptive_rank_from_gradient(
    grad_matrix: torch.Tensor,
    tau: float = 0.90,
    r_min: int = 2,
    r_max: int = 32,
    eps: float = 1e-12,
) -> int:
    """Compute one LoRA rank from the SVD energy of a weight gradient.

    The returned rank is the smallest r whose cumulative squared singular-value
    energy reaches tau, clamped into [r_min, r_max].
    """
    if grad_matrix is None:
        raise ValueError("grad_matrix must not be None")
    if grad_matrix.ndim < 2:
        raise ValueError("grad_matrix must have at least 2 dimensions")
    if not 0 < tau <= 1:
        raise ValueError("tau must be in (0, 1]")
    if r_min < 0 or r_max < 0 or r_min > r_max:
        raise ValueError("expected 0 <= r_min <= r_max")

    grad = grad_matrix.detach().float()
    if grad.ndim > 2:
        grad = grad.reshape(grad.shape[0], -1)

    max_rank = min(grad.shape)
    r_min = min(r_min, max_rank)
    r_max = min(r_max, max_rank)
    if max_rank == 0 or r_max == 0:
        return 0

    singular_values = torch.linalg.svdvals(grad)
    energy = singular_values.pow(2)
    total_energy = energy.sum()
    if total_energy <= eps:
        return r_min

    cumulative_energy = torch.cumsum(energy, dim=0) / total_energy
    rank = int(torch.searchsorted(cumulative_energy, grad.new_tensor(tau)).item() + 1)
    return max(r_min, min(rank, r_max))


def compute_adaptive_rank_pattern_from_gradients(
    model: nn.Module,
    target_modules: Optional[Union[str, Iterable[str], Callable[[str, nn.Module], bool]]] = None,
    tau: float = 0.90,
    r_min: int = 2,
    r_max: int = 32,
    return_energy: bool = False,
) -> Union[Dict[str, int], Tuple[Dict[str, int], Dict[str, Dict[str, float]]]]:
    """Build a module-name -> rank pattern from already-computed gradients.

    Run a calibration forward/backward pass before calling this function. By
    default, every module with a 2D-or-higher ``weight.grad`` is considered.
    ``target_modules`` may be an iterable of exact module names/suffixes, or a
    predicate receiving ``(name, module)``.
    """
    ranks: Dict[str, int] = {}
    stats: Dict[str, Dict[str, float]] = {}
    if isinstance(target_modules, str):
        target_set = {target_modules}
    elif target_modules is not None and not callable(target_modules):
        target_set = set(target_modules)
    else:
        target_set = None

    for name, module in model.named_modules():
        if not name:
            continue
        if callable(target_modules):
            if not target_modules(name, module):
                continue
        elif target_set is not None:
            if name not in target_set and not any(name.endswith("." + target) for target in target_set):
                continue

        weight = getattr(module, "weight", None)
        if weight is None:
            conv = getattr(module, "conv", None)
            weight = getattr(conv, "weight", None) if conv is not None else None
        grad = getattr(weight, "grad", None)
        if grad is None or grad.ndim < 2:
            continue

        rank = compute_adaptive_rank_from_gradient(grad, tau=tau, r_min=r_min, r_max=r_max)
        ranks[name] = rank

        if return_energy:
            flat_grad = grad.detach().float().reshape(grad.shape[0], -1)
            singular_values = torch.linalg.svdvals(flat_grad)
            energy = singular_values.pow(2)
            total_energy = energy.sum().item()
            kept = energy[:rank].sum().item()
            stats[name] = {
                "rank": float(rank),
                "energy_ratio": 0.0 if total_energy == 0.0 else kept / total_energy,
                "total_energy": total_energy,
            }

    if return_energy:
        return ranks, stats
    return ranks


def get_rank_from_pattern(
    name: str,
    default_rank: int,
    rank_pattern: Optional[RankPattern] = None,
    layer_idx: Optional[int] = None,
) -> int:
    """Resolve a LoRA rank from an optional adaptive-rank pattern.

    Supported patterns:
      * ``None`` or int: use the fixed rank
      * list/tuple: index by ``layer_idx``
      * dict: exact name, suffix name, or integer layer index
      * callable: called as ``rank_pattern(name, default_rank, layer_idx)``
    """
    if rank_pattern is None:
        return int(default_rank)
    if isinstance(rank_pattern, int):
        return int(rank_pattern)
    if callable(rank_pattern):
        return int(rank_pattern(name, default_rank, layer_idx))
    if isinstance(rank_pattern, dict):
        if name in rank_pattern:
            return int(rank_pattern[name])
        for key, rank in rank_pattern.items():
            if isinstance(key, str) and name.endswith(key):
                return int(rank)
        if layer_idx is not None and layer_idx in rank_pattern:
            return int(rank_pattern[layer_idx])
        return int(default_rank)
    if layer_idx is not None:
        pattern = list(rank_pattern)
        if 0 <= layer_idx < len(pattern):
            return int(pattern[layer_idx])
    return int(default_rank)


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
