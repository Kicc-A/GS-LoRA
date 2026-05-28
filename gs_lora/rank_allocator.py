from typing import Dict, Tuple

import torch


def adaptive_rank_from_gradient(
    grad_matrix: torch.Tensor,
    tau: float = 0.90,
    r_min: int = 2,
    r_max: int = 32,
    eps: float = 1e-12,
) -> int:
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


def allocate_independent_ranks(
    grad_cache: Dict[str, Dict[str, torch.Tensor]],
    config,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, float]]]:
    rank_pattern: Dict[str, int] = {}
    rank_stats: Dict[str, Dict[str, float]] = {}

    for name, item in grad_cache.items():
        grad = item["grad"]
        flat_grad = grad.detach().float().reshape(grad.shape[0], -1)
        singular_values = torch.linalg.svdvals(flat_grad)
        energy = singular_values.pow(2)
        total_energy = energy.sum().item()
        max_rank = min(flat_grad.shape)
        r_min = min(config.r_min, max_rank)
        r_max = min(config.r_max, max_rank)
        if max_rank == 0 or r_max == 0:
            rank = 0
        elif total_energy <= 1e-12:
            rank = r_min
        else:
            cumulative_energy = torch.cumsum(energy, dim=0) / energy.sum()
            rank = int(torch.searchsorted(cumulative_energy, flat_grad.new_tensor(config.tau)).item() + 1)
            rank = max(r_min, min(rank, r_max))
        kept = energy[:rank].sum().item()

        rank_pattern[name] = int(rank)
        rank_stats[name] = {
            "rank": float(rank),
            "energy_ratio": 0.0 if total_energy == 0.0 else kept / total_energy,
            "total_energy": total_energy,
        }

    return rank_pattern, rank_stats


def allocate_global_param_budget_ranks(
    grad_cache: Dict[str, Dict[str, torch.Tensor]],
    config,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, float]]]:
    module_items = {}
    min_cost = 0
    base_cost = 0
    max_cost = 0
    candidates = []

    for name, item in grad_cache.items():
        grad = item["grad"]
        flat_grad = grad.detach().float().reshape(grad.shape[0], -1)
        singular_values = torch.linalg.svdvals(flat_grad)
        energy = singular_values.pow(2)
        total_energy = energy.sum().item()
        max_rank = min(flat_grad.shape)
        r_min = min(config.r_min, max_rank)
        r_max = min(config.r_max, max_rank)
        rank_cost = int(flat_grad.shape[0] + flat_grad.shape[1])

        module_items[name] = {
            "energy": energy,
            "total_energy": total_energy,
            "r_min": r_min,
            "r_max": r_max,
            "rank_cost": rank_cost,
        }
        min_cost += r_min * rank_cost
        base_cost += min(config.base_rank, r_max) * rank_cost
        max_cost += r_max * rank_cost

        for rank_index in range(r_min, r_max):
            marginal_energy = energy[rank_index].item() if rank_index < energy.numel() else 0.0
            candidates.append((marginal_energy / max(rank_cost, 1), marginal_energy, rank_cost, name))

    budget = int(config.param_budget) if config.param_budget is not None else base_cost
    budget = max(min_cost, min(budget, max_cost))
    rank_pattern = {name: int(values["r_min"]) for name, values in module_items.items()}
    used_cost = min_cost

    candidates.sort(reverse=True)
    for _, _, rank_cost, name in candidates:
        values = module_items[name]
        if rank_pattern[name] >= values["r_max"]:
            continue
        if used_cost + rank_cost > budget:
            continue
        rank_pattern[name] += 1
        used_cost += rank_cost

    rank_stats: Dict[str, Dict[str, float]] = {}
    for name, rank in rank_pattern.items():
        values = module_items[name]
        energy = values["energy"]
        total_energy = values["total_energy"]
        kept = energy[:rank].sum().item()
        rank_cost = values["rank_cost"]
        rank_stats[name] = {
            "rank": float(rank),
            "energy_ratio": 0.0 if total_energy == 0.0 else kept / total_energy,
            "total_energy": total_energy,
            "rank_cost": rank_cost,
            "param_count": int(rank * rank_cost),
            "budget_param_count": int(used_cost),
            "budget_target": int(budget),
            "budget_base_rank_param_count": int(base_cost),
        }

    return rank_pattern, rank_stats


def allocate_ranks(
    grad_cache: Dict[str, Dict[str, torch.Tensor]],
    config,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, float]]]:
    if getattr(config, "rank_budget_mode", "independent") == "param":
        return allocate_global_param_budget_ranks(grad_cache, config)
    return allocate_independent_ranks(grad_cache, config)
