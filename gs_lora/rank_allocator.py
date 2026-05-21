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
        rank = adaptive_rank_from_gradient(
            grad,
            tau=config.tau,
            r_min=config.r_min,
            r_max=config.r_max,
        )
        flat_grad = grad.detach().float().reshape(grad.shape[0], -1)
        singular_values = torch.linalg.svdvals(flat_grad)
        energy = singular_values.pow(2)
        total_energy = energy.sum().item()
        kept = energy[:rank].sum().item()

        rank_pattern[name] = int(rank)
        rank_stats[name] = {
            "rank": float(rank),
            "energy_ratio": 0.0 if total_energy == 0.0 else kept / total_energy,
            "total_energy": total_energy,
        }

    return rank_pattern, rank_stats


def allocate_ranks(
    grad_cache: Dict[str, Dict[str, torch.Tensor]],
    config,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, float]]]:
    return allocate_independent_ranks(grad_cache, config)
