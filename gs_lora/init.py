from typing import Dict

import torch


def svd_lora_factors(
    grad,
    rank: int,
    method: str = "svd_sqrt",
    init_scale: float = 1e-3,
    effective_scaling: float = 1.0,
):
    if method == "none":
        return None
    if rank <= 0:
        return None

    grad_matrix = grad.detach().float()
    if grad_matrix.ndim > 2:
        grad_matrix = grad_matrix.reshape(grad_matrix.shape[0], -1)

    u, singular_values, vh = torch.linalg.svd(grad_matrix, full_matrices=False)
    rank = min(rank, singular_values.numel())
    if rank <= 0:
        return None

    u = u[:, :rank]
    singular_values = singular_values[:rank]
    vh = vh[:rank, :]

    if method == "svd_sqrt":
        root = torch.sqrt(singular_values.clamp_min(0.0))
        lora_b = -u * root.unsqueeze(0)
        lora_a = root.unsqueeze(1) * vh
    elif method == "svd_sigma":
        lora_b = -u * singular_values.unsqueeze(0)
        lora_a = vh
    else:
        raise ValueError(f"Unknown init_method: {method}")

    factor_scale = init_scale / effective_scaling if effective_scaling > 0 else init_scale
    factor_scale = factor_scale ** 0.5
    return {
        "lora_A": (lora_a * factor_scale).cpu(),
        "lora_B": (lora_b * factor_scale).cpu(),
    }


def compute_effective_scaling(rank: int, avg_rank: float, config) -> float:
    rank = max(float(rank), 1.0)
    avg_rank = max(float(avg_rank), 1.0)
    if config.scaling_mode == "rank":
        return float(config.lora_alpha) / rank
    if config.scaling_mode == "sqrt_rank":
        return float(config.lora_alpha) / (rank ** 0.5)
    if config.scaling_mode == "avg_rank":
        return float(config.lora_alpha) / ((rank * avg_rank) ** 0.5)
    raise ValueError(f"Unknown scaling_mode: {config.scaling_mode}")


def build_init_state(grad_cache, rank_pattern: Dict[str, int], config):
    if config.init_method == "none":
        return {}

    init_state = {}
    ranks = [int(rank) for rank in rank_pattern.values()]
    avg_rank = sum(ranks) / len(ranks) if ranks else config.base_rank
    for name, item in grad_cache.items():
        rank = int(rank_pattern[name])
        scaling = compute_effective_scaling(rank, avg_rank, config) if config.compensate_scaling else 1.0
        factors = svd_lora_factors(
            item["grad"],
            rank,
            method=config.init_method,
            init_scale=config.init_scale,
            effective_scaling=scaling,
        )
        if factors is not None:
            init_state[name] = factors
    return init_state
