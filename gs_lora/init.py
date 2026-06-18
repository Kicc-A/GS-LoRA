from typing import Dict

import torch

SVD_INIT_METHODS = {
    "none",
    "svd_sqrt",
    "svd_sigma",
    "svd_a_zero_b",
    "svd_a_energy_zero_b",
    "svd_a_energy_small_b",
}


def _svd_compute_device(source: torch.Tensor) -> torch.device:
    if source.is_cuda:
        return source.device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return source.device


def _energy_weights(
    singular_values: torch.Tensor,
    beta: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    energy = singular_values.pow(2)
    prob = energy / (energy.sum() + eps)
    weights = (prob + eps).pow(beta)
    weights = weights / (weights.pow(2).mean().sqrt() + eps)
    return weights


def svd_lora_factors(
    grad,
    rank: int,
    method: str = "svd_a_zero_b",
    init_scale: float = 1e-3,
    effective_scaling: float = 1.0,
    energy_beta: float = 0.5,
    energy_eps: float = 1e-8,
    small_b_scale: float = 1e-4,
):
    if method == "none":
        return None
    if rank <= 0:
        return None

    grad_matrix = grad.detach().float()
    if grad_matrix.ndim > 2:
        grad_matrix = grad_matrix.reshape(grad_matrix.shape[0], -1)
    grad_matrix = grad_matrix.to(_svd_compute_device(grad_matrix), non_blocking=True)

    u, singular_values, vh = torch.linalg.svd(grad_matrix, full_matrices=False)
    rank = min(rank, singular_values.numel())
    if rank <= 0:
        return None

    u = u[:, :rank]
    singular_values = singular_values[:rank]
    vh = vh[:rank, :]

    if method == "svd_sqrt":
        root = torch.sqrt(singular_values.clamp_min(0.0))
        lora_a = root.unsqueeze(1) * vh
        lora_b = -u * root.unsqueeze(0)
    elif method == "svd_sigma":
        lora_a = vh
        lora_b = -u * singular_values.unsqueeze(0)
    elif method == "svd_a_zero_b":
        lora_a = vh
        lora_b = torch.zeros(
            grad_matrix.shape[0],
            rank,
            device=grad_matrix.device,
            dtype=grad_matrix.dtype,
        )
    elif method == "svd_a_energy_zero_b":
        weights = _energy_weights(
            singular_values,
            beta=energy_beta,
            eps=energy_eps,
        )
        lora_a = weights.unsqueeze(1) * vh
        lora_b = torch.zeros(
            grad_matrix.shape[0],
            rank,
            device=grad_matrix.device,
            dtype=grad_matrix.dtype,
        )
    elif method == "svd_a_energy_small_b":
        weights = _energy_weights(
            singular_values,
            beta=energy_beta,
            eps=energy_eps,
        )
        lora_a = weights.unsqueeze(1) * vh
        lora_b = -u * weights.unsqueeze(0)
    else:
        raise ValueError(f"Unknown init_method: {method}")

    factor_scale = init_scale / effective_scaling if effective_scaling > 0 else init_scale
    factor_scale = factor_scale ** 0.5
    lora_a = lora_a * factor_scale
    if method in {"svd_sqrt", "svd_sigma"}:
        lora_b = lora_b * factor_scale
    elif method == "svd_a_energy_small_b":
        lora_b = lora_b * factor_scale * small_b_scale
    return {
        "lora_A": lora_a.cpu(),
        "lora_B": lora_b.cpu(),
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
    if config.init_method not in SVD_INIT_METHODS:
        raise ValueError(f"Unsupported init_method: {config.init_method}")

    ranks = [int(rank) for rank in rank_pattern.values()]
    avg_rank = sum(ranks) / len(ranks) if ranks else config.base_rank
    init_state = {}
    for name, item in grad_cache.items():
        rank = int(rank_pattern[name])
        scaling = compute_effective_scaling(rank, avg_rank, config)
        factors = svd_lora_factors(
            item["grad"],
            rank,
            method=config.init_method,
            init_scale=config.init_scale,
            effective_scaling=scaling,
            energy_beta=getattr(config, "init_energy_beta", 0.5),
            energy_eps=getattr(config, "init_energy_eps", 1e-8),
            small_b_scale=getattr(config, "init_small_b_scale", 1e-4),
        )
        if factors is not None:
            init_state[name] = factors
    return init_state
