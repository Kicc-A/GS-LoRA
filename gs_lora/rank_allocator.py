from typing import Dict, Optional, Tuple

import torch


def _svd_compute_device(source: torch.Tensor) -> torch.device:
    if source.is_cuda:
        return source.device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return source.device



def _is_auto(value) -> bool:
    return isinstance(value, str) and value.lower() == "auto"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value, default: float) -> float:
    if _is_auto(value):
        return default
    return float(value)


def _energy_concentration(energy: torch.Tensor, total_energy: float) -> float:
    if total_energy <= 1e-12 or energy.numel() == 0:
        return 0.0
    return float((energy[0] / max(float(total_energy), 1e-12)).item())


def _base_candidate_score(marginal_energy: float, total_energy: float, rank_cost: int, gamma: float) -> float:
    if marginal_energy <= 0.0:
        return 0.0
    relative_energy = marginal_energy / max(float(total_energy), 1e-12)
    sharpened_energy = marginal_energy * (relative_energy ** (gamma - 1.0))
    return sharpened_energy / max(rank_cost, 1)


def _resolve_rank_score_gamma(module_items, config) -> float:
    configured = getattr(config, "rank_score_gamma", 1.0)
    if not _is_auto(configured):
        return max(float(configured), 1e-6)
    concentrations = [
        _energy_concentration(values["energy"], values["total_energy"])
        for values in module_items.values()
        if values["total_energy"] > 1e-12
    ]
    if not concentrations:
        return 1.0
    mean_concentration = sum(concentrations) / len(concentrations)
    return 1.0 + _clamp((mean_concentration - 0.15) / 0.35, 0.0, 0.5)


def _resolve_attention_rank_prior(module_items, config, gamma: float) -> float:
    configured = getattr(config, "attention_rank_prior", 1.0)
    if not _is_auto(configured):
        return float(configured)
    attn_scores = []
    other_scores = []
    for name, values in module_items.items():
        total_energy = values["total_energy"]
        rank_cost = values["rank_cost"]
        energy = values["energy"]
        for rank_index in range(values["r_min"], values["r_max"]):
            marginal_energy = energy[rank_index].item() if rank_index < energy.numel() else 0.0
            score = _base_candidate_score(marginal_energy, total_energy, rank_cost, gamma)
            if score <= 0.0:
                continue
            if _target_module_leaf(name) in {"q", "k", "v", "o"}:
                attn_scores.append(score)
            else:
                other_scores.append(score)
    if not attn_scores or not other_scores:
        return 1.0
    attn_mean = sum(attn_scores) / len(attn_scores)
    other_mean = sum(other_scores) / len(other_scores)
    return _clamp(attn_mean / max(other_mean, 1e-12), 0.8, 1.2)

def _target_module_leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _module_rank_prior(name: str, config, resolved_attention_prior: float = None) -> float:
    leaf = _target_module_leaf(name)
    if leaf in {"q", "k", "v", "o"}:
        if resolved_attention_prior is not None:
            return float(resolved_attention_prior)
        return _safe_float(getattr(config, "attention_rank_prior", 1.0), 1.0)
    return 1.0


def _marginal_rank_score(
    marginal_energy: float,
    total_energy: float,
    rank_cost: int,
    name: str,
    config,
    resolved_gamma: float = None,
    resolved_attention_prior: float = None,
) -> float:
    gamma = max(float(resolved_gamma if resolved_gamma is not None else _safe_float(getattr(config, "rank_score_gamma", 1.0), 1.0)), 1e-6)
    return _module_rank_prior(name, config, resolved_attention_prior) * _base_candidate_score(
        marginal_energy, total_energy, rank_cost, gamma
    )


def _flatten_gradient(grad: torch.Tensor) -> torch.Tensor:
    flat_grad = grad.detach().float()
    if flat_grad.ndim > 2:
        flat_grad = flat_grad.reshape(flat_grad.shape[0], -1)
    return flat_grad


def _squared_svd_energy(flat_grad: torch.Tensor) -> torch.Tensor:
    flat_grad = flat_grad.to(_svd_compute_device(flat_grad), non_blocking=True)
    return torch.linalg.svdvals(flat_grad).pow(2)


def _candidate_energy_for(name: str, fallback: torch.Tensor, candidate_energy) -> torch.Tensor:
    if candidate_energy is None or name not in candidate_energy:
        return fallback
    candidate = candidate_energy[name]
    if not torch.is_tensor(candidate):
        candidate = torch.tensor(candidate, dtype=torch.float32)
    return candidate.detach().float().to(fallback.device, non_blocking=True)


def _rank_report_summary(rank_pattern, module_items, budget_target=None, budget_used=None):
    ranks = [int(rank) for rank in rank_pattern.values()]
    if not ranks:
        target = int(budget_target or 0)
        used = int(budget_used or 0)
        return {
            "total_adapter_params": used,
            "budget_target": target,
            "budget_used": used,
            "budget_unused": max(target - used, 0),
            "rank_min": 0,
            "rank_mean": 0.0,
            "rank_max": 0,
            "num_layers_at_r_min": 0,
            "num_layers_at_r_max": 0,
        }
    total_params = int(sum(int(rank_pattern[name]) * int(module_items[name]["rank_cost"]) for name in rank_pattern))
    target = int(total_params if budget_target is None else budget_target)
    used = int(total_params if budget_used is None else budget_used)
    return {
        "total_adapter_params": total_params,
        "budget_target": target,
        "budget_used": used,
        "budget_unused": max(target - used, 0),
        "rank_min": min(ranks),
        "rank_mean": float(sum(ranks) / len(ranks)),
        "rank_max": max(ranks),
        "num_layers_at_r_min": int(sum(1 for name, rank in rank_pattern.items() if int(rank) == int(module_items[name]["r_min"]))),
        "num_layers_at_r_max": int(sum(1 for name, rank in rank_pattern.items() if int(rank) == int(module_items[name]["r_max"]))),
    }


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
    grad = grad.to(_svd_compute_device(grad), non_blocking=True)

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
    module_items = {}

    for name, item in grad_cache.items():
        grad = item["grad"]
        flat_grad = _flatten_gradient(grad)
        energy = _squared_svd_energy(flat_grad)
        total_energy = energy.sum().item()
        max_rank = min(flat_grad.shape)
        r_min = min(config.r_min, max_rank)
        r_max = min(config.r_max, max_rank)
        rank_cost = int(flat_grad.shape[0] + flat_grad.shape[1])
        module_items[name] = {"r_min": r_min, "r_max": r_max, "rank_cost": rank_cost}
        if max_rank == 0 or r_max == 0:
            rank = 0
        elif total_energy <= 1e-12:
            rank = r_min
        else:
            cumulative_energy = torch.cumsum(energy, dim=0) / energy.sum()
            rank = int(torch.searchsorted(cumulative_energy, energy.new_tensor(config.tau)).item() + 1)
            rank = max(r_min, min(rank, r_max))
        kept = energy[:rank].sum().item()

        rank_pattern[name] = int(rank)
        rank_stats[name] = {
            "module_name": name,
            "rank": float(rank),
            "rank_cost": rank_cost,
            "param_count": int(rank * rank_cost),
            "energy_ratio": 0.0 if total_energy == 0.0 else kept / total_energy,
            "total_energy": total_energy,
        }

    rank_stats["__summary__"] = _rank_report_summary(rank_pattern, module_items)
    return rank_pattern, rank_stats

def allocate_global_param_budget_ranks(
    grad_cache: Dict[str, Dict[str, torch.Tensor]],
    config,
    candidate_energy: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, float]]]:
    module_items = {}
    min_cost = 0
    base_cost = 0
    max_cost = 0

    for name, item in grad_cache.items():
        grad = item["grad"]
        flat_grad = _flatten_gradient(grad)
        energy = _squared_svd_energy(flat_grad)
        selection_energy = _candidate_energy_for(name, energy, candidate_energy)
        total_energy = energy.sum().item()
        selection_total_energy = selection_energy.sum().item()
        max_rank = min(flat_grad.shape)
        r_min = min(config.r_min, max_rank)
        r_max = min(config.r_max, max_rank)
        rank_cost = int(flat_grad.shape[0] + flat_grad.shape[1])

        module_items[name] = {
            "energy": energy,
            "selection_energy": selection_energy,
            "total_energy": total_energy,
            "selection_total_energy": selection_total_energy,
            "r_min": r_min,
            "r_max": r_max,
            "rank_cost": rank_cost,
        }
        min_cost += r_min * rank_cost
        base_cost += min(config.base_rank, max_rank) * rank_cost
        max_cost += r_max * rank_cost

    resolved_gamma = _resolve_rank_score_gamma(module_items, config)
    resolved_attention_prior = _resolve_attention_rank_prior(module_items, config, resolved_gamma)

    candidates = []
    for name, values in module_items.items():
        energy = values["selection_energy"]
        total_energy = values["selection_total_energy"]
        rank_cost = values["rank_cost"]
        for rank_index in range(values["r_min"], values["r_max"]):
            marginal_energy = energy[rank_index].item() if rank_index < energy.numel() else 0.0
            score = _marginal_rank_score(
                marginal_energy,
                total_energy,
                rank_cost,
                name,
                config,
                resolved_gamma=resolved_gamma,
                resolved_attention_prior=resolved_attention_prior,
            )
            candidates.append((score, marginal_energy, rank_cost, name, rank_index))

    budget_arg = getattr(config, "param_budget", None)
    budget = int(budget_arg) if budget_arg is not None else base_cost
    budget = max(min_cost, min(budget, max_cost))
    rank_pattern = {name: int(values["r_min"]) for name, values in module_items.items()}
    used_cost = min_cost

    candidates.sort(reverse=True)
    stage1_rank = getattr(config, "stable_budget_stage1_rank", None)
    stage1_rank = int(stage1_rank) if stage1_rank is not None and int(stage1_rank) > 0 else None

    def apply_candidates(limit_rank=None):
        nonlocal used_cost
        changed = True
        while changed:
            changed = False
            for _, _, rank_cost, name, rank_index in candidates:
                values = module_items[name]
                current_rank = rank_pattern[name]
                allowed_max = values["r_max"] if limit_rank is None else min(values["r_max"], int(limit_rank))
                if current_rank != rank_index or current_rank >= allowed_max:
                    continue
                if used_cost + rank_cost > budget:
                    continue
                rank_pattern[name] += 1
                used_cost += rank_cost
                changed = True
                break

    if stage1_rank is not None:
        apply_candidates(stage1_rank)
    apply_candidates(None)

    rank_stats: Dict[str, Dict[str, float]] = {}
    for name, rank in rank_pattern.items():
        values = module_items[name]
        energy = values["energy"]
        total_energy = values["total_energy"]
        kept = energy[:rank].sum().item()
        rank_cost = values["rank_cost"]
        rank_stats[name] = {
            "module_name": name,
            "rank": float(rank),
            "rank_cost": rank_cost,
            "param_count": int(rank * rank_cost),
            "energy_ratio": 0.0 if total_energy == 0.0 else kept / total_energy,
            "total_energy": total_energy,
            "budget_param_count": int(used_cost),
            "budget_target": int(budget),
            "budget_base_rank_param_count": int(base_cost),
            "rank_score_gamma": float(resolved_gamma),
            "attention_rank_prior": float(resolved_attention_prior),
            "module_rank_prior": float(_module_rank_prior(name, config, resolved_attention_prior)),
            "used_candidate_energy": bool(candidate_energy is not None and name in candidate_energy),
            "stable_budget_stage1_rank": int(stage1_rank) if stage1_rank is not None else None,
        }

    rank_stats["__summary__"] = _rank_report_summary(rank_pattern, module_items, budget, used_cost)
    return rank_pattern, rank_stats

def allocate_ranks(
    grad_cache: Dict[str, Dict[str, torch.Tensor]],
    config,
    candidate_energy: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, float]]]:
    if getattr(config, "rank_budget_mode", "independent") == "param":
        return allocate_global_param_budget_ranks(grad_cache, config, candidate_energy=candidate_energy)
    return allocate_independent_ranks(grad_cache, config)
