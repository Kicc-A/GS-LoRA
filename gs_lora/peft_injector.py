def _get_active_adapter_name(module):
    if hasattr(module, "active_adapters"):
        active_adapters = module.active_adapters
        active_adapters = active_adapters() if callable(active_adapters) else active_adapters
        if active_adapters:
            return active_adapters[0]
    if hasattr(module, "active_adapter"):
        active_adapter = module.active_adapter
        active_adapter = active_adapter() if callable(active_adapter) else active_adapter
        if active_adapter:
            return active_adapter
    return "default"


def apply_svd_init(model, init_state):
    import torch

    if not init_state:
        return {}

    applied = {}
    for peft_name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue

        original_name = next((name for name in init_state if peft_name.endswith(name)), None)
        if original_name is None:
            continue

        adapter_name = _get_active_adapter_name(module)
        lora_a = module.lora_A[adapter_name]
        lora_b = module.lora_B[adapter_name]
        init = init_state[original_name]

        if tuple(lora_a.weight.shape) != tuple(init["lora_A"].shape):
            raise ValueError(
                f"LoRA A shape mismatch for {original_name}: "
                f"expected {tuple(lora_a.weight.shape)}, got {tuple(init['lora_A'].shape)}"
            )
        if tuple(lora_b.weight.shape) != tuple(init["lora_B"].shape):
            raise ValueError(
                f"LoRA B shape mismatch for {original_name}: "
                f"expected {tuple(lora_b.weight.shape)}, got {tuple(init['lora_B'].shape)}"
            )

        with torch.no_grad():
            lora_a.weight.copy_(init["lora_A"].to(device=lora_a.weight.device, dtype=lora_a.weight.dtype))
            lora_b.weight.copy_(init["lora_B"].to(device=lora_b.weight.device, dtype=lora_b.weight.dtype))
        applied[original_name] = peft_name

    missing = sorted(set(init_state) - set(applied))
    if missing:
        raise ValueError(f"SVD init not applied for: {missing[:5]}")

    return applied


def _compute_gora_pinv_b(grad, lora_a_weight, module, config, rank, avg_rank):
    import torch

    grad_matrix = grad.to(device=lora_a_weight.device, dtype=torch.float32)
    if grad_matrix.ndim > 2:
        grad_matrix = grad_matrix.reshape(grad_matrix.shape[0], -1)

    lora_a = lora_a_weight.detach().to(dtype=torch.float32)
    if grad_matrix.shape[1] != lora_a.shape[1]:
        raise ValueError(
            f"GoRA init shape mismatch: grad={tuple(grad_matrix.shape)}, "
            f"lora_A={tuple(lora_a.shape)}"
        )

    aat = lora_a @ lora_a.T
    eye = torch.eye(lora_a.shape[0], device=lora_a.device, dtype=lora_a.dtype)
    pinv_term = lora_a.T @ torch.linalg.pinv(aat + 1e-8 * eye)
    lora_b = grad_matrix @ pinv_term

    scaling = _compute_dynamic_scaling(rank, avg_rank, config)
    scale_rank = float(config.lora_alpha) / scaling if scaling > 0 else float(rank)
    stable_gamma = float(config.gora_stable_gamma)
    if config.gora_scale_by_lr:
        stable_gamma = (float(config.gora_lr) / ((float(rank) / float(module.in_features)) ** 0.5)) * scale_rank
    lora_b = lora_b * (stable_gamma / float(config.lora_alpha))
    return lora_b.to(device=lora_a_weight.device, dtype=lora_a_weight.dtype)


def apply_gora_pinv_init(model, config, init_state, rank_pattern=None):
    import torch

    if not init_state:
        return {}

    applied = {}
    rank_values = [int(rank) for rank in (rank_pattern or {}).values()]
    avg_rank = sum(rank_values) / len(rank_values) if rank_values else config.base_rank

    for peft_name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue

        original_name = next((name for name in init_state if peft_name.endswith(name)), None)
        if original_name is None:
            continue

        adapter_name = _get_active_adapter_name(module)
        lora_a = module.lora_A[adapter_name]
        lora_b = module.lora_B[adapter_name]
        rank = int((rank_pattern or {}).get(original_name, _get_module_rank(module, adapter_name, config.base_rank)))
        init = init_state[original_name]
        if "grad" not in init:
            continue

        with torch.no_grad():
            lora_b.weight.copy_(
                _compute_gora_pinv_b(init["grad"], lora_a.weight, module, config, rank, avg_rank)
            )
        applied[original_name] = {
            "peft_name": peft_name,
            "rank": rank,
            "method": "gora_pinv",
            "gora_stable_gamma": config.gora_stable_gamma,
            "gora_scale_by_lr": config.gora_scale_by_lr,
            "gora_lr": config.gora_lr,
        }

    missing = sorted(set(init_state) - set(applied))
    if missing:
        raise ValueError(f"GoRA pinv init not applied for: {missing[:5]}")

    return applied


def _get_module_rank(module, adapter_name, default_rank):
    rank = getattr(module, "r", None)
    if isinstance(rank, dict):
        return int(rank.get(adapter_name, default_rank))
    if rank is not None:
        return int(rank)
    return int(default_rank)


def _compute_dynamic_scaling(rank, avg_rank, config):
    rank = max(float(rank), 1.0)
    avg_rank = max(float(avg_rank), 1.0)
    if config.scaling_mode == "rank":
        return float(config.lora_alpha) / rank
    if config.scaling_mode == "sqrt_rank":
        return float(config.lora_alpha) / (rank ** 0.5)
    if config.scaling_mode == "avg_rank":
        return float(config.lora_alpha) / ((rank * avg_rank) ** 0.5)
    raise ValueError(f"Unknown scaling_mode: {config.scaling_mode}")


def apply_dynamic_scaling(model, config, rank_pattern=None):
    rank_values = [int(rank) for rank in (rank_pattern or {}).values()]
    avg_rank = sum(rank_values) / len(rank_values) if rank_values else config.base_rank
    applied = {}

    for peft_name, module in model.named_modules():
        if not hasattr(module, "scaling") or not hasattr(module, "lora_A"):
            continue

        original_name = next((name for name in (rank_pattern or {}) if peft_name.endswith(name)), None)
        adapter_name = _get_active_adapter_name(module)
        rank = int((rank_pattern or {}).get(original_name, _get_module_rank(module, adapter_name, config.base_rank)))
        scaling = _compute_dynamic_scaling(rank, avg_rank, config)

        if isinstance(module.scaling, dict):
            module.scaling[adapter_name] = scaling
        else:
            module.scaling = scaling

        applied[original_name or peft_name] = {
            "peft_name": peft_name,
            "rank": rank,
            "scaling": scaling,
            "scaling_mode": config.scaling_mode,
        }

    return applied


def inject_gslora(model, config, target_names, rank_pattern=None, init_state=None):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("GS-LoRA PEFT injection requires the `torch` and `peft` packages.") from exc

    for param in model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        r=config.base_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        target_modules=target_names,
        rank_pattern=rank_pattern or {},
    )
    model = get_peft_model(model, peft_config)
    applied_scaling = apply_dynamic_scaling(model, config, rank_pattern or {})
    if config.init_method == "gora_pinv":
        applied_init = apply_gora_pinv_init(model, config, init_state or {}, rank_pattern or {})
    else:
        applied_init = apply_svd_init(model, init_state or {})
    return model, applied_init, applied_scaling
