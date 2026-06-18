from .init import SVD_INIT_METHODS

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


def apply_svd_init(model, init_state, method="svd_a_zero_b"):
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
        applied[original_name] = {
            "peft_name": peft_name,
            "method": method,
        }

    missing = sorted(set(init_state) - set(applied))
    if missing:
        raise ValueError(f"SVD init not applied for: {missing[:5]}")

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
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("GS-LoRA PEFT injection requires the `torch` and `peft` packages.") from exc

    if config.init_method not in SVD_INIT_METHODS:
        raise ValueError(f"Unsupported init_method: {config.init_method}")

    for param in model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config.base_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        target_modules=target_names,
        rank_pattern=rank_pattern or {},
    )
    model = get_peft_model(model, peft_config)
    applied_scaling = apply_dynamic_scaling(model, config, rank_pattern or {})
    applied_init = apply_svd_init(model, init_state or {}, method=config.init_method)
    return model, applied_init, applied_scaling
