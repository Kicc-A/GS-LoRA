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
    applied_init = apply_svd_init(model, init_state or {})
    return model, applied_init
