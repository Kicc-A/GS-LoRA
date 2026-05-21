def inject_gslora(model, config, target_names, rank_pattern=None):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("GS-LoRA PEFT injection requires the `peft` package.") from exc

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
    return get_peft_model(model, peft_config)
