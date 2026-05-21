from .config import GSLoraConfig
from .report import count_trainable_params, summarize_ranks
from .targets import find_target_module_names, is_target_module


def calibrate_gslora(model, dataloader, loss_fn, config: GSLoraConfig):
    from .calibration import calibrate_gslora as _calibrate_gslora

    return _calibrate_gslora(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        config=config,
    )


def inject_gslora(model, config, target_names, rank_pattern=None):
    from .peft_injector import inject_gslora as _inject_gslora

    return _inject_gslora(
        model=model,
        config=config,
        target_names=target_names,
        rank_pattern=rank_pattern,
    )


def prepare_gslora_model(model, dataloader, loss_fn, config: GSLoraConfig, device=None):
    if config.adaptive_rank:
        calibration = calibrate_gslora(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            config=config,
        )
        target_names = calibration["target_names"]
        rank_pattern = calibration["rank_pattern"]
        rank_stats = calibration["rank_stats"]
    else:
        target_names = find_target_module_names(
            model,
            config.target_modules,
            config.target_prefix,
        )
        rank_pattern = {}
        rank_stats = {}

    model = inject_gslora(
        model=model,
        config=config,
        target_names=target_names,
        rank_pattern=rank_pattern,
    )
    report = {
        "target_names": target_names,
        "rank_pattern": rank_pattern,
        "rank_stats": rank_stats,
        "rank_summary": summarize_ranks(rank_pattern),
        "params": count_trainable_params(model),
    }
    return model, report


__all__ = [
    "GSLoraConfig",
    "calibrate_gslora",
    "count_trainable_params",
    "find_target_module_names",
    "inject_gslora",
    "is_target_module",
    "prepare_gslora_model",
    "summarize_ranks",
]
