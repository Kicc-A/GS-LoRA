try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        class _NoOpProgress:
            def update(self, *unused_args, **unused_kwargs):
                return None

            def close(self):
                return None

        return _NoOpProgress()

import time

from .rank_allocator import allocate_ranks
from .init import build_init_state
from .targets import find_target_module_names


def clear_gradients(model):
    for param in model.parameters():
        param.grad = None


def collect_gradients(model, dataloader, loss_fn, config):
    target_names = find_target_module_names(
        model,
        config.target_modules,
        config.target_prefix,
    )
    target_set = set(target_names)

    for param in model.parameters():
        param.requires_grad = False
    for name, module in model.named_modules():
        if name in target_set:
            module.weight.requires_grad = True

    model.train()
    clear_gradients(model)

    progress = tqdm(total=config.calibration_steps, desc="Calibrating gradients")
    for step, batch in enumerate(dataloader):
        if step >= config.calibration_steps:
            break
        loss = loss_fn(model, batch)
        (loss / config.calibration_steps).backward()
        progress.update(1)
    progress.close()

    grad_cache = {}
    for name, module in model.named_modules():
        if name not in target_set:
            continue
        grad = getattr(module.weight, "grad", None)
        if grad is None:
            continue
        grad_cache[name] = {
            "grad": grad.detach().float().cpu(),
        }

    clear_gradients(model)
    for param in model.parameters():
        param.requires_grad = False

    if not grad_cache:
        raise ValueError("No gradients collected. Check target modules and loss_fn.")

    return target_names, grad_cache


def calibrate_gslora(model, dataloader, loss_fn, config):
    target_names, grad_cache = collect_gradients(model, dataloader, loss_fn, config)
    print(f"Collected gradients for {len(grad_cache)} LoRA target matrices", flush=True)
    if config.adaptive_rank:
        start = time.time()
        print("Allocating adaptive ranks...", flush=True)
        rank_pattern, rank_stats = allocate_ranks(grad_cache, config)
        print(f"Rank allocation finished in {time.time() - start:.1f}s", flush=True)
    else:
        rank_pattern = {name: int(config.base_rank) for name in target_names if name in grad_cache}
        rank_stats = {}
    start = time.time()
    print(f"Building init state with method={config.init_method}...", flush=True)
    init_state = build_init_state(grad_cache, rank_pattern, config)
    print(f"Init state built in {time.time() - start:.1f}s", flush=True)
    return {
        "target_names": target_names,
        "rank_pattern": rank_pattern,
        "rank_stats": rank_stats,
        "init_state": init_state,
    }
