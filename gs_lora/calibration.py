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

import math
import time

import torch

from .rank_allocator import allocate_ranks
from .init import build_init_state
from .targets import find_target_module_names


def clear_gradients(model):
    for param in model.parameters():
        param.grad = None


def _set_trainable_targets(model, target_set):
    for param in model.parameters():
        param.requires_grad = False
    for name, module in model.named_modules():
        if name in target_set:
            module.weight.requires_grad = True


def _average_distributed_gradient(grad_fp32):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(grad_fp32, op=torch.distributed.ReduceOp.SUM)
        grad_fp32.div_(torch.distributed.get_world_size())
    return grad_fp32


def capture_target_gradients(model, target_names, divisor=1):
    grad_cache = {}
    target_set = set(target_names)
    scale = float(max(int(divisor), 1))
    for name, module in model.named_modules():
        if name not in target_set:
            continue
        grad = getattr(module.weight, "grad", None)
        if grad is None:
            continue
        grad_fp32 = grad.detach().float().clone()
        grad_fp32 = _average_distributed_gradient(grad_fp32)
        grad_cache[name] = {"grad": grad_fp32.div(scale).cpu()}
    return grad_cache


def _validate_stable_budget_config(config):
    calibration_blocks = int(getattr(config, "calibration_blocks", 1))
    if calibration_blocks < 1:
        raise ValueError("calibration_blocks must be >= 1")
    stable_min = float(getattr(config, "stable_score_min", 0.5))
    stable_max = float(getattr(config, "stable_score_max", 1.5))
    if stable_min <= 0 or stable_max <= 0 or stable_min > stable_max:
        raise ValueError("expected 0 < stable_score_min <= stable_score_max")


def _stable_budget_enabled(config):
    return (
        getattr(config, "rank_budget_mode", "independent") == "param"
        and int(getattr(config, "calibration_blocks", 1)) > 1
        and bool(getattr(config, "use_stable_budget_score", False))
    )


def collect_gradients(model, dataloader, loss_fn, config):
    target_names = find_target_module_names(
        model,
        config.target_modules,
        config.target_prefix,
    )
    target_set = set(target_names)

    _set_trainable_targets(model, target_set)
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
    grad_cache = capture_target_gradients(model, target_names, divisor=1)

    clear_gradients(model)
    for param in model.parameters():
        param.requires_grad = False

    if not grad_cache:
        raise ValueError("No gradients collected. Check target modules and loss_fn.")

    return target_names, grad_cache, [], None


def _add_grad_sums(total_sums, grad_sums):
    for name, item in grad_sums.items():
        grad = item["grad"]
        if name not in total_sums:
            total_sums[name] = {"grad": grad.clone()}
        else:
            total_sums[name]["grad"].add_(grad)


def _divide_grad_cache(grad_cache, divisor):
    scale = float(max(int(divisor), 1))
    return {
        name: {
            "grad": item["grad"].div(scale),
        }
        for name, item in grad_cache.items()
    }


def collect_stable_budget_gradients(model, dataloader, loss_fn, config):
    _validate_stable_budget_config(config)
    target_names = find_target_module_names(
        model,
        config.target_modules,
        config.target_prefix,
    )
    target_set = set(target_names)

    _set_trainable_targets(model, target_set)
    model.train()
    clear_gradients(model)

    calibration_steps = int(config.calibration_steps)
    calibration_blocks = int(getattr(config, "calibration_blocks", 1))
    block_size = max(1, math.ceil(max(calibration_steps, 1) / calibration_blocks))
    total_sums = {}
    block_grad_caches = []
    processed_steps = 0
    current_block_steps = 0

    def flush_block():
        nonlocal current_block_steps
        if current_block_steps <= 0:
            return
        block_sums = capture_target_gradients(model, target_names, divisor=1)
        _add_grad_sums(total_sums, block_sums)
        block_grad_caches.append(_divide_grad_cache(block_sums, current_block_steps))
        clear_gradients(model)
        current_block_steps = 0

    progress = tqdm(total=config.calibration_steps, desc="Calibrating gradients")
    for step, batch in enumerate(dataloader):
        if step >= calibration_steps:
            break
        if current_block_steps >= block_size and len(block_grad_caches) < calibration_blocks - 1:
            flush_block()
        loss = loss_fn(model, batch)
        loss.backward()
        processed_steps += 1
        current_block_steps += 1
        progress.update(1)
    progress.close()
    flush_block()

    grad_cache = _divide_grad_cache(total_sums, processed_steps) if processed_steps > 0 else {}

    clear_gradients(model)
    for param in model.parameters():
        param.requires_grad = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not grad_cache:
        raise ValueError("No gradients collected. Check target modules and loss_fn.")

    return target_names, grad_cache, block_grad_caches, processed_steps


def _svd_energy(grad):
    flat_grad = grad.detach().float()
    if flat_grad.ndim > 2:
        flat_grad = flat_grad.reshape(flat_grad.shape[0], -1)
    device = flat_grad.device
    if not flat_grad.is_cuda and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    flat_grad = flat_grad.to(device, non_blocking=True)
    return torch.linalg.svdvals(flat_grad).pow(2).cpu()


def build_stable_candidate_energy(block_grad_caches, config):
    eps = float(getattr(config, "stable_score_eps", 1e-8))
    stable_min = float(getattr(config, "stable_score_min", 0.5))
    stable_max = float(getattr(config, "stable_score_max", 1.5))
    candidate_energy = {}
    module_names = sorted({name for block in block_grad_caches for name in block})
    for name in module_names:
        energies = [_svd_energy(block[name]["grad"]) for block in block_grad_caches if name in block]
        if not energies:
            continue
        stacked = torch.stack(energies, dim=0)
        mean_energy = stacked.mean(dim=0)
        if stacked.shape[0] <= 1:
            stability = torch.ones_like(mean_energy)
        else:
            std_energy = stacked.std(dim=0, unbiased=False)
            raw_stability = mean_energy / (std_energy + eps)
            stability = raw_stability.clamp(stable_min, stable_max)
        candidate_energy[name] = mean_energy * stability
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return candidate_energy


def calibrate_gslora(model, dataloader, loss_fn, config):
    candidate_energy = None
    if _stable_budget_enabled(config):
        target_names, grad_cache, block_grad_caches, processed_steps = collect_stable_budget_gradients(
            model, dataloader, loss_fn, config
        )
        print(
            f"Collected gradients for {len(grad_cache)} LoRA target matrices "
            f"from {processed_steps} calibration batches",
            flush=True,
        )
        start = time.time()
        print(
            f"Building stable budget scores from {len(block_grad_caches)} calibration blocks...",
            flush=True,
        )
        candidate_energy = build_stable_candidate_energy(block_grad_caches, config)
        block_grad_caches = []
        print(f"Stable budget scores built in {time.time() - start:.1f}s", flush=True)
    else:
        target_names, grad_cache, _, _ = collect_gradients(model, dataloader, loss_fn, config)
        print(f"Collected gradients for {len(grad_cache)} LoRA target matrices", flush=True)

    if config.adaptive_rank:
        start = time.time()
        print("Allocating adaptive ranks...", flush=True)
        rank_pattern, rank_stats = allocate_ranks(grad_cache, config, candidate_energy=candidate_energy)
        print(f"Rank allocation finished in {time.time() - start:.1f}s", flush=True)
        summary = rank_stats.get("__summary__") if isinstance(rank_stats, dict) else None
        if summary:
            print(f"Rank report summary: {summary}", flush=True)
    else:
        rank_pattern = {name: int(config.base_rank) for name in target_names if name in grad_cache}
        rank_stats = {}
    candidate_energy = None

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
