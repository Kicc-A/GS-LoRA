from typing import Iterable, Optional


def is_target_module(
    name: str,
    module,
    target_modules: Iterable[str],
    target_prefix: Optional[str] = None,
) -> bool:
    if target_prefix is not None and not name.startswith(target_prefix):
        return False
    if not any(name == target or name.endswith("." + target) for target in target_modules):
        return False
    weight = getattr(module, "weight", None)
    return weight is not None and getattr(weight, "ndim", 0) >= 2


def find_target_module_names(model, target_modules, target_prefix=None):
    names = []
    for name, module in model.named_modules():
        if is_target_module(name, module, target_modules, target_prefix):
            names.append(name)
    if not names:
        raise ValueError(
            f"No target modules found. Check target_prefix={target_prefix} "
            f"and target_modules={target_modules}."
        )
    return names
