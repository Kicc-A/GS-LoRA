from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GSLoraConfig:
    target_modules: List[str]
    target_prefix: Optional[str] = None
    base_rank: int = 8
    tau: float = 0.90
    r_min: int = 2
    r_max: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    calibration_steps: int = 16
    adaptive_rank: bool = True
    bias: str = "none"
    init_method: str = "none"
    init_scale: float = 1e-3
    gora_stable_gamma: float = 5e-2
    gora_scale_by_lr: bool = False
    gora_lr: float = 5e-2
    compensate_scaling: bool = True
    scaling_mode: str = "rank"
    rank_budget_mode: str = "independent"
    param_budget: Optional[int] = None
