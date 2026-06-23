from dataclasses import dataclass
from typing import List, Optional, Union


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
    init_scale: Union[float, str] = 1e-3
    init_auto_target_ratio: float = 0.01
    init_auto_scale_min: float = 0.3
    init_auto_scale_max: float = 1.2
    init_energy_beta: float = 0.5
    init_energy_eps: float = 1e-8
    init_small_b_scale: float = 1e-4
    scaling_mode: str = "rank"
    rank_budget_mode: str = "independent"
    param_budget: Optional[int] = None
    rank_score_gamma: Union[float, str] = 1.0
    attention_rank_prior: Union[float, str] = 1.0
