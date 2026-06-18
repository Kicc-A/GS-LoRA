import torch

from gs_lora.init import svd_lora_factors


def test_svd_a_energy_small_b_shapes_and_finiteness():
    out_features = 13
    in_features = 17
    rank = 5

    factors = svd_lora_factors(
        grad=torch.randn(out_features, in_features),
        rank=rank,
        method="svd_a_energy_small_b",
        init_scale=1e-3,
        effective_scaling=1.0,
        energy_beta=0.5,
        small_b_scale=1e-4,
    )

    assert factors["lora_A"].shape == (rank, in_features)
    assert factors["lora_B"].shape == (out_features, rank)
    assert torch.isfinite(factors["lora_A"]).all()
    assert torch.isfinite(factors["lora_B"]).all()
