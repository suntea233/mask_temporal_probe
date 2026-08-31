import pytest
import torch

from src.common_mode_sampler import _pairwise_cosine_mean
from src.directionality_sampler import _match_direction


def test_energy_decomposition_identity():
    velocities = torch.tensor([[1.0, 2.0], [3.0, -1.0], [-2.0, 4.0], [0.5, 0.25]])
    mean = velocities.mean(0)
    total = (velocities.square()).sum()
    common = len(velocities) * mean.square().sum()
    residual = (velocities - mean).square().sum()
    assert torch.allclose(total, common + residual)


def test_leave_one_out_residual_reconstructs_velocity():
    velocities = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    total = velocities.sum(0)
    for row in range(4):
        common_loo = (total - velocities[row]) / 3
        residual = velocities[row] - common_loo
        assert torch.allclose(common_loo + residual, velocities[row])


def test_shuffled_norm_matching_and_pairwise_cosine():
    source = torch.tensor([1.0, 1.0, 0.0])
    target = torch.tensor([0.0, 0.0, 3.0])
    assert torch.allclose(_match_direction(source, target).norm(), target.norm())
    assert _pairwise_cosine_mean([source, source]) == pytest.approx(1.0)
