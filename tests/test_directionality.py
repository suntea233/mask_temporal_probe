import torch

from src.directionality_sampler import _finite_mean, _match_direction


def test_match_direction_preserves_target_norm_and_source_direction():
    source = torch.tensor([3.0, 4.0])
    target = torch.tensor([0.0, 2.0])
    matched = _match_direction(source, target)
    assert torch.allclose(matched.norm(), target.norm())
    assert torch.dot(matched, source) > 0


def test_match_direction_handles_zero_source():
    assert torch.equal(_match_direction(torch.zeros(3), torch.ones(3)), torch.zeros(3))


def test_finite_mean_ignores_undefined_cosines():
    assert _finite_mean([float("nan"), 0.25, 0.75]) == 0.5
    assert _finite_mean([float("nan")]) is None
