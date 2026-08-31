import torch

from src.unified_latent_sampler import CANDIDATE_LAYERS, MATURITY_EPSILON, SCENARIO_BATCH_SIZE


def test_unified_probe_constants_are_predeclared():
    assert CANDIDATE_LAYERS == (20, 24, 26, 28, 31)
    assert SCENARIO_BATCH_SIZE == 8
    assert MATURITY_EPSILON > 0


def test_maturity_is_not_clipped():
    entropy, early = 3.0, 2.0
    maturity = 1 - entropy / (early + MATURITY_EPSILON)
    assert maturity < 0


def test_single_row_hidden_replacement_shape():
    current = torch.randn(4096); previous = torch.randn(4096)
    assert torch.stack([previous, current]).shape == (2, 4096)
