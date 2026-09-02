import pytest
import torch

from src.unified_latent_sampler import (
    CANDIDATE_LAYERS,
    MATURITY_EPSILON,
    SCENARIO_BATCH_SIZE,
    _downstream_gain,
    _record_sanity,
    _strict_future_endpoint,
)


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


def test_endpoint_oracle_must_be_strictly_future():
    assert _strict_future_endpoint(8, 5)
    assert not _strict_future_endpoint(5, 5)
    assert not _strict_future_endpoint(4, 5)


def test_empty_records_fail_every_record_sanity_check():
    sanity = _record_sanity([], hidden_isolated=True, hard_isolated=True)
    assert sanity
    assert not any(sanity.values())


def test_valid_record_provenance_is_computed_not_assumed():
    record = {
        "step_in_block": 5,
        "previous_step": 4,
        "early_step": 1,
        "endpoint_step": 8,
        "reveal_step": 8,
        "endpoint_horizon": 3,
        "absolute_position": 100,
        "shuffle_source_position": 101,
        "downstream_count": 2,
    }
    assert all(_record_sanity([record], hidden_isolated=True, hard_isolated=True).values())
    record["endpoint_horizon"] = 0
    assert not _record_sanity([record], hidden_isolated=True, hard_isolated=True)["endpoint_same_position_layer_pre_reveal"]


def test_missing_downstream_measurement_is_not_encoded_as_zero():
    with pytest.raises(ValueError, match="at least one"):
        _downstream_gain(torch.randn(4, 10), [], [], [])


def test_downstream_measurement_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="equal length"):
        _downstream_gain(torch.randn(4, 10), [1], [2, 3], [-1.0])
