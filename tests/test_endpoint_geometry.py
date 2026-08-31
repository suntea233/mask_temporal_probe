import pytest

from src.endpoint_geometry_sampler import LARGE_REVERSAL_THRESHOLD, _shape_metrics


def test_shape_metrics_monotonic_curve():
    result = _shape_metrics([0.2, 0.4, 0.8, 1.0], [0.0, 1 / 3, 2 / 3, 1.0])
    assert result["spearman"] == pytest.approx(1.0)
    assert result["backtracking_rate"] == 0.0
    assert result["large_reversal_rate"] == 0.0
    assert result["endpoint_gain"] > 0


def test_shape_metrics_backtracking_and_predeclared_reversal():
    result = _shape_metrics([0.5, 0.4, 0.6], [0.0, 0.5, 1.0])
    assert result["backtracking_rate"] == 0.5
    assert result["large_reversal_rate"] == 0.5
    assert LARGE_REVERSAL_THRESHOLD == -0.05
