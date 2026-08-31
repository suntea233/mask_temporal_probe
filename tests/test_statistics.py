import numpy as np

from src.statistics import clustered_bootstrap
from src.temporal_sampler import _schedule


def test_probe_schedule_respects_four_previous_states():
    assert _schedule(16, 4, (0.25, 0.5, 0.75)) == {5: "25%", 8: "50%", 12: "75%"}


def test_cluster_bootstrap_is_deterministic():
    values = np.array([1.0, 3.0, 5.0])
    clusters = np.array([0, 0, 1])
    assert clustered_bootstrap(values, clusters, seed=9, draws=100) == clustered_bootstrap(values, clusters, seed=9, draws=100)
