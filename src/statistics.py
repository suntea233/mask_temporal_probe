from __future__ import annotations

import numpy as np


def clustered_bootstrap(values: np.ndarray, clusters: np.ndarray, *, seed: int, draws: int = 10000) -> tuple[float, float]:
    """Percentile CI resampling samples, not MASK observations."""
    unique = np.unique(clusters)
    cluster_sums = np.array([values[clusters == c].sum() for c in unique], dtype=np.float64)
    cluster_counts = np.array([(clusters == c).sum() for c in unique], dtype=np.int64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(draws, len(unique)))
    means = cluster_sums[sampled].sum(axis=1) / cluster_counts[sampled].sum(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))
