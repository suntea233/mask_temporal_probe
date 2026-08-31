from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.statistics import clustered_bootstrap  # noqa: E402

CONDITIONS = ("real", "shuffle", "random", "logit_history")


def load_records() -> list[dict]:
    paths = sorted((PROJECT / "traces/mask_temporal_state_probe/formal").glob("sample_*.json"))
    if len(paths) != 200:
        raise RuntimeError(f"Expected 200 formal samples, found {len(paths)}")
    return [record for path in paths for record in json.loads(path.read_text())["observations"]]


def summarize(records: list[dict], seed: int) -> dict:
    clusters = np.array([r["sample_id"] for r in records])
    result = {"n": len(records), "samples": len(np.unique(clusters))}
    for index, condition in enumerate(CONDITIONS):
        values = np.array([r["conditions"][condition]["delta_logp"] for r in records])
        result[condition] = {
            "mean_delta_logp": float(values.mean()),
            "median_delta_logp": float(np.median(values)),
            "percent_positive": float(100 * (values > 0).mean()),
            "cluster_bootstrap_95_ci": list(clustered_bootstrap(values, clusters, seed=seed + index)),
        }
    return result


def main() -> None:
    records = load_records()
    groups: dict[str, list[dict]] = {
        "vanilla_hit": [r for r in records if r["conditions"]["vanilla"]["hit"]],
        "vanilla_miss": [r for r in records if not r["conditions"]["vanilla"]["hit"]],
        "target_already_top1_at_t_minus_4": [r for r in records if r["top1_already_future_target_by_offset"]["-4"]],
        "target_not_top1_at_t_minus_4": [r for r in records if not r["top1_already_future_target_by_offset"]["-4"]],
        "trajectory_stable_through_reveal": [r for r in records if r["top1_stable"]],
        "trajectory_changes_before_reveal": [r for r in records if not r["top1_stable"]],
    }
    quartile_specs = {
        "vanilla_future_target_logp": lambda r: r["conditions"]["vanilla"]["logp"],
        "vanilla_entropy": lambda r: r["conditions"]["vanilla"]["entropy"],
        "kv_drift": lambda r: r["drift_kv_mean_layer_l2"],
    }
    thresholds = {}
    for name, getter in quartile_specs.items():
        values = np.array([getter(r) for r in records])
        cuts = np.quantile(values, [0.25, 0.5, 0.75])
        thresholds[name] = cuts.tolist()
        for index, (low, high) in enumerate(zip([-np.inf, *cuts], [*cuts, np.inf]), 1):
            groups[f"{name}_q{index}"] = [r for r in records if low < getter(r) <= high]
    summaries = {name: summarize(group, 20261000 + i * 10) for i, (name, group) in enumerate(groups.items())}
    result = {
        "status": "complete",
        "exploratory": True,
        "multiple_comparisons_warning": "Subgroups are descriptive hypothesis generation; no confirmatory claim is based on them.",
        "quartile_thresholds": thresholds,
        "groups": summaries,
        "interpretation": [
            "No difficulty, instability, entropy, or drift subgroup shows a clear positive Real Hidden-History mean effect.",
            "Negative Real effects become larger in high-entropy, changing-top1, and high-drift observations.",
            "Logit History is slightly positive only when the future target was already top-1 at t-4; this motivates a directionality/sharpening control rather than another hidden-memory method.",
        ],
    }
    (PROJECT / "results/subgroup_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["EXPLORATORY SUBGROUP ANALYSIS", "=" * 29, "", *result["interpretation"], "", result["multiple_comparisons_warning"], "", json.dumps(result, indent=2)]
    (PROJECT / "results/subgroup_analysis.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
