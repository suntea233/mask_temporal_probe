from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.config import ProbeConfig  # noqa: E402
from src.statistics import clustered_bootstrap  # noqa: E402

CONDITIONS = ("real", "shuffle", "random", "logit_history")


def load() -> tuple[list[dict], list[dict]]:
    paths = sorted((PROJECT / "traces/mask_temporal_state_probe/formal").glob("sample_*.json"))
    if len(paths) != 200:
        raise RuntimeError(f"Refusing incomplete analysis: found {len(paths)}/200 formal sample files")
    samples = [json.loads(path.read_text()) for path in paths]
    if not all(sample["sanity"]["passed"] for sample in samples):
        raise RuntimeError("Refusing analysis because at least one sample failed sanity checks")
    records = [record for sample in samples for record in sample["observations"]]
    if not records:
        raise RuntimeError("No eligible MASK observations")
    return samples, records


def history_bin(value: int) -> str:
    if value <= 5:
        return "4-5"
    if value <= 8:
        return "6-8"
    return "9+"


def metric(values: np.ndarray, clusters: np.ndarray, seed: int) -> dict:
    lo, hi = clustered_bootstrap(values, clusters, seed=seed)
    return {
        "n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
        "percent_positive": float(100 * (values > 0).mean()), "cluster_bootstrap_95_ci": [lo, hi],
    }


def summarize_group(records: list[dict], seed: int) -> dict:
    clusters = np.array([r["sample_id"] for r in records])
    result = {}
    for index, condition in enumerate(CONDITIONS):
        values = np.array([r["conditions"][condition]["delta_logp"] for r in records], dtype=float)
        hits = np.array([r["conditions"][condition]["hit"] for r in records], dtype=bool)
        w2r = sum(r["conditions"][condition]["w2r"] for r in records)
        r2w = sum(r["conditions"][condition]["r2w"] for r in records)
        result[condition] = metric(values, clusters, seed + index)
        result[condition].update({"hit_rate": float(hits.mean()), "w2r": int(w2r), "r2w": int(r2w), "net_gain": int(w2r - r2w)})
    result["vanilla"] = {"hit_rate": float(np.mean([r["conditions"]["vanilla"]["hit"] for r in records]))}
    comparisons = {"real_minus_shuffle": "shuffle", "real_minus_random": "random", "real_minus_logit_history": "logit_history"}
    result["comparisons"] = {}
    for index, (name, other) in enumerate(comparisons.items()):
        values = np.array([r["conditions"]["real"]["delta_logp"] - r["conditions"][other]["delta_logp"] for r in records])
        result["comparisons"][name] = metric(values, clusters, seed + 20 + index)
    return result


def correlations(records: list[dict]) -> dict:
    delta = np.array([r["conditions"]["real"]["delta_logp"] for r in records])
    result = {}
    for key in ("drift_k_mean_layer_l2", "drift_v_mean_layer_l2", "drift_kv_mean_layer_l2"):
        drift = np.array([r[key] for r in records])
        pearson = pearsonr(drift, delta)
        spearman = spearmanr(drift, delta)
        result[key] = {"pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue), "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue)}
    return result


def trajectory(records: list[dict]) -> dict:
    config = ProbeConfig()
    return {
        "top1_already_future_target_rate_by_offset": {
            str(offset): float(np.mean([r["top1_already_future_target_by_offset"][str(offset)] for r in records]))
            for offset in range(-config.history, 1)
        },
        "top1_stable_rate": float(np.mean([r["top1_stable"] for r in records])),
        "top1_stable_confidence_rises_rate": float(np.mean([r["top1_stable_confidence_rises"] for r in records])),
        "top1_changes_before_reveal_rate": float(np.mean([r["top1_change_count"] > 0 for r in records])),
        "mean_top1_change_count": float(np.mean([r["top1_change_count"] for r in records])),
    }


def figures(records: list[dict], overall: dict) -> None:
    figdir = PROJECT / "figures"
    figdir.mkdir(exist_ok=True)
    labels = ["Real hidden", "Shuffled", "Random", "Logit history"]
    means = [overall[c]["mean"] for c in CONDITIONS]
    cis = [overall[c]["cluster_bootstrap_95_ci"] for c in CONDITIONS]
    errors = [[m - ci[0] for m, ci in zip(means, cis)], [ci[1] - m for m, ci in zip(means, cis)]]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.errorbar(range(4), means, yerr=errors, fmt="o", capsize=5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(4), labels, rotation=15)
    ax.set_ylabel("Delta log P(future vanilla target)")
    fig.tight_layout(); fig.savefig(figdir / "01_delta_logp.png", dpi=180); plt.close(fig)

    hit_conditions = ("vanilla",) + CONDITIONS
    hit_labels = ["Vanilla", "Real", "Shuffle", "Random", "Logit hist"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(hit_labels, [overall[c]["hit_rate"] for c in hit_conditions])
    ax.set_ylim(0, 1); ax.set_ylabel("Future-target top-1 hit rate")
    fig.tight_layout(); fig.savefig(figdir / "02_top1_hit_rate.png", dpi=180); plt.close(fig)

    x = np.array([r["drift_kv_mean_layer_l2"] for r in records])
    y = np.array([r["conditions"]["real"]["delta_logp"] for r in records])
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(x, y, s=9, alpha=0.25)
    if len(x) > 1:
        slope, intercept = np.polyfit(x, y, 1); grid = np.linspace(x.min(), x.max(), 100); ax.plot(grid, slope * grid + intercept)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean layer K/V drift L2"); ax.set_ylabel("Real-history Delta log P")
    fig.tight_layout(); fig.savefig(figdir / "03_drift_vs_utility.png", dpi=180); plt.close(fig)


def decide(overall: dict) -> tuple[str, str]:
    positive_real = overall["real"]["cluster_bootstrap_95_ci"][0] > 0
    beats_controls = all(overall["comparisons"][key]["cluster_bootstrap_95_ci"][0] > 0 for key in ("real_minus_shuffle", "real_minus_random"))
    beats_logits = overall["comparisons"]["real_minus_logit_history"]["cluster_bootstrap_95_ci"][0] > 0
    logit_helps = overall["logit_history"]["cluster_bootstrap_95_ci"][0] > 0
    if positive_real and beats_controls and beats_logits:
        return "A", "HIDDEN TEMPORAL STATE CONTAINS EXTRA INFORMATION"
    if logit_helps:
        return "B", "TEMPORAL INFORMATION EXISTS, BUT IS MOSTLY IN LOGITS"
    return "C", "NEGATIVE"


def main() -> None:
    samples, records = load()
    overall = summarize_group(records, 20260827)
    by_progress = {label: summarize_group([r for r in records if r["progress"] == label], 20260830 + i) for i, label in enumerate(("25%", "50%", "75%")) if any(r["progress"] == label for r in records)}
    by_history = {label: summarize_group([r for r in records if history_bin(r["unresolved_steps"]) == label], 20260900 + i) for i, label in enumerate(("4-5", "6-8", "9+")) if any(history_bin(r["unresolved_steps"]) == label for r in records)}
    decision, title = decide(overall)
    config = ProbeConfig()
    conclusion_text = {
        "A": "Unresolved MASK positions contain non-redundant temporal representation information that vanilla dLLM decoding discards between denoising steps.",
        "B": "Temporal prediction history is useful, but there is little evidence that preserving full MASK hidden states provides additional value beyond logits.",
        "C": "This probe provides no evidence that unresolved MASK hidden-state history contains useful predictive information beyond the current denoising state.",
    }[decision]
    summary = {
        "status": "complete", "config": config.as_dict(), "counts": {
            "samples": len(samples),
            "generation_blocks": len(samples) * (config.gen_length // config.block_length),
            "scheduled_probe_states": len(samples) * (config.gen_length // config.block_length) * len(config.progress_fractions),
            "retained_blocks_with_observations": len({(r["sample_id"], r["block_index"]) for r in records}),
            "retained_probe_states_with_observations": len({(r["sample_id"], r["block_index"], r["step_in_block"]) for r in records}),
            "mask_observations": len(records),
        },
        "official_equals_traced_all": all(s["sanity"]["official_equals_traced"] for s in samples),
        "overall": overall, "by_progress": by_progress, "by_unresolved_history": by_history,
        "state_drift_correlations": correlations(records), "prediction_trajectory": trajectory(records),
        "decision": {"code": decision, "title": title, "conclusion": conclusion_text},
        "design_limitations": [
            "With all active-block positions initially masked, probe progress and consecutive unresolved duration are perfectly confounded: 25%=5 steps, 50%=8 steps, and 75%=12 steps. Their grouped tables are descriptive, not independent effects."
        ],
    }
    (PROJECT / "results/mask_temporal_state_probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figures(records, overall)
    lines = [
        "UNRESOLVED-MASK TEMPORAL STATE PROBE", "=" * 43, "", "STATUS: COMPLETE",
        f"FINAL DECISION: {decision}. {title}", f"CONCLUSION: {conclusion_text}", "",
        "EXPERIMENT CONFIG", json.dumps(summary["config"], indent=2), "",
        f"Counts: {summary['counts']}", f"Official == traced for all 200 samples: {summary['official_equals_traced_all']}", "",
        "DECISION RATIONALE",
        f"Real vs Vanilla mean Delta LogP: {overall['real']['mean']:.8f}, 95% CI {overall['real']['cluster_bootstrap_95_ci']}",
        f"Real minus Shuffle: {overall['comparisons']['real_minus_shuffle']['mean']:.8f}, 95% CI {overall['comparisons']['real_minus_shuffle']['cluster_bootstrap_95_ci']}",
        f"Real minus Random: {overall['comparisons']['real_minus_random']['mean']:.8f}, 95% CI {overall['comparisons']['real_minus_random']['cluster_bootstrap_95_ci']}",
        f"Real minus Logit History: {overall['comparisons']['real_minus_logit_history']['mean']:.8f}, 95% CI {overall['comparisons']['real_minus_logit_history']['cluster_bootstrap_95_ci']}",
        f"Logit History vs Vanilla mean Delta LogP: {overall['logit_history']['mean']:.8f}, 95% CI {overall['logit_history']['cluster_bootstrap_95_ci']}", "",
        "DESIGN LIMITATION", summary["design_limitations"][0], "",
        "OVERALL METRICS", json.dumps(overall, indent=2), "", "25% / 50% / 75%", json.dumps(by_progress, indent=2), "",
        "UNRESOLVED HISTORY LENGTH", json.dumps(by_history, indent=2), "", "STATE DRIFT CORRELATIONS",
        json.dumps(summary["state_drift_correlations"], indent=2), "", "PREDICTION TRAJECTORY", json.dumps(summary["prediction_trajectory"], indent=2),
    ]
    environment = PROJECT / "results/environment.json"
    if environment.exists():
        lines[5:5] = ["", "ENVIRONMENT", environment.read_text().strip()]
    (PROJECT / "results/report_mask_temporal_state_probe.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
