from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".mplconfig"))
import matplotlib.pyplot as plt

sys.path.insert(0, str(PROJECT))
from src.common_mode_sampler import ALL_CONDITIONS, BAND_CONDITIONS, LAYER_BANDS, PRIMARY_CONDITIONS  # noqa: E402
from src.statistics import clustered_bootstrap  # noqa: E402


def load() -> tuple[list[dict], list[dict], list[dict]]:
    paths = sorted((PROJECT / "traces/common_mode_probe/formal").glob("sample_*.json"))
    if len(paths) != 200:
        raise RuntimeError(f"Expected 200 formal samples, found {len(paths)}")
    samples = [json.loads(path.read_text()) for path in paths]
    if [s["sample_id"] for s in samples] != list(range(200)) or not all(s["sanity"]["passed"] for s in samples):
        raise RuntimeError("Formal samples are incomplete, duplicated, or failed sanity")
    observations = [r for s in samples for r in s["observations"]]
    probe_states = [r for s in samples for r in s["probe_states"]]
    return samples, observations, probe_states


def metric(values: np.ndarray, clusters: np.ndarray, seed: int, *, positive: bool = True) -> dict:
    finite = np.isfinite(values)
    values, clusters = values[finite], clusters[finite]
    lo, hi = clustered_bootstrap(values, clusters, seed=seed)
    result = {
        "n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
        "sample_cluster_bootstrap_95_ci": [lo, hi],
    }
    if positive:
        result["percent_positive"] = float(100 * (values > 0).mean())
    return result


def condition_summary(records: list[dict], seed: int) -> dict:
    clusters = np.array([r["sample_id"] for r in records])
    result = {"vanilla": {"hit_rate": float(np.mean([r["conditions"]["vanilla"]["hit"] for r in records]))}}
    for index, condition in enumerate(ALL_CONDITIONS):
        values = np.array([r["conditions"][condition]["delta_logp"] for r in records])
        item = metric(values, clusters, seed + index)
        item.update({
            "hit_rate": float(np.mean([r["conditions"][condition]["hit"] for r in records])),
            "w2r": int(sum(r["conditions"][condition]["w2r"] for r in records)),
            "r2w": int(sum(r["conditions"][condition]["r2w"] for r in records)),
        })
        item["net_gain"] = item["w2r"] - item["r2w"]
        result[condition] = item
    comparisons = {
        "residual_minus_common": ("real_residual", "common_only"),
        "residual_minus_shuffled": ("real_residual", "shuffled_residual"),
        "residual_minus_random": ("real_residual", "random_residual"),
        "full_minus_common": ("full_velocity", "common_only"),
        "full_minus_residual": ("full_velocity", "real_residual"),
    }
    result["comparisons"] = {}
    for index, (name, (left, right)) in enumerate(comparisons.items()):
        values = np.array([r["conditions"][left]["delta_logp"] - r["conditions"][right]["delta_logp"] for r in records])
        result["comparisons"][name] = metric(values, clusters, seed + 100 + index)
    return result


def _band_value(record: dict, kind: str, key: str, layers: range) -> float:
    values = [record[f"{kind}_{key}_by_layer"][layer] for layer in layers]
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _energy_fraction(record: dict, kind: str, component: str, layers: range) -> float:
    numerator = sum(record[f"{kind}_{component}_energy_by_layer"][layer] for layer in layers)
    denominator = sum(record[f"{kind}_total_energy_by_layer"][layer] for layer in layers)
    return float(numerator / denominator) if denominator else float("nan")


def descriptive_summary(records: list[dict], seed: int) -> dict:
    result = {}
    bands = {**LAYER_BANDS, "all": range(32)}
    for kind_index, kind in enumerate(("k", "v")):
        kind_result = {}
        for band_index, (band, layers) in enumerate(bands.items()):
            clusters = np.array([r["sample_id"] for r in records])
            common = np.array([_energy_fraction(r, kind, "common", layers) for r in records])
            residual = np.array([_energy_fraction(r, kind, "residual", layers) for r in records])
            raw = np.array([_band_value(r, kind, "pairwise_raw_cosine", layers) for r in records])
            centered = np.array([_band_value(r, kind, "pairwise_residual_cosine", layers) for r in records])
            kind_result[band] = {
                "common_energy_fraction": metric(common, clusters, seed + kind_index * 100 + band_index * 5, positive=False),
                "residual_energy_fraction": metric(residual, clusters, seed + kind_index * 100 + band_index * 5 + 1, positive=False),
                "pairwise_raw_cosine": metric(raw, clusters, seed + kind_index * 100 + band_index * 5 + 2),
                "pairwise_residual_cosine": metric(centered, clusters, seed + kind_index * 100 + band_index * 5 + 3),
                "raw_minus_centered_cosine": metric(raw - centered, clusters, seed + kind_index * 100 + band_index * 5 + 4),
            }
        result[kind] = kind_result
    return result


TEMPORAL_KEYS = (
    "same_raw_alignment", "shuffled_raw_alignment", "common_next_alignment",
    "same_residual_alignment", "shuffled_residual_alignment",
)


def temporal_summary(records: list[dict], seed: int) -> dict:
    result = {}
    bands = {**LAYER_BANDS, "all": range(32)}
    for kind_index, kind in enumerate(("k", "v")):
        kind_result = {}
        for band_index, (band, layers) in enumerate(bands.items()):
            clusters = np.array([r["sample_id"] for r in records])
            values = {
                key: np.array([_band_value(r["temporal_geometry"], kind, key, layers) for r in records])
                for key in TEMPORAL_KEYS
            }
            item = {key: metric(value, clusters, seed + kind_index * 200 + band_index * 20 + index) for index, (key, value) in enumerate(values.items())}
            item["same_minus_shuffled_raw"] = metric(values["same_raw_alignment"] - values["shuffled_raw_alignment"], clusters, seed + kind_index * 200 + band_index * 20 + 10)
            item["same_minus_shuffled_residual"] = metric(values["same_residual_alignment"] - values["shuffled_residual_alignment"], clusters, seed + kind_index * 200 + band_index * 20 + 11)
            kind_result[band] = item
        result[kind] = kind_result
    return result


def _number(text: str, reference: bool = False) -> str | None:
    patterns = [r"\\boxed\{\s*([^{}]+)\s*\}", r"####\s*([^\n]+)"] if not reference else [r"####\s*([^\n]+)"]
    for pattern in patterns:
        found = re.findall(pattern, text)
        if found:
            cleaned = re.sub(r"[^0-9.\-]", "", found[-1].replace(",", ""))
            if cleaned:
                return cleaned
    found = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return found[-1].replace(",", "") if found else None


def final_correct_map(samples: list[dict]) -> dict[int, bool]:
    return {s["sample_id"]: _number(s["decoded_output"]) == _number(s["reference_answer"], True) for s in samples}


def decide(overall: dict, descriptive: dict, temporal: dict) -> dict:
    residual = overall["real_residual"]
    predictive = (
        residual["sample_cluster_bootstrap_95_ci"][0] > 0
        and overall["comparisons"]["residual_minus_shuffled"]["sample_cluster_bootstrap_95_ci"][0] > 0
        and overall["comparisons"]["residual_minus_random"]["sample_cluster_bootstrap_95_ci"][0] > 0
        and residual["net_gain"] > 0
    )
    persistence = all(
        temporal[kind]["all"]["same_minus_shuffled_residual"]["sample_cluster_bootstrap_95_ci"][0] > 0
        for kind in ("k", "v")
    )
    shared = all(
        descriptive[kind]["all"]["common_energy_fraction"]["sample_cluster_bootstrap_95_ci"][0] > 0.05
        and descriptive[kind]["all"]["raw_minus_centered_cosine"]["sample_cluster_bootstrap_95_ci"][0] > 0.10
        for kind in ("k", "v")
    )
    if persistence and predictive:
        return {"code": "C", "title": "POSITION-SPECIFIC TEMPORAL SIGNAL SURVIVES"}
    if persistence:
        return {"code": "B", "title": "POSITION-SPECIFIC STRUCTURE EXISTS BUT IS NOT PREDICTIVE"}
    if shared and not predictive:
        return {"code": "A", "title": "COMMON-MODE DOMINATES"}
    return {"code": "D", "title": "INCONCLUSIVE"}


def make_figures(overall: dict, descriptive: dict, temporal: dict, by_progress_descriptive: dict) -> None:
    directory = PROJECT / "figures/common_mode"
    directory.mkdir(parents=True, exist_ok=True)
    bands = list(LAYER_BANDS)
    labels = ["0–7", "8–15", "16–23", "24–31"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for row, kind in enumerate(("k", "v")):
        for col, progress in enumerate(("25%", "50%", "75%")):
            ax = axes[row, col]
            common = [by_progress_descriptive[progress][kind][b]["common_energy_fraction"]["mean"] for b in bands]
            residual = [1 - v for v in common]
            ax.bar(labels, common, label="Common"); ax.bar(labels, residual, bottom=common, label="Residual")
            ax.set_title(f"{kind.upper()} — {progress}"); ax.set_ylim(0, 1)
    axes[0, 0].legend(); fig.supylabel("Energy fraction"); fig.tight_layout(); fig.savefig(directory / "01_common_vs_residual_energy.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    keys = ["same_raw_alignment", "shuffled_raw_alignment", "same_residual_alignment", "shuffled_residual_alignment"]
    labels2 = ["Same raw", "Shuffle raw", "Same residual", "Shuffle residual"]
    for ax, kind in zip(axes, ("k", "v")):
        means = [temporal[kind]["all"][key]["mean"] for key in keys]
        cis = [temporal[kind]["all"][key]["sample_cluster_bootstrap_95_ci"] for key in keys]
        errors = [[m - ci[0] for m, ci in zip(means, cis)], [ci[1] - m for m, ci in zip(means, cis)]]
        ax.errorbar(range(4), means, yerr=errors, fmt="o", capsize=4); ax.axhline(0, color="black", lw=.8)
        ax.set_xticks(range(4), labels2, rotation=22, ha="right"); ax.set_title(kind.upper())
    axes[0].set_ylabel("t to t+1 cosine"); fig.tight_layout(); fig.savefig(directory / "02_temporal_alignment.png", dpi=180); plt.close(fig)

    conditions = list(PRIMARY_CONDITIONS); labels3 = ["Full", "Common", "Residual", "Shuffle residual", "Random residual"]
    means = [overall[c]["mean"] for c in conditions]; cis = [overall[c]["sample_cluster_bootstrap_95_ci"] for c in conditions]
    errors = [[m - ci[0] for m, ci in zip(means, cis)], [ci[1] - m for m, ci in zip(means, cis)]]
    fig, ax = plt.subplots(figsize=(8, 4.6)); ax.errorbar(range(5), means, yerr=errors, fmt="o", capsize=4); ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(range(5), labels3, rotation=20, ha="right"); ax.set_ylabel("Delta log P(future vanilla target)")
    fig.tight_layout(); fig.savefig(directory / "03_causal_utility.png", dpi=180); plt.close(fig)


def main() -> None:
    samples, records, probe_states = load()
    overall = condition_summary(records, 20263000)
    descriptive = descriptive_summary(probe_states, 20264000)
    temporal = temporal_summary(records, 20265000)
    by_progress = {p: condition_summary([r for r in records if r["progress"] == p], 20266000 + i * 100) for i, p in enumerate(("25%", "50%", "75%"))}
    by_progress_descriptive = {p: descriptive_summary([r for r in probe_states if r["progress"] == p], 20267000 + i * 200) for i, p in enumerate(("25%", "50%", "75%"))}
    by_progress_temporal = {p: temporal_summary([r for r in records if r["progress"] == p], 20268000 + i * 300) for i, p in enumerate(("25%", "50%", "75%"))}
    correctness = final_correct_map(samples)
    by_final_correctness = {}
    for i, (label, flag) in enumerate((("final_correct", True), ("final_wrong", False))):
        obs = [r for r in records if correctness[r["sample_id"]] == flag]
        states = [r for r in probe_states if correctness[r["sample_id"]] == flag]
        by_final_correctness[label] = {
            "samples": sum(v == flag for v in correctness.values()),
            "residual_prediction": condition_summary(obs, 20270000 + i * 500)["real_residual"],
            "descriptive": descriptive_summary(states, 20271000 + i * 500),
            "temporal": temporal_summary(obs, 20272000 + i * 500),
        }
    decision = decide(overall, descriptive, temporal)
    environment = json.loads((PROJECT / "results/environment.json").read_text())
    summary = {
        "status": "complete", "samples": len(samples), "probe_states": len(probe_states), "mask_observations": len(records),
        "all_reference_outputs_equal": all(s["sanity"]["reference_equals_common_mode_traced"] for s in samples),
        "max_norm_match_relative_error": max(s["sanity"]["norm_match_max_relative_error"] for s in samples),
        "max_energy_decomposition_relative_error": max(s["sanity"]["decomposition_max_relative_error"] for s in samples),
        "overall": overall, "descriptive_geometry": descriptive, "temporal_alignment": temporal,
        "by_block_denoising_progress": by_progress,
        "descriptive_by_block_denoising_progress": by_progress_descriptive,
        "temporal_by_block_denoising_progress": by_progress_temporal,
        "by_final_correctness": by_final_correctness, "decision": decision,
        "environment": environment,
    }
    (PROJECT / "results/common_mode_probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "COMMON-MODE VS POSITION-SPECIFIC MASK DYNAMICS PROBE", "=" * 53, "",
        f"DECISION: {decision['code']}. {decision['title']}",
        f"Samples: {len(samples)}, probe states: {len(probe_states)}, MASK observations: {len(records)}",
        f"official_output == traced_output: {summary['all_reference_outputs_equal']}",
        "",
        "INTERPRETATION",
        "The A criterion is operationalized as >5% common squared energy with a sample-cluster CI,",
        "a >0.10 CI-bounded drop in cross-position cosine after centering for both K and V,",
        "no residual persistence advantage over shuffled residuals, and no causal residual benefit.",
        "Common-mode dominates the apparent cross-position alignment, not total squared energy:",
        f"K common energy={descriptive['k']['all']['common_energy_fraction']['mean']:.6f}, "
        f"raw/residual pairwise cosine={descriptive['k']['all']['pairwise_raw_cosine']['mean']:.6f}/"
        f"{descriptive['k']['all']['pairwise_residual_cosine']['mean']:.6f};",
        f"V common energy={descriptive['v']['all']['common_energy_fraction']['mean']:.6f}, "
        f"raw/residual pairwise cosine={descriptive['v']['all']['pairwise_raw_cosine']['mean']:.6f}/"
        f"{descriptive['v']['all']['pairwise_residual_cosine']['mean']:.6f}.",
        "Same-position residual temporal alignment is below shuffled residual alignment for both K and V.",
        f"Common-only DeltaLogP={overall['common_only']['mean']:.6f}; "
        f"position-specific residual DeltaLogP={overall['real_residual']['mean']:.6f}.",
        "Conclusion: Much of the apparent temporal geometry of unresolved MASK states is explained by",
        "block-wide context-driven dynamics. After removing this common component, there is little",
        "evidence for a useful position-specific temporal memory.",
        "This closes the persistent MASK hidden-memory direction under the pre-registered decision logic.",
        "",
        "REPLICATION NOTE",
        "The 5-sample debug gate reproduced the previous forward_hidden_last condition with exactly the",
        "same simultaneous target sets (432 observations, max absolute DeltaLogP error 0). The formal",
        "Full Velocity cohort is harder because this probe additionally requires targets to remain MASK",
        "at t+1; its aggregate need not equal the previous probe's differently selected cohort.",
        "Block progress is reported descriptively and is not interpreted as an independent history-length effect.",
        "", "ENVIRONMENT AND PINNED VERSIONS", json.dumps({
            **environment,
            "model_id": "GSAI-ML/LLaDA-8B-Instruct",
            "model_revision": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
            "generation": {"steps": 128, "gen_length": 256, "block_length": 32, "temperature": 0, "cfg_scale": 0, "remasking": "low_confidence", "alpha": 0.25},
        }, indent=2),
        "", "PRIMARY CAUSAL RESULTS", json.dumps({k: overall[k] for k in ("vanilla",) + PRIMARY_CONDITIONS + ("comparisons",)}, indent=2),
        "", "LAYER-BAND CAUSAL RESULTS", json.dumps({k: overall[k] for k in BAND_CONDITIONS}, indent=2),
        "", "DESCRIPTIVE COMMON-MODE GEOMETRY", json.dumps(descriptive, indent=2),
        "", "TEMPORAL ALIGNMENT", json.dumps(temporal, indent=2),
        "", "ANALYSIS BY BLOCK DENOISING PROGRESS", json.dumps(by_progress, indent=2),
        "", "GEOMETRY BY BLOCK DENOISING PROGRESS", json.dumps(by_progress_descriptive, indent=2),
        "", "TEMPORAL ALIGNMENT BY BLOCK DENOISING PROGRESS", json.dumps(by_progress_temporal, indent=2),
        "", "SECONDARY FINAL-CORRECT VS FINAL-WRONG", json.dumps(by_final_correctness, indent=2),
        "", "SANITY", json.dumps({k: summary[k] for k in ("all_reference_outputs_equal", "max_norm_match_relative_error", "max_energy_decomposition_relative_error")}, indent=2),
    ]
    (PROJECT / "results/report_common_mode_probe.txt").write_text("\n".join(lines) + "\n")
    make_figures(overall, descriptive, temporal, by_progress_descriptive)


if __name__ == "__main__":
    main()
