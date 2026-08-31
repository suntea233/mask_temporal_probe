from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".mplconfig"))
import matplotlib.pyplot as plt

sys.path.insert(0, str(PROJECT))
from src.endpoint_geometry_sampler import KINDS, LARGE_REVERSAL_THRESHOLD, LAYER_BANDS  # noqa: E402
from src.statistics import clustered_bootstrap  # noqa: E402

ENDPOINT_KEYS = (
    "early_pre_same", "early_pre_shuffle", "early_pre_same_minus_shuffle",
    "early_pre_centered_same", "early_pre_centered_shuffle", "early_pre_centered_same_minus_shuffle",
    "pre_post_same", "pre_post_shuffle", "pre_post_same_minus_shuffle",
    "early_post_same", "early_post_shuffle", "early_post_same_minus_shuffle",
    "early_pre_displacement", "pre_post_displacement", "pre_post_minus_early_pre_displacement",
)
SHAPE_KEYS = ("endpoint_gain", "spearman", "backtracking_rate", "large_reversal_rate")
BANDS_WITH_ALL = {**LAYER_BANDS, "all": range(32)}


def load() -> tuple[list[dict], list[dict]]:
    paths = sorted((PROJECT / "traces/mask_endpoint_geometry/formal").glob("sample_*.json"))
    if len(paths) != 200:
        raise RuntimeError(f"Expected 200 formal samples, found {len(paths)}")
    samples = [json.loads(path.read_text()) for path in paths]
    if [s["sample_id"] for s in samples] != list(range(200)) or not all(s["sanity"]["passed"] for s in samples):
        raise RuntimeError("Formal endpoint samples incomplete, duplicated, or failed sanity")
    return samples, [record for sample in samples for record in sample["positions"]]


def metric(values: list[float | None] | np.ndarray, clusters: list[int] | np.ndarray, seed: int) -> dict:
    values = np.asarray(values, dtype=float); clusters = np.asarray(clusters)
    valid = np.isfinite(values); values, clusters = values[valid], clusters[valid]
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "percent_positive": None, "sample_cluster_bootstrap_95_ci": [None, None]}
    lo, hi = clustered_bootstrap(values, clusters, seed=seed)
    return {
        "n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
        "percent_positive": float(100 * np.mean(values > 0)),
        "sample_cluster_bootstrap_95_ci": [lo, hi],
    }


def _layer_values(record: dict, kind: str, key: str) -> list[float | None]:
    geometry = record["endpoint_geometry"][kind]
    if key.endswith("_same_minus_shuffle"):
        prefix = key[:-len("_same_minus_shuffle")]
        left = geometry[f"{prefix}_same_by_layer"]; right = geometry[f"{prefix}_shuffle_by_layer"]
        return [None if a is None or b is None else a - b for a, b in zip(left, right)]
    if key == "pre_post_minus_early_pre_displacement":
        left = geometry["pre_post_displacement_by_layer"]; right = geometry["early_pre_displacement_by_layer"]
        return [None if a is None or b is None else a - b for a, b in zip(left, right)]
    return geometry[f"{key}_by_layer"]


def _band_mean(values: list[float | None], layers: range) -> float | None:
    finite = [values[layer] for layer in layers if values[layer] is not None and np.isfinite(values[layer])]
    return float(np.mean(finite)) if finite else None


def endpoint_summary(records: list[dict], seed: int) -> dict:
    result = {}
    clusters = [r["sample_id"] for r in records]
    for kind_index, kind in enumerate(KINDS):
        item = {"per_layer_mean": {}, "bands": {band: {} for band in BANDS_WITH_ALL}}
        for key_index, key in enumerate(ENDPOINT_KEYS):
            rows = [_layer_values(record, kind, key) for record in records]
            item["per_layer_mean"][key] = [
                float(np.mean([row[layer] for row in rows if row[layer] is not None and np.isfinite(row[layer])]))
                for layer in range(32)
            ]
            for band_index, (band, layers) in enumerate(BANDS_WITH_ALL.items()):
                values = [_band_mean(row, layers) for row in rows]
                item["bands"][band][key] = metric(values, clusters, seed + kind_index * 2000 + key_index * 100 + band_index)
        result[kind] = item
    return result


def convergence_summary(records: list[dict], seed: int) -> dict:
    result = {}
    clusters = [r["sample_id"] for r in records]
    for kind_index, kind in enumerate(KINDS):
        item = {"per_layer_mean": {}, "bands": {band: {} for band in BANDS_WITH_ALL}}
        for key_index, key in enumerate(SHAPE_KEYS):
            rows = [record["convergence_shape"][kind][key] for record in records]
            item["per_layer_mean"][key] = [
                float(np.mean([row[layer] for row in rows if row[layer] is not None and np.isfinite(row[layer])]))
                for layer in range(32)
            ]
            for band_index, (band, layers) in enumerate(BANDS_WITH_ALL.items()):
                values = [_band_mean(row, layers) for row in rows]
                summary = metric(values, clusters, seed + kind_index * 1000 + key_index * 100 + band_index)
                if key == "spearman":
                    finite = np.array([v for v in values if v is not None and np.isfinite(v)])
                    summary["fraction_gt_0"] = float(np.mean(finite > 0))
                    summary["fraction_gt_0_5"] = float(np.mean(finite > .5))
                item["bands"][band][key] = summary
        result[kind] = item
    return result


def convergence_curves(records: list[dict], seed: int) -> dict:
    bins = np.linspace(0, 1, 11)
    result: dict[str, Any] = {kind: {band: [] for band in LAYER_BANDS} for kind in KINDS}
    for kind_index, kind in enumerate(KINDS):
        for band_index, band in enumerate(LAYER_BANDS):
            rows = []
            for record in records:
                for trajectory in record["trajectory"]:
                    value = trajectory["endpoint_cosine_by_band"][kind][band]
                    rows.append((record["sample_id"], trajectory["progress"], value))
            for bin_index, center in enumerate(bins):
                if bin_index == 0:
                    selected = [r for r in rows if r[1] <= .05]
                elif bin_index == 10:
                    selected = [r for r in rows if r[1] > .95]
                else:
                    selected = [r for r in rows if center - .05 < r[1] <= center + .05]
                summary = metric([r[2] for r in selected], [r[0] for r in selected], seed + kind_index * 1000 + band_index * 100 + bin_index)
                summary["progress"] = float(center); result[kind][band].append(summary)
    return result


def prediction_geometry(records: list[dict]) -> dict:
    result = {}
    for kind in KINDS:
        item = {}
        for band in LAYER_BANDS:
            cosine, logp, probability, hit = [], [], [], []
            for record in records:
                for row in record["trajectory"]:
                    value = row["endpoint_cosine_by_band"][kind][band]
                    if value is None: continue
                    cosine.append(value); logp.append(row["prediction"]["future_target_logp"])
                    probability.append(row["prediction"]["future_target_probability"]); hit.append(float(row["prediction"]["hit"]))
            c, lp = np.array(cosine), np.array(logp)
            item[band] = {
                "n": len(c), "logp_pearson": float(pearsonr(c, lp).statistic),
                "logp_spearman": float(spearmanr(c, lp).statistic),
                "cosine": cosine, "probability": probability, "hit": hit,
            }
        result[kind] = item
    return result


def _number(text: str, reference: bool = False) -> str | None:
    patterns = [r"\\boxed\{\s*([^{}]+)\s*\}", r"####\s*([^\n]+)"] if not reference else [r"####\s*([^\n]+)"]
    for pattern in patterns:
        found = re.findall(pattern, text)
        if found:
            value = re.sub(r"[^0-9.\-]", "", found[-1].replace(",", ""))
            if value: return value
    found = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return found[-1].replace(",", "") if found else None


def correctness(samples: list[dict]) -> dict[int, bool]:
    return {s["sample_id"]: _number(s["decoded_output"]) == _number(s["reference_answer"], True) for s in samples}


def compact_group(records: list[dict], seed: int) -> dict:
    clusters = [r["sample_id"] for r in records]
    result: dict[str, Any] = {"positions": len(records), "kinds": {}}
    selected_endpoint = (
        "early_pre_same", "pre_post_same", "early_post_same",
        "early_pre_centered_same_minus_shuffle", "early_pre_displacement", "pre_post_displacement",
    )
    for kind_index, kind in enumerate(KINDS):
        endpoint = {}
        for key_index, key in enumerate(selected_endpoint):
            endpoint[key] = metric(
                [_band_mean(_layer_values(r, kind, key), range(32)) for r in records], clusters,
                seed + kind_index * 1000 + key_index,
            )
        shape = {}
        for key_index, key in enumerate(SHAPE_KEYS):
            values = [_band_mean(r["convergence_shape"][kind][key], range(32)) for r in records]
            shape[key] = metric(values, clusters, seed + kind_index * 1000 + 100 + key_index)
        curve = []
        for bin_index, center in enumerate(np.linspace(0, 1, 11)):
            rows = []
            for r in records:
                for trajectory in r["trajectory"]:
                    p = trajectory["progress"]
                    include = p <= .05 if bin_index == 0 else p > .95 if bin_index == 10 else center - .05 < p <= center + .05
                    values = [trajectory["endpoint_cosine_by_band"][kind][band] for band in LAYER_BANDS]
                    finite = [v for v in values if v is not None and np.isfinite(v)]
                    if include and finite: rows.append((r["sample_id"], float(np.mean(finite))))
            item = metric([row[1] for row in rows], [row[0] for row in rows], seed + kind_index * 1000 + 200 + bin_index)
            item["progress"] = float(center); curve.append(item)
        result["kinds"][kind] = {"endpoint": endpoint, "convergence": shape, "curve": curve}
    return result


def decide(endpoint: dict, convergence: dict) -> dict:
    gradual = all(
        convergence[kind]["bands"]["all"]["endpoint_gain"]["sample_cluster_bootstrap_95_ci"][0] > 0
        and convergence[kind]["bands"]["all"]["spearman"]["sample_cluster_bootstrap_95_ci"][0] > 0
        and convergence[kind]["bands"]["all"]["backtracking_rate"]["mean"] < .40
        and endpoint[kind]["bands"]["all"]["early_pre_centered_same_minus_shuffle"]["sample_cluster_bootstrap_95_ci"][0] > 0
        for kind in KINDS
    )
    reveal_jump = all(
        endpoint[kind]["bands"]["all"]["pre_post_minus_early_pre_displacement"]["sample_cluster_bootstrap_95_ci"][0] > 0
        for kind in KINDS
    )
    stable_unresolved = all(endpoint[kind]["bands"]["all"]["early_pre_same"]["mean"] > .90 for kind in KINDS)
    wandering = np.mean([convergence[k]["bands"]["all"]["backtracking_rate"]["mean"] for k in KINDS]) >= .40
    weak_centered = not all(endpoint[k]["bands"]["all"]["early_pre_centered_same_minus_shuffle"]["sample_cluster_bootstrap_95_ci"][0] > 0 for k in KINDS)
    if gradual and reveal_jump:
        return {"code": "D", "title": "TWO-STAGE DYNAMICS"}
    if gradual:
        return {"code": "A", "title": "GRADUAL ENDPOINT CONVERGENCE"}
    if stable_unresolved and reveal_jump:
        return {"code": "C", "title": "DISCRETE REVEAL JUMP DOMINATES"}
    if wandering and weak_centered:
        return {"code": "B", "title": "CONTEXT-DRIVEN RECONSTRUCTION / WANDERING"}
    return {"code": "E", "title": "NO CLEAR GEOMETRIC STRUCTURE"}


def make_figures(endpoint: dict, curves: dict, convergence: dict, prediction: dict) -> None:
    directory = PROJECT / "figures/mask_endpoint_geometry"; directory.mkdir(parents=True, exist_ok=True)
    layers = np.arange(32)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, kind in zip(axes, KINDS):
        p = endpoint[kind]["per_layer_mean"]
        ax.plot(layers, p["early_pre_same"], label="Same"); ax.plot(layers, p["early_pre_shuffle"], label="Shuffle")
        ax.plot(layers, p["early_pre_same_minus_shuffle"], label="Same−Shuffle"); ax.axhline(0, color="black", lw=.7)
        ax.set_title(kind.upper()); ax.set_xlabel("Layer")
    axes[0].set_ylabel("Cosine"); axes[0].legend(); fig.tight_layout(); fig.savefig(directory / "01_early_vs_pre_endpoint.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, kind in zip(axes, KINDS):
        for band in LAYER_BANDS:
            rows = curves[kind][band]; ax.plot([r["progress"] for r in rows], [r["mean"] for r in rows], marker="o", label=band.replace("_", "–"))
        ax.set_title(kind.upper()); ax.set_xlabel("Normalized unresolved progress"); ax.set_ylim(-.1, 1.05)
    axes[0].set_ylabel("Cosine to final pre-reveal endpoint"); axes[0].legend(); fig.tight_layout(); fig.savefig(directory / "02_endpoint_convergence.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, kind in zip(axes, KINDS):
        p = endpoint[kind]["per_layer_mean"]
        for key, label in (("early_pre_same", "Early–Pre"), ("pre_post_same", "Pre–Post"), ("early_post_same", "Early–Post")):
            ax.plot(layers, p[key], label=label)
        ax.set_title(kind.upper()); ax.set_xlabel("Layer")
    axes[0].set_ylabel("Cosine"); axes[0].legend(); fig.tight_layout(); fig.savefig(directory / "03_reveal_transition_geometry.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5)); bands = list(LAYER_BANDS); labels = [b.replace("_", "–") for b in bands]
    for ax, metric_name, title in zip(axes, ("endpoint_gain", "spearman", "backtracking_rate"), ("Endpoint gain", "Monotonicity", "Backtracking")):
        for kind in KINDS:
            ax.plot(labels, [convergence[kind]["bands"][b][metric_name]["mean"] for b in bands], marker="o", label=kind.upper())
        ax.set_title(title); ax.axhline(0, color="black", lw=.7)
    axes[0].legend(); fig.tight_layout(); fig.savefig(directory / "04_convergence_by_layer_band.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for kind in KINDS:
        cosine, probability, hit = [], [], []
        for band in LAYER_BANDS:
            item = prediction[kind][band]; cosine.extend(item.pop("cosine")); probability.extend(item.pop("probability")); hit.extend(item.pop("hit"))
        cosine=np.array(cosine); probability=np.array(probability); hit=np.array(hit); edges=np.quantile(cosine, np.linspace(0,1,11)); centers=[]; probs=[]; hits=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            mask=(cosine>=lo)&(cosine<=hi); centers.append(float(cosine[mask].mean())); probs.append(float(probability[mask].mean())); hits.append(float(hit[mask].mean()))
        axes[0].plot(centers,probs,marker="o",label=kind.upper()); axes[1].plot(centers,hits,marker="o",label=kind.upper())
    axes[0].set_ylabel("P(future vanilla target)"); axes[1].set_ylabel("Future-target hit rate")
    for ax in axes: ax.set_xlabel("Cosine to pre-reveal endpoint"); ax.legend()
    fig.tight_layout(); fig.savefig(directory / "05_geometry_vs_prediction.png", dpi=180); plt.close(fig)


def main() -> None:
    samples, records = load(); endpoint = endpoint_summary(records, 20273000); convergence = convergence_summary(records, 20276000)
    curves = convergence_curves(records, 20279000); prediction = prediction_geometry(records); correct = correctness(samples)
    reveal_groups = {
        "early_0_25": [r for r in records if r["reveal_fraction"] <= .25],
        "mid_25_75": [r for r in records if .25 < r["reveal_fraction"] <= .75],
        "late_75_100": [r for r in records if r["reveal_fraction"] > .75],
    }
    by_reveal = {name: compact_group(group, 20282000 + i * 40000) for i, (name, group) in enumerate(reveal_groups.items())}
    by_correctness = {}
    for i, (name, flag) in enumerate((("final_correct", True), ("final_wrong", False))):
        subset = [r for r in records if correct[r["sample_id"]] == flag]
        by_correctness[name] = {"samples": sum(value == flag for value in correct.values()), **compact_group(subset, 20410000 + i * 40000)}
    decision = decide(endpoint, convergence); environment = json.loads((PROJECT / "results/environment.json").read_text())
    prediction_summary = {
        kind: {band: {key: value for key, value in item.items() if key not in ("cosine", "probability", "hit")}
               for band, item in bands.items()}
        for kind, bands in prediction.items()
    }
    summary = {
        "status": "complete", "samples": len(samples), "blocks": 8 * len(samples), "eligible_positions": len(records),
        "positions_with_valid_post": sum(r["t_post"] is not None for r in records),
        "all_official_outputs_equal_traced": all(s["sanity"]["reference_equals_endpoint_traced"] for s in samples),
        "hidden_definition": samples[0]["sanity"]["hidden_definition"], "large_reversal_threshold": LARGE_REVERSAL_THRESHOLD,
        "endpoint_geometry": endpoint, "convergence_shape": convergence, "convergence_curves": curves,
        "prediction_geometry_correlations": prediction_summary, "by_reveal_time": by_reveal,
        "by_final_correctness": by_correctness, "decision": decision, "environment": environment,
    }
    (PROJECT / "results/mask_endpoint_geometry_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    all_band = lambda kind, source, key: source[kind]["bands"]["all"][key]
    lines = [
        "MASK ENDPOINT GEOMETRY AND REVEAL-TRANSITION PROBE", "=" * 51, "", f"DECISION: {decision['code']}. {decision['title']}",
        f"Samples: {len(samples)}, blocks: {8*len(samples)}, eligible positions: {len(records)}, valid post: {summary['positions_with_valid_post']}",
        f"official_output == traced_output: {summary['all_official_outputs_equal_traced']}",
        f"H definition: {summary['hidden_definition']}", f"Large reversal threshold: {LARGE_REVERSAL_THRESHOLD}", "",
        "KEY OVERALL METRICS",
    ]
    for kind in KINDS:
        e, c = endpoint[kind]["bands"]["all"], convergence[kind]["bands"]["all"]
        lines.extend([
            f"{kind.upper()}: Early-Pre cosine={e['early_pre_same']['mean']:.6f}; Same-Shuffle={e['early_pre_same_minus_shuffle']['mean']:.6f}; centered Same-Shuffle={e['early_pre_centered_same_minus_shuffle']['mean']:.6f}",
            f"{kind.upper()}: Pre-Post cosine={e['pre_post_same']['mean']:.6f}; Early-Post cosine={e['early_post_same']['mean']:.6f}",
            f"{kind.upper()}: Early-Pre displacement={e['early_pre_displacement']['mean']:.6f}; Pre-Post displacement={e['pre_post_displacement']['mean']:.6f}",
            f"{kind.upper()}: EndpointGain={c['endpoint_gain']['mean']:.6f}; Spearman={c['spearman']['mean']:.6f}; backtracking={c['backtracking_rate']['mean']:.6f}; large reversal={c['large_reversal_rate']['mean']:.6f}",
        ])
    lines.extend([
        "", "ENVIRONMENT AND PINNED VERSIONS", json.dumps({**environment, "model_id":"GSAI-ML/LLaDA-8B-Instruct", "model_revision":"08b83a6feb34df1a6011b80c3c00c7563e963b07"}, indent=2),
        "", "ENDPOINT GEOMETRY", json.dumps(endpoint, indent=2), "", "CONVERGENCE SHAPE", json.dumps(convergence, indent=2),
        "", "CONVERGENCE CURVES", json.dumps(curves, indent=2), "", "PREDICTION/GEOMETRY CORRELATIONS", json.dumps(prediction_summary, indent=2),
        "", "REVEAL-TIME STRATIFICATION", json.dumps(by_reveal, indent=2), "", "FINAL-CORRECT VS FINAL-WRONG", json.dumps(by_correctness, indent=2),
    ])
    (PROJECT / "results/report_mask_endpoint_geometry.txt").write_text("\n".join(lines) + "\n")
    make_figures(endpoint, curves, convergence, prediction)


if __name__ == "__main__": main()
