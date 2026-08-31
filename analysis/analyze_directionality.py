from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".mplconfig"))
import matplotlib.pyplot as plt

sys.path.insert(0, str(PROJECT))
from src.directionality_sampler import HIDDEN_CONDITIONS, LOGIT_CONDITIONS  # noqa: E402
from src.statistics import clustered_bootstrap  # noqa: E402

CONDITIONS = HIDDEN_CONDITIONS + LOGIT_CONDITIONS


def load() -> tuple[list[dict], list[dict]]:
    paths = sorted((PROJECT / "traces/directionality_probe/formal").glob("sample_*.json"))
    if len(paths) != 200:
        raise RuntimeError(f"Expected 200 samples, found {len(paths)}")
    samples = [json.loads(path.read_text()) for path in paths]
    if [sample["sample_id"] for sample in samples] != list(range(200)):
        raise RuntimeError("Incomplete or duplicate sample IDs")
    if not all(sample["sanity"]["passed"] for sample in samples):
        raise RuntimeError("At least one follow-up sample failed sanity")
    return samples, [record for sample in samples for record in sample["observations"]]


def metric(values: np.ndarray, clusters: np.ndarray, seed: int) -> dict:
    lo, hi = clustered_bootstrap(values, clusters, seed=seed)
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)), "percent_positive": float(100 * (values > 0).mean()), "cluster_bootstrap_95_ci": [lo, hi]}


def condition_summary(records: list[dict], seed: int) -> dict:
    clusters = np.array([r["sample_id"] for r in records])
    result = {"vanilla": {"hit_rate": float(np.mean([r["conditions"]["vanilla"]["hit"] for r in records]))}}
    for index, condition in enumerate(CONDITIONS):
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
        "forward_last_minus_backward": ("forward_hidden_last", "backward_hidden_mean"),
        "forward_last_minus_shuffled": ("forward_hidden_last", "shuffled_hidden_last"),
        "forward_last_minus_random": ("forward_hidden_last", "random_hidden_last"),
        "forward_last_minus_logit_last": ("forward_hidden_last", "forward_logit_last"),
        "forward_last_minus_sharpen": ("forward_hidden_last", "matched_logit_sharpen"),
        "forward_mean_minus_backward": ("forward_hidden_mean", "backward_hidden_mean"),
        "forward_mean_minus_random": ("forward_hidden_mean", "random_hidden_last"),
        "forward_logit_mean_minus_sharpen": ("forward_logit_mean", "matched_logit_sharpen"),
    }
    result["comparisons"] = {}
    for index, (name, (left, right)) in enumerate(comparisons.items()):
        values = np.array([r["conditions"][left]["delta_logp"] - r["conditions"][right]["delta_logp"] for r in records])
        result["comparisons"][name] = metric(values, clusters, seed + 100 + index)
    return result


def geometry_summary(records: list[dict]) -> dict:
    result = {}
    for kind_index, kind in enumerate(("k", "v")):
        valid = [r for r in records if r["geometry"][f"{kind}_last_velocity_future_cosine"] is not None]
        clusters = np.array([r["sample_id"] for r in valid])
        names = ("mean_velocity_future_cosine", "last_velocity_future_cosine", "shuffled_velocity_future_cosine")
        item = {}
        for index, name in enumerate(names):
            values = np.array([r["geometry"][f"{kind}_{name}"] for r in valid])
            item[name] = metric(values, clusters, 20262000 + kind_index * 20 + index)
        difference = np.array([r["geometry"][f"{kind}_last_velocity_future_cosine"] - r["geometry"][f"{kind}_shuffled_velocity_future_cosine"] for r in valid])
        item["last_minus_shuffled_future_cosine"] = metric(difference, clusters, 20262010 + kind_index)
        item["by_layer"] = {}
        for name in names:
            layer_values = []
            for layer in range(32):
                values = [r["geometry"][f"{kind}_{name}_by_layer"][layer] for r in valid]
                finite = [value for value in values if value is not None]
                layer_values.append(float(np.mean(finite)) if finite else None)
            item["by_layer"][name] = layer_values
        alignment = np.array([r["geometry"][f"{kind}_last_velocity_future_cosine"] for r in valid])
        utility = np.array([r["conditions"]["forward_hidden_last"]["delta_logp"] for r in valid])
        item["alignment_utility_pearson"] = float(np.corrcoef(alignment, utility)[0, 1])
        result[kind] = item
    return result


def decide(overall: dict, geometry: dict) -> dict:
    hidden = overall["forward_hidden_last"]["cluster_bootstrap_95_ci"][0] > 0
    controls = all(overall["comparisons"][name]["cluster_bootstrap_95_ci"][0] > 0 for name in ("forward_last_minus_shuffled", "forward_last_minus_random", "forward_last_minus_logit_last", "forward_last_minus_sharpen"))
    logit = (
        max(overall["forward_logit_mean"]["cluster_bootstrap_95_ci"][0], overall["forward_logit_last"]["cluster_bootstrap_95_ci"][0]) > 0
        and overall["comparisons"]["forward_logit_mean_minus_sharpen"]["cluster_bootstrap_95_ci"][0] > 0
    )
    geometric = any(geometry[k]["last_minus_shuffled_future_cosine"]["cluster_bootstrap_95_ci"][0] > 0 for k in ("k", "v"))
    if hidden and controls:
        return {"code": "D1", "title": "CAUSAL HIDDEN TEMPORAL DIRECTION SIGNAL"}
    if logit:
        return {"code": "D2", "title": "TEMPORAL DIRECTION IS MOSTLY LOGIT-READABLE"}
    if geometric:
        return {"code": "D3", "title": "GEOMETRIC TEMPORAL STRUCTURE WITHOUT CAUSAL UTILITY"}
    return {"code": "D4", "title": "NO TEMPORAL DIRECTION SIGNAL"}


def make_figures(overall: dict, geometry: dict) -> None:
    directory = PROJECT / "figures/directionality"
    directory.mkdir(parents=True, exist_ok=True)
    conditions = ["backward_hidden_mean", "forward_hidden_mean", "forward_hidden_last", "shuffled_hidden_last", "random_hidden_last", "forward_logit_mean", "forward_logit_last", "matched_logit_sharpen"]
    labels = ["Backward H", "Forward mean H", "Forward last H", "Shuffle", "Random", "Forward mean L", "Forward last L", "Sharpen"]
    means = [overall[c]["mean"] for c in conditions]; cis = [overall[c]["cluster_bootstrap_95_ci"] for c in conditions]
    errors = [[m-ci[0] for m,ci in zip(means,cis)],[ci[1]-m for m,ci in zip(means,cis)]]
    fig,ax=plt.subplots(figsize=(10,4.8)); ax.errorbar(range(len(means)),means,yerr=errors,fmt="o",capsize=4); ax.axhline(0,color="black",lw=.8); ax.set_xticks(range(len(labels)),labels,rotation=25,ha="right"); ax.set_ylabel("Delta log P"); fig.tight_layout(); fig.savefig(directory/"01_direction_conditions.png",dpi=180); plt.close(fig)
    bands=["forward_layers_00_07","forward_layers_08_15","forward_layers_16_23","forward_layers_24_31"]
    fig,ax=plt.subplots(figsize=(7,4.5)); ax.bar(["0-7","8-15","16-23","24-31"],[overall[c]["mean"] for c in bands]); ax.axhline(0,color="black",lw=.8); ax.set_ylabel("Forward-last Delta log P"); ax.set_xlabel("Intervened layers"); fig.tight_layout(); fig.savefig(directory/"02_layer_bands.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,kind in zip(axes,("k","v")):
        for name,label in (("last_velocity_future_cosine","same-position"),("shuffled_velocity_future_cosine","shuffled")):
            ax.plot(range(32),geometry[kind]["by_layer"][name],label=label)
        ax.axhline(0,color="black",lw=.8); ax.set_title(kind.upper()); ax.set_xlabel("Layer"); ax.legend()
    axes[0].set_ylabel("Velocity vs reveal cosine"); fig.tight_layout(); fig.savefig(directory/"03_geometry_by_layer.png",dpi=180); plt.close(fig)


def main() -> None:
    samples, records = load()
    overall = condition_summary(records, 20261900)
    by_progress = {label: condition_summary([r for r in records if r["progress"] == label], 20262100 + i * 50) for i,label in enumerate(("25%","50%","75%"))}
    by_vanilla_status = {
        "hit": condition_summary([r for r in records if r["conditions"]["vanilla"]["hit"]], 20262400),
        "miss": condition_summary([r for r in records if not r["conditions"]["vanilla"]["hit"]], 20262500),
    }
    geometry = geometry_summary(records)
    decision = decide(overall, geometry)
    summary = {"status":"complete","samples":len(samples),"observations":len(records),"all_reference_outputs_equal":all(s["sanity"]["reference_equals_followup_traced"] for s in samples),"overall":overall,"by_progress":by_progress,"by_vanilla_status":by_vanilla_status,"geometry":geometry,"decision":decision}
    (PROJECT/"results/directionality_probe_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["TEMPORAL DIRECTIONALITY AND LAYER PROBE","="*39,"",f"DECISION: {decision['code']}. {decision['title']}",f"Samples: {len(samples)}, observations: {len(records)}, trajectory parity: {summary['all_reference_outputs_equal']}","","OVERALL",json.dumps(overall,indent=2),"","BY VANILLA HIT/MISS",json.dumps(by_vanilla_status,indent=2),"","BY PROGRESS",json.dumps(by_progress,indent=2),"","GEOMETRY",json.dumps(geometry,indent=2)]
    (PROJECT/"results/report_directionality_probe.txt").write_text("\n".join(lines)+"\n")
    make_figures(overall,geometry)


if __name__ == "__main__": main()
