from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .config import ProbeConfig
from .directionality_sampler import _cosine, _finite_or_none
from .projection import ProjectedState, ProjectionController
from .temporal_sampler import _seed, get_num_transfer_tokens


KINDS = ("h", "k", "v")
LAYER_BANDS = {
    "00_07": range(0, 8),
    "08_15": range(8, 16),
    "16_23": range(16, 24),
    "24_31": range(24, 32),
}
LARGE_REVERSAL_THRESHOLD = -0.05


def _row(state: ProjectedState, kind: str, layer: int, row: int) -> torch.Tensor:
    return getattr(state, kind)[layer][row].float()


def _safe_displacement(start: torch.Tensor, end: torch.Tensor) -> float | None:
    denominator = float(start.float().norm())
    if denominator == 0:
        return None
    return float((end.float() - start.float()).norm() / denominator)


def _mean_layers(values: list[float | None], layers: range) -> float | None:
    finite = [values[layer] for layer in layers if values[layer] is not None and np.isfinite(values[layer])]
    return float(np.mean(finite)) if finite else None


def _shape_metrics(curve: list[float | None], progress: list[float]) -> dict[str, float | None]:
    valid = [(p, value) for p, value in zip(progress, curve) if value is not None and np.isfinite(value)]
    if not valid:
        return {"endpoint_gain": None, "spearman": None, "backtracking_rate": None, "large_reversal_rate": None}
    p = np.array([item[0] for item in valid]); c = np.array([item[1] for item in valid])
    gain = float(c[np.argmin(abs(p - .75))] - c[np.argmin(abs(p - .25))]) if len(c) >= 2 else None
    rho = float(spearmanr(p, c).statistic) if len(c) >= 3 and np.ptp(c) > 0 else None
    changes = np.diff(c)
    return {
        "endpoint_gain": gain,
        "spearman": rho if rho is not None and np.isfinite(rho) else None,
        "backtracking_rate": float(np.mean(changes < 0)) if len(changes) else None,
        "large_reversal_rate": float(np.mean(changes < LARGE_REVERSAL_THRESHOLD)) if len(changes) else None,
    }


def _prediction_rows(logits: torch.Tensor, block_positions: list[int], reference: torch.Tensor, prompt_length: int) -> list[dict[str, Any]]:
    rows = logits[0, block_positions].float()
    probabilities = F.softmax(rows, dim=-1)
    log_probabilities = F.log_softmax(rows, dim=-1)
    top_probability, top_token = probabilities.max(-1)
    targets = reference[torch.tensor([p - prompt_length for p in block_positions])].long().to(rows.device)
    target_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
    target_logp = log_probabilities.gather(1, targets[:, None]).squeeze(1)
    entropy = -(probabilities * log_probabilities).sum(-1)
    return [{
        "top1_token": int(top_token[row]), "top1_probability": float(top_probability[row]),
        "future_target_probability": float(target_probability[row]), "future_target_logp": float(target_logp[row]),
        "entropy": float(entropy[row]), "hit": bool(top_token[row].cpu() == targets[row].cpu()),
    } for row in range(len(block_positions))]


def _transfer(logits: torch.Tensor, x: torch.Tensor, mask_index: torch.Tensor, block_end: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = torch.argmax(logits, dim=-1)
    probabilities = F.softmax(logits, dim=-1)
    confidence = torch.gather(probabilities, -1, x0.unsqueeze(-1)).squeeze(-1)
    confidence[:, block_end:] = -np.inf
    confidence = torch.where(mask_index, confidence, -np.inf)
    transfer = torch.zeros_like(x, dtype=torch.bool)
    _, indices = torch.topk(confidence[0], k=count)
    transfer[0, indices] = True
    return torch.where(mask_index, x0, x), transfer


def _centered(
    captures: dict[int, ProjectedState], inputs: dict[int, torch.Tensor], step: int,
    row: int, kind: str, layer: int, mask_id: int,
) -> torch.Tensor:
    unresolved = torch.nonzero(inputs[step] == mask_id).flatten().tolist()
    values = getattr(captures[step], kind)[layer].float()
    return values[row] - values[unresolved].mean(0)


def _position_record(
    *, sample_id: int, block: int, row: int, absolute_position: int, prompt_length: int,
    target: int, reveal_step: int, post_step: int | None, shuffle_row: int,
    post_shuffle_row: int | None, captures: dict[int, ProjectedState], inputs: dict[int, torch.Tensor],
    predictions: dict[int, list[dict[str, Any]]], config: ProbeConfig,
) -> dict[str, Any]:
    early_step, pre_step = 1, reveal_step
    lifetime_steps = list(range(early_step, pre_step + 1))
    progress = [0.0 if pre_step == early_step else (step - early_step) / (pre_step - early_step) for step in lifetime_steps]
    endpoint: dict[str, Any] = {}
    trajectory: list[dict[str, Any]] = [{"step": step, "progress": value, "prediction": predictions[step][row], "endpoint_cosine_by_band": {}} for step, value in zip(lifetime_steps, progress)]
    shapes: dict[str, Any] = {}
    for kind in KINDS:
        early_pre, early_pre_shuffle, centered_same, centered_shuffle = [], [], [], []
        pre_post, pre_post_shuffle, early_post, early_post_shuffle = [], [], [], []
        displacement_early_pre, displacement_pre_post = [], []
        per_layer_curves: list[list[float | None]] = [[] for _ in range(32)]
        for layer in range(32):
            early = _row(captures[early_step], kind, layer, row)
            pre = _row(captures[pre_step], kind, layer, row)
            shuffled_pre_step = reveal_step if False else None
            # The shuffle row's own endpoint is supplied through metadata below.
            early_pre.append(_finite_or_none(_cosine(early, pre)))
            displacement_early_pre.append(_safe_displacement(early, pre))
            centered_early = _centered(captures, inputs, early_step, row, kind, layer, config.mask_id)
            centered_pre = _centered(captures, inputs, pre_step, row, kind, layer, config.mask_id)
            centered_same.append(_finite_or_none(_cosine(centered_early, centered_pre)))
            for step in lifetime_steps:
                per_layer_curves[layer].append(_finite_or_none(_cosine(_row(captures[step], kind, layer, row), pre)))
            if post_step is not None:
                post = _row(captures[post_step], kind, layer, row)
                pre_post.append(_finite_or_none(_cosine(pre, post)))
                early_post.append(_finite_or_none(_cosine(early, post)))
                displacement_pre_post.append(_safe_displacement(pre, post))
            else:
                pre_post.append(None); early_post.append(None); displacement_pre_post.append(None)

        # Shuffle endpoints use the other position's own reveal/post times.
        shuffle_absolute = prompt_length + block * config.block_length + shuffle_row
        post_shuffle_absolute = None if post_shuffle_row is None else prompt_length + block * config.block_length + post_shuffle_row
        # These fields are patched by the caller after all reveal times are known.
        endpoint[kind] = {
            "early_pre_same_by_layer": early_pre,
            "early_pre_shuffle_by_layer": early_pre_shuffle,
            "early_pre_centered_same_by_layer": centered_same,
            "early_pre_centered_shuffle_by_layer": centered_shuffle,
            "pre_post_same_by_layer": pre_post,
            "pre_post_shuffle_by_layer": pre_post_shuffle,
            "early_post_same_by_layer": early_post,
            "early_post_shuffle_by_layer": early_post_shuffle,
            "early_pre_displacement_by_layer": displacement_early_pre,
            "pre_post_displacement_by_layer": displacement_pre_post,
        }
        shapes[kind] = {name: [] for name in ("endpoint_gain", "spearman", "backtracking_rate", "large_reversal_rate")}
        for layer in range(32):
            metrics = _shape_metrics(per_layer_curves[layer], progress)
            for name, value in metrics.items():
                shapes[kind][name].append(value)
        for trajectory_row, curve_index in zip(trajectory, range(len(lifetime_steps))):
            trajectory_row["endpoint_cosine_by_band"][kind] = {
                band: _mean_layers([per_layer_curves[layer][curve_index] for layer in range(32)], layers)
                for band, layers in LAYER_BANDS.items()
            }
    return {
        "sample_id": sample_id, "block_index": block, "absolute_position": absolute_position,
        "generation_position": absolute_position - prompt_length, "future_vanilla_target": target,
        "t_early": early_step, "t_pre": pre_step, "t_post": post_step,
        "reveal_step": reveal_step, "unresolved_lifetime": reveal_step,
        "reveal_fraction": reveal_step / (config.steps // (config.gen_length // config.block_length)),
        "shuffle_source_position": shuffle_absolute, "post_shuffle_source_position": post_shuffle_absolute,
        "endpoint_geometry": endpoint, "convergence_shape": shapes, "trajectory": trajectory,
    }


def _patch_shuffle_geometry(
    record: dict[str, Any], *, row: int, shuffle_row: int, post_shuffle_row: int | None,
    reveal_steps: dict[int, int], captures: dict[int, ProjectedState], inputs: dict[int, torch.Tensor], config: ProbeConfig,
) -> None:
    shuffle_pre_step = reveal_steps[shuffle_row]
    shuffle_post_step = shuffle_pre_step + 1 if shuffle_pre_step < config.steps // (config.gen_length // config.block_length) else None
    post_source_step = None if post_shuffle_row is None else reveal_steps[post_shuffle_row] + 1
    for kind in KINDS:
        geometry = record["endpoint_geometry"][kind]
        for layer in range(32):
            early = _row(captures[1], kind, layer, row)
            pre = _row(captures[record["t_pre"]], kind, layer, row)
            shuffled_pre = _row(captures[shuffle_pre_step], kind, layer, shuffle_row)
            geometry["early_pre_shuffle_by_layer"].append(_finite_or_none(_cosine(early, shuffled_pre)))
            centered_early = _centered(captures, inputs, 1, row, kind, layer, config.mask_id)
            centered_shuffled_pre = _centered(captures, inputs, shuffle_pre_step, shuffle_row, kind, layer, config.mask_id)
            geometry["early_pre_centered_shuffle_by_layer"].append(_finite_or_none(_cosine(centered_early, centered_shuffled_pre)))
            if record["t_post"] is not None and post_shuffle_row is not None and post_source_step is not None:
                shuffled_post = _row(captures[post_source_step], kind, layer, post_shuffle_row)
                geometry["pre_post_shuffle_by_layer"].append(_finite_or_none(_cosine(pre, shuffled_post)))
                geometry["early_post_shuffle_by_layer"].append(_finite_or_none(_cosine(early, shuffled_post)))
            else:
                geometry["pre_post_shuffle_by_layer"].append(None)
                geometry["early_post_shuffle_by_layer"].append(None)


@torch.inference_mode()
def endpoint_geometry_generate(
    model, prompt: torch.Tensor, attention_mask: torch.Tensor | None, config: ProbeConfig, *,
    sample_id: int, reference_generated: torch.Tensor, special_token_ids: set[int],
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    if prompt.shape[0] != 1 or config.temperature != 0 or config.cfg_scale != 0:
        raise ValueError("Endpoint geometry probe requires batch=1, temperature=0, cfg=0")
    device = model.device
    prompt_length = prompt.shape[1]
    x = torch.full((1, prompt_length + config.gen_length), config.mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((1, config.gen_length), dtype=attention_mask.dtype, device=device)], -1)
    blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // blocks
    reference_generated = reference_generated.cpu()
    eos_id = getattr(model.config, "eos_token_id", 126081)
    eos_positions = torch.nonzero(reference_generated == eos_id).flatten()
    content_end = int(eos_positions[0]) if eos_positions.numel() else config.gen_length
    controller = ProjectionController(model)
    records: list[dict[str, Any]] = []
    sanity_counts = {"early_mask": True, "pre_mask": True, "post_token": True, "pre_is_reveal_forward": True, "post_is_first_later_forward": True}
    try:
        for block in range(blocks):
            block_start = prompt_length + block * config.block_length
            block_end = block_start + config.block_length
            block_positions = list(range(block_start, block_end))
            transfer_counts = get_num_transfer_tokens(x[:, block_start:block_end] == config.mask_id, steps_per_block)
            captures: dict[int, ProjectedState] = {}
            inputs: dict[int, torch.Tensor] = {}
            predictions: dict[int, list[dict[str, Any]]] = {}
            reveal_steps: dict[int, int] = {}
            for step_zero in range(steps_per_block):
                step = step_zero + 1
                inputs[step] = x[0, block_start:block_end].detach().cpu().clone()
                with controller.mode(block_positions, capture=True):
                    logits = model(x, attention_mask=attention_mask).logits
                captures[step] = controller.state()
                predictions[step] = _prediction_rows(logits, block_positions, reference_generated, prompt_length)
                mask_index = x == config.mask_id
                x0, transfer = _transfer(logits, x, mask_index, block_end, int(transfer_counts[0, step_zero]))
                for absolute in torch.nonzero(transfer[0, block_start:block_end]).flatten().add(block_start).tolist():
                    reveal_steps[absolute - block_start] = step
                x[transfer] = x0[transfer]

            eligible_rows = []
            for row in range(config.block_length):
                gen_position = block * config.block_length + row
                target = int(reference_generated[gen_position])
                if row in reveal_steps and gen_position < content_end and target not in special_token_ids:
                    eligible_rows.append(row)
            post_rows = [row for row in eligible_rows if reveal_steps[row] < steps_per_block]
            for row in eligible_rows:
                reveal = reveal_steps[row]
                post = reveal + 1 if reveal < steps_per_block else None
                post_candidates = [other for other in post_rows if other != row]
                candidates = post_candidates if post is not None and post_candidates else [other for other in eligible_rows if other != row]
                if not candidates:
                    continue
                shuffle_row = candidates[_seed(config.seed, "endpoint-shuffle", sample_id, block, row) % len(candidates)]
                post_shuffle_row = shuffle_row if post is not None and shuffle_row in post_rows else None
                absolute = block_start + row
                target = int(reference_generated[block * config.block_length + row])
                sanity_counts["early_mask"] &= int(inputs[1][row]) == config.mask_id
                sanity_counts["pre_mask"] &= int(inputs[reveal][row]) == config.mask_id
                sanity_counts["pre_is_reveal_forward"] &= reveal_steps[row] == reveal
                if post is not None:
                    sanity_counts["post_token"] &= int(inputs[post][row]) == target
                    sanity_counts["post_is_first_later_forward"] &= int(inputs[post][row]) != config.mask_id and int(inputs[post - 1][row]) == config.mask_id
                record = _position_record(
                    sample_id=sample_id, block=block, row=row, absolute_position=absolute,
                    prompt_length=prompt_length, target=target, reveal_step=reveal, post_step=post,
                    shuffle_row=shuffle_row, post_shuffle_row=post_shuffle_row, captures=captures,
                    inputs=inputs, predictions=predictions, config=config,
                )
                _patch_shuffle_geometry(
                    record, row=row, shuffle_row=shuffle_row, post_shuffle_row=post_shuffle_row,
                    reveal_steps=reveal_steps, captures=captures, inputs=inputs, config=config,
                )
                records.append(record)
    finally:
        controller.close()
    generated = x[0, prompt_length:].detach().cpu()
    sanity = {
        "reference_equals_endpoint_traced": bool(torch.equal(generated, reference_generated)),
        "projection_layers": controller.n_layers, "hidden_definition": "post-block residual stream x after attention, MLP, and residual additions",
        "eligible_positions": len(records), "positions_with_post": sum(r["t_post"] is not None for r in records),
        "shuffle_same_sample_block_different_position": all(r["absolute_position"] != r["shuffle_source_position"] for r in records),
        **sanity_counts,
    }
    return x, records, sanity
