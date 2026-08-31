from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import ProbeConfig
from .projection import ProjectionController, matched_random
from .temporal_sampler import (
    StepState,
    _counterfactual,
    _mean_previous,
    _schedule,
    _seed,
    _split_state,
    _summarize_logits,
    get_num_transfer_tokens,
)


HIDDEN_CONDITIONS = (
    "backward_hidden_mean",
    "forward_hidden_mean",
    "forward_hidden_last",
    "shuffled_hidden_last",
    "random_hidden_last",
    "forward_layers_00_07",
    "forward_layers_08_15",
    "forward_layers_16_23",
    "forward_layers_24_31",
)
LOGIT_CONDITIONS = (
    "backward_logit_mean",
    "forward_logit_mean",
    "forward_logit_last",
    "matched_logit_sharpen",
)
LAYER_BANDS = {
    "forward_layers_00_07": range(0, 8),
    "forward_layers_08_15": range(8, 16),
    "forward_layers_16_23": range(16, 24),
    "forward_layers_24_31": range(24, 32),
}


def _match_direction(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source = source.float()
    source_norm = source.norm()
    target_norm = target.float().norm()
    if source_norm == 0:
        return torch.zeros_like(source)
    return source * (target_norm / source_norm)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left, right = left.float(), right.float()
    denominator = left.norm() * right.norm()
    if denominator == 0:
        return float("nan")
    return float(torch.dot(left.flatten(), right.flatten()) / denominator)


def _finite_or_none(value: float) -> float | None:
    return value if np.isfinite(value) else None


def _finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _condition_result(logits: torch.Tensor, target: int, vanilla: dict[str, Any]) -> dict[str, Any]:
    result = _summarize_logits(logits, target)
    result["delta_logp"] = result["logp"] - vanilla["logp"]
    result["w2r"] = bool(not vanilla["hit"] and result["hit"])
    result["r2w"] = bool(vanilla["hit"] and not result["hit"])
    return result


@torch.inference_mode()
def directionality_generate(
    model,
    prompt: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: ProbeConfig,
    *,
    sample_id: int,
    reference_generated: torch.Tensor,
    special_token_ids: set[int],
    verify_vanilla_probe: bool,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    if prompt.shape[0] != 1 or config.cfg_scale != 0 or config.temperature != 0:
        raise ValueError("Fixed follow-up requires batch=1, cfg=0, temperature=0")
    if len(reference_generated) != config.gen_length:
        raise ValueError("Reference generation length mismatch")
    device = model.device
    x = torch.full((1, prompt.shape[1] + config.gen_length), config.mask_id, dtype=torch.long, device=device)
    x[:, : prompt.shape[1]] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, config.gen_length), dtype=attention_mask.dtype, device=device),
        ], dim=-1)
    blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // blocks
    schedule = _schedule(steps_per_block, config.history, config.progress_fractions)
    reference_generated = reference_generated.cpu()
    eos_id = getattr(model.config, "eos_token_id", 126081)
    eos_positions = torch.nonzero(reference_generated == eos_id).flatten()
    content_end = int(eos_positions[0]) if eos_positions.numel() else config.gen_length

    controller = ProjectionController(model)
    records: list[dict[str, Any]] = []
    geometry_pending: list[dict[str, Any]] = []
    max_vanilla_error = 0.0
    max_norm_error = 0.0
    max_norm_relative_error = 0.0
    try:
        if controller.n_layers != 32:
            raise RuntimeError(f"Expected 32 layers, found {controller.n_layers}")
        for block in range(blocks):
            block_start = prompt.shape[1] + block * config.block_length
            block_end = block_start + config.block_length
            transfer_counts = get_num_transfer_tokens(x[:, block_start:block_end] == config.mask_id, steps_per_block)
            histories: dict[int, deque[StepState]] = defaultdict(lambda: deque(maxlen=config.history + 1))
            logit_histories: dict[int, deque[torch.Tensor]] = defaultdict(lambda: deque(maxlen=config.history + 1))
            unresolved = defaultdict(int)
            for step_zero in range(steps_per_block):
                step = step_zero + 1
                mask_index = x == config.mask_id
                active = [int(p) for p in torch.nonzero(mask_index[0, block_start:block_end]).flatten().add(block_start)]
                with controller.mode(active, capture=True):
                    logits = model(x, attention_mask=attention_mask).logits
                states = _split_state(controller.state(), active)
                for pending in geometry_pending:
                    position = pending["position"]
                    if pending["block_index"] == block and position in states:
                        pending["reveal_state"] = states[position]
                        pending["reveal_step"] = step
                for position in active:
                    unresolved[position] += 1
                    histories[position].append(states[position])
                    logit_histories[position].append(logits[0, position].detach().float().cpu().clone())

                eligible = [p for p in active if len(histories[p]) == config.history + 1]
                if step in schedule and len(eligible) >= 2:
                    eligible.sort(key=lambda p: (-unresolved[p], p))
                    selected = eligible[: config.n_mask]
                    shuffle_source = {p: eligible[(eligible.index(p) + 1) % len(eligible)] for p in selected}
                    replacements: dict[str, dict[str, dict[int, torch.Tensor]]] = {
                        name: {"k": {}, "v": {}} for name in HIDDEN_CONDITIONS
                    }
                    geometry_inputs: dict[int, dict[str, StepState]] = {}
                    for layer in range(controller.n_layers):
                        per_condition_k = {name: [] for name in HIDDEN_CONDITIONS}
                        per_condition_v = {name: [] for name in HIDDEN_CONDITIONS}
                        for position in selected:
                            current = histories[position][-1]
                            previous = histories[position][-2]
                            source = shuffle_source[position]
                            source_current = histories[source][-1]
                            source_previous = histories[source][-2]
                            if position not in geometry_inputs:
                                geometry_inputs[position] = {
                                    "current": current,
                                    "previous": previous,
                                    "history_mean": StepState(k={}, v={}),
                                    "shuffled_last": StepState(k={}, v={}),
                                }
                            for kind, destination in (("k", per_condition_k), ("v", per_condition_v)):
                                current_value = getattr(current, kind)[layer].float()
                                previous_value = getattr(previous, kind)[layer].float()
                                history_mean = _mean_previous(histories[position], kind, layer, config.history)
                                mean_velocity = current_value - history_mean
                                last_velocity = current_value - previous_value
                                source_last = getattr(source_current, kind)[layer].float() - getattr(source_previous, kind)[layer].float()
                                shuffled_last = _match_direction(source_last, last_velocity)
                                random_last = matched_random(
                                    last_velocity.unsqueeze(0),
                                    _seed(config.seed, "direction", sample_id, block, step, position, layer, kind),
                                )[0].float()
                                target_norm = float(last_velocity.norm())
                                shuffled_error = abs(float(shuffled_last.norm()) - target_norm)
                                random_error = abs(float(random_last.norm()) - target_norm)
                                max_norm_error = max(max_norm_error, shuffled_error, random_error)
                                denominator = max(target_norm, torch.finfo(torch.float32).tiny)
                                max_norm_relative_error = max(
                                    max_norm_relative_error,
                                    shuffled_error / denominator,
                                    random_error / denominator,
                                )
                                destination["backward_hidden_mean"].append(current_value - config.alpha * mean_velocity)
                                destination["forward_hidden_mean"].append(current_value + config.alpha * mean_velocity)
                                destination["forward_hidden_last"].append(current_value + config.alpha * last_velocity)
                                destination["shuffled_hidden_last"].append(current_value + config.alpha * shuffled_last)
                                destination["random_hidden_last"].append(current_value + config.alpha * random_last)
                                for band_name, layers in LAYER_BANDS.items():
                                    if layer in layers:
                                        destination[band_name].append(current_value + config.alpha * last_velocity)
                                getattr(geometry_inputs[position]["history_mean"], kind)[layer] = history_mean.to(current_value.dtype)
                                getattr(geometry_inputs[position]["shuffled_last"], kind)[layer] = shuffled_last.to(current_value.dtype)
                        for name in HIDDEN_CONDITIONS:
                            if per_condition_k[name]:
                                replacements[name]["k"][layer] = torch.stack(per_condition_k[name])
                                replacements[name]["v"][layer] = torch.stack(per_condition_v[name])

                    vanilla_rows = logits[0, selected].detach().float().cpu()
                    if verify_vanilla_probe:
                        vanilla_probe = _counterfactual(model, controller, x, attention_mask, selected)
                        max_vanilla_error = max(max_vanilla_error, float((vanilla_probe - vanilla_rows).abs().max()))
                    hidden_logits = {
                        name: _counterfactual(
                            model, controller, x, attention_mask, selected,
                            replacements[name]["k"], replacements[name]["v"],
                        ) for name in HIDDEN_CONDITIONS
                    }
                    logit_rows: dict[str, torch.Tensor] = {name: [] for name in LOGIT_CONDITIONS}  # type: ignore[assignment]
                    for position in selected:
                        history = list(logit_histories[position])
                        current_logits = history[-1]
                        mean_previous = torch.stack(history[:-1]).mean(0)
                        mean_velocity = current_logits - mean_previous
                        last_velocity = current_logits - history[-2]
                        centered_velocity = last_velocity - last_velocity.mean()
                        sharpen_direction = current_logits - current_logits.mean()
                        sharpen_direction = _match_direction(sharpen_direction, centered_velocity)
                        values = {
                            "backward_logit_mean": current_logits - config.alpha * mean_velocity,
                            "forward_logit_mean": current_logits + config.alpha * mean_velocity,
                            "forward_logit_last": current_logits + config.alpha * last_velocity,
                            "matched_logit_sharpen": current_logits + config.alpha * sharpen_direction,
                        }
                        for name, value in values.items():
                            logit_rows[name].append(value)  # type: ignore[union-attr]
                    logit_rows = {name: torch.stack(value) for name, value in logit_rows.items()}  # type: ignore[arg-type,assignment]

                    for row, position in enumerate(selected):
                        gen_position = position - prompt.shape[1]
                        target = int(reference_generated[gen_position])
                        if gen_position >= content_end or target in special_token_ids:
                            continue
                        vanilla = _summarize_logits(vanilla_rows[row], target)
                        conditions = {"vanilla": vanilla}
                        for name in HIDDEN_CONDITIONS:
                            conditions[name] = _condition_result(hidden_logits[name][row], target, vanilla)
                        for name in LOGIT_CONDITIONS:
                            conditions[name] = _condition_result(logit_rows[name][row], target, vanilla)
                        record = {
                            "sample_id": sample_id,
                            "block_index": block,
                            "step_in_block": step,
                            "progress": schedule[step],
                            "absolute_position": position,
                            "generation_position": gen_position,
                            "unresolved_steps": unresolved[position],
                            "shuffle_source_position": shuffle_source[position],
                            "future_vanilla_target": target,
                            "conditions": conditions,
                        }
                        records.append(record)
                        geometry_pending.append({
                            "record": record,
                            "position": position,
                            "block_index": block,
                            "probe_step": step,
                            "reveal_step": step,
                            "current": geometry_inputs[position]["current"],
                            "previous": geometry_inputs[position]["previous"],
                            "history_mean": geometry_inputs[position]["history_mean"],
                            "shuffled_last": geometry_inputs[position]["shuffled_last"],
                            "reveal_state": geometry_inputs[position]["current"],
                        })

                x0 = torch.argmax(logits, dim=-1)
                probabilities = F.softmax(logits, dim=-1)
                x0_p = torch.gather(probabilities, -1, x0.unsqueeze(-1)).squeeze(-1)
                x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                count = int(transfer_counts[0, step_zero])
                transfer = torch.zeros_like(x0, dtype=torch.bool)
                _, indices = torch.topk(confidence[0], k=count)
                transfer[0, indices] = True
                x[transfer] = x0[transfer]
    finally:
        controller.close()

    for pending in geometry_pending:
        current, previous = pending["current"], pending["previous"]
        history_mean, reveal = pending["history_mean"], pending["reveal_state"]
        shuffled_last = pending["shuffled_last"]
        geometry: dict[str, Any] = {
            "reveal_step_in_block": pending["reveal_step"],
            "steps_probe_to_reveal": pending["reveal_step"] - pending["probe_step"],
        }
        for kind in ("k", "v"):
            mean_cos, last_cos, shuffle_cos, shuffled_future_cos = [], [], [], []
            past_norm, last_norm, future_norm = [], [], []
            for layer in range(controller.n_layers):
                cur = getattr(current, kind)[layer].float()
                mean_velocity = cur - getattr(history_mean, kind)[layer].float()
                last_velocity = cur - getattr(previous, kind)[layer].float()
                future_velocity = getattr(reveal, kind)[layer].float() - cur
                shuffled_velocity = getattr(shuffled_last, kind)[layer].float()
                mean_cos.append(_cosine(mean_velocity, future_velocity))
                last_cos.append(_cosine(last_velocity, future_velocity))
                shuffle_cos.append(_cosine(last_velocity, shuffled_velocity))
                shuffled_future_cos.append(_cosine(shuffled_velocity, future_velocity))
                past_norm.append(float(mean_velocity.norm()))
                last_norm.append(float(last_velocity.norm()))
                future_norm.append(float(future_velocity.norm()))
            geometry.update({
                f"{kind}_mean_velocity_future_cosine_by_layer": [_finite_or_none(v) for v in mean_cos],
                f"{kind}_last_velocity_future_cosine_by_layer": [_finite_or_none(v) for v in last_cos],
                f"{kind}_same_shuffled_velocity_cosine_by_layer": [_finite_or_none(v) for v in shuffle_cos],
                f"{kind}_shuffled_velocity_future_cosine_by_layer": [_finite_or_none(v) for v in shuffled_future_cos],
                f"{kind}_mean_velocity_norm_by_layer": past_norm,
                f"{kind}_last_velocity_norm_by_layer": last_norm,
                f"{kind}_future_displacement_norm_by_layer": future_norm,
                f"{kind}_mean_velocity_future_cosine": _finite_mean(mean_cos),
                f"{kind}_last_velocity_future_cosine": _finite_mean(last_cos),
                f"{kind}_shuffled_velocity_future_cosine": _finite_mean(shuffled_future_cos),
            })
        pending["record"]["geometry"] = geometry

    generated = x[0, prompt.shape[1]:].detach().cpu()
    sanity = {
        "reference_equals_followup_traced": bool(torch.equal(generated, reference_generated)),
        "vanilla_probe_max_abs_logit_error": max_vanilla_error if verify_vanilla_probe else None,
        "random_shuffle_norm_max_abs_error": max_norm_error,
        "random_shuffle_norm_max_relative_error": max_norm_relative_error,
        "projection_layers": controller.n_layers,
        "probe_records": len(records),
        "schedule": schedule,
    }
    return x, records, sanity
