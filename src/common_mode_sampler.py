from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import ProbeConfig
from .directionality_sampler import _condition_result, _cosine, _finite_mean, _finite_or_none, _match_direction
from .projection import ProjectionController, matched_random
from .temporal_sampler import StepState, _counterfactual, _schedule, _seed, _split_state, _summarize_logits, get_num_transfer_tokens


PRIMARY_CONDITIONS = (
    "full_velocity",
    "common_only",
    "real_residual",
    "shuffled_residual",
    "random_residual",
)
LAYER_BANDS = {
    "00_07": range(0, 8),
    "08_15": range(8, 16),
    "16_23": range(16, 24),
    "24_31": range(24, 32),
}
BAND_CONDITIONS = tuple(
    f"{component}_layers_{band}"
    for component in ("common", "residual")
    for band in LAYER_BANDS
)
ALL_CONDITIONS = PRIMARY_CONDITIONS + BAND_CONDITIONS


def _mean_velocity(states: dict[int, StepState], histories: dict[int, deque[StepState]], positions: list[int], kind: str, layer: int) -> torch.Tensor:
    return torch.stack([
        getattr(states[p], kind)[layer].float() - getattr(histories[p][-2], kind)[layer].float()
        for p in positions
    ]).mean(0)


def _pairwise_cosine_mean(vectors: list[torch.Tensor]) -> float | None:
    if len(vectors) < 2:
        return None
    matrix = torch.stack([value.float().flatten() for value in vectors])
    norms = matrix.norm(dim=1)
    matrix = matrix[norms > 0]
    norms = norms[norms > 0]
    if len(matrix) < 2:
        return None
    normalized = matrix / norms[:, None]
    similarities = normalized @ normalized.T
    upper = similarities[torch.triu(torch.ones_like(similarities, dtype=torch.bool), diagonal=1)]
    return float(upper.mean())


def _transfer_mask(logits: torch.Tensor, x: torch.Tensor, mask_index: torch.Tensor, block_end: int, count: int) -> torch.Tensor:
    x0 = torch.argmax(logits, dim=-1)
    probabilities = F.softmax(logits, dim=-1)
    confidence = torch.gather(probabilities, -1, x0.unsqueeze(-1)).squeeze(-1)
    confidence[:, block_end:] = -np.inf
    confidence = torch.where(mask_index, confidence, -np.inf)
    transfer = torch.zeros_like(x, dtype=torch.bool)
    _, indices = torch.topk(confidence[0], k=count)
    transfer[0, indices] = True
    return transfer


def _descriptive_geometry(
    states: dict[int, StepState], histories: dict[int, deque[StepState]], positions: list[int], n_layers: int
) -> dict[str, Any]:
    result: dict[str, Any] = {"unresolved_count": len(positions)}
    max_decomposition_error = 0.0
    for kind in ("k", "v"):
        total, common, residual = [], [], []
        common_fraction, residual_fraction = [], []
        raw_pairwise, centered_pairwise = [], []
        for layer in range(n_layers):
            velocities = [
                getattr(states[p], kind)[layer].float() - getattr(histories[p][-2], kind)[layer].float()
                for p in positions
            ]
            mean = torch.stack(velocities).mean(0)
            centered = [value - mean for value in velocities]
            total_energy = float(sum(torch.dot(value.flatten(), value.flatten()) for value in velocities))
            common_energy = float(len(positions) * torch.dot(mean.flatten(), mean.flatten()))
            residual_energy = float(sum(torch.dot(value.flatten(), value.flatten()) for value in centered))
            denominator = max(total_energy, torch.finfo(torch.float32).tiny)
            max_decomposition_error = max(max_decomposition_error, abs(total_energy - common_energy - residual_energy) / denominator)
            total.append(total_energy); common.append(common_energy); residual.append(residual_energy)
            common_fraction.append(common_energy / denominator); residual_fraction.append(residual_energy / denominator)
            raw_pairwise.append(_pairwise_cosine_mean(velocities))
            centered_pairwise.append(_pairwise_cosine_mean(centered))
        result.update({
            f"{kind}_total_energy_by_layer": total,
            f"{kind}_common_energy_by_layer": common,
            f"{kind}_residual_energy_by_layer": residual,
            f"{kind}_common_energy_fraction_by_layer": common_fraction,
            f"{kind}_residual_energy_fraction_by_layer": residual_fraction,
            f"{kind}_pairwise_raw_cosine_by_layer": raw_pairwise,
            f"{kind}_pairwise_residual_cosine_by_layer": centered_pairwise,
        })
    result["decomposition_max_relative_error"] = max_decomposition_error
    return result


@torch.inference_mode()
def common_mode_generate(
    model,
    prompt: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: ProbeConfig,
    *,
    sample_id: int,
    reference_generated: torch.Tensor,
    special_token_ids: set[int],
    verify_vanilla_probe: bool,
    forward_reference: dict[tuple[int, int], dict[int, float]] | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if prompt.shape[0] != 1 or config.cfg_scale != 0 or config.temperature != 0:
        raise ValueError("Common-mode probe requires batch=1, cfg=0, temperature=0")
    device = model.device
    x = torch.full((1, prompt.shape[1] + config.gen_length), config.mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((1, config.gen_length), dtype=attention_mask.dtype, device=device)], -1)
    blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // blocks
    schedule = _schedule(steps_per_block, config.history, config.progress_fractions)
    reference_generated = reference_generated.cpu()
    eos_id = getattr(model.config, "eos_token_id", 126081)
    eos_positions = torch.nonzero(reference_generated == eos_id).flatten()
    content_end = int(eos_positions[0]) if eos_positions.numel() else config.gen_length

    controller = ProjectionController(model)
    records: list[dict[str, Any]] = []
    probe_states: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    max_vanilla_error = 0.0
    max_norm_relative_error = 0.0
    max_decomposition_error = 0.0
    modified_positions_valid = True
    replication_overlap = 0
    replication_max_error = 0.0
    try:
        if controller.n_layers != 32:
            raise RuntimeError(f"Expected 32 layers, found {controller.n_layers}")
        for block in range(blocks):
            block_start = prompt.shape[1] + block * config.block_length
            block_end = block_start + config.block_length
            transfer_counts = get_num_transfer_tokens(x[:, block_start:block_end] == config.mask_id, steps_per_block)
            histories: dict[int, deque[StepState]] = defaultdict(lambda: deque(maxlen=config.history + 1))
            unresolved = defaultdict(int)
            for step_zero in range(steps_per_block):
                step = step_zero + 1
                mask_index = x == config.mask_id
                active = [int(p) for p in torch.nonzero(mask_index[0, block_start:block_end]).flatten().add(block_start)]
                with controller.mode(active, capture=True):
                    logits = model(x, attention_mask=attention_mask).logits
                states = _split_state(controller.state(), active)

                # Complete t -> t+1 temporal geometry before appending the new states.
                current_m = [p for p in active if len(histories[p]) >= 1]
                for item in [q for q in pending if q["block_index"] == block and q["next_step"] == step]:
                    if item["position"] not in current_m or len(current_m) < 4:
                        continue
                    temporal: dict[str, Any] = {}
                    for kind in ("k", "v"):
                        same_raw, shuffled_raw, common_raw, same_residual, shuffled_residual = [], [], [], [], []
                        for layer in range(controller.n_layers):
                            next_velocities = {
                                p: getattr(states[p], kind)[layer].float() - getattr(histories[p][-1], kind)[layer].float()
                                for p in current_m
                            }
                            mu_next = torch.stack(list(next_velocities.values())).mean(0)
                            next_target = next_velocities[item["position"]]
                            next_target_residual = next_target - mu_next
                            same_raw.append(_cosine(item[f"{kind}_raw"][layer], next_target))
                            shuffled_raw.append(_cosine(item[f"{kind}_shuffle_raw"][layer], next_target))
                            common_raw.append(_cosine(item[f"{kind}_mean"][layer], next_target))
                            same_residual.append(_cosine(item[f"{kind}_residual"][layer], next_target_residual))
                            shuffled_residual.append(_cosine(item[f"{kind}_shuffle_residual"][layer], next_target_residual))
                        temporal.update({
                            f"{kind}_same_raw_alignment_by_layer": [_finite_or_none(v) for v in same_raw],
                            f"{kind}_shuffled_raw_alignment_by_layer": [_finite_or_none(v) for v in shuffled_raw],
                            f"{kind}_common_next_alignment_by_layer": [_finite_or_none(v) for v in common_raw],
                            f"{kind}_same_residual_alignment_by_layer": [_finite_or_none(v) for v in same_residual],
                            f"{kind}_shuffled_residual_alignment_by_layer": [_finite_or_none(v) for v in shuffled_residual],
                        })
                    item["record"]["temporal_geometry"] = temporal

                for position in active:
                    unresolved[position] += 1
                    histories[position].append(states[position])

                count = int(transfer_counts[0, step_zero])
                transfer = _transfer_mask(logits, x, mask_index, block_end, count)
                m_positions = [p for p in active if len(histories[p]) >= 2]
                if step in schedule and len(m_positions) >= 4:
                    geometry = _descriptive_geometry(states, histories, m_positions, controller.n_layers)
                    max_decomposition_error = max(max_decomposition_error, geometry["decomposition_max_relative_error"])
                    probe_record = {
                        "sample_id": sample_id, "block_index": block, "step_in_block": step,
                        "progress": schedule[step], **geometry,
                    }
                    probe_states.append(probe_record)

                    # Debug-only exact reproduction using the old probe's full
                    # simultaneous target set. Primary targets below obey the
                    # stricter t+1 eligibility rule and are intentionally different.
                    old = (forward_reference or {}).get((block, step), {})
                    old_positions = sorted(old)
                    if len(old_positions) == config.n_mask and all(p in m_positions for p in old_positions):
                        old_k: dict[int, torch.Tensor] = {}
                        old_v: dict[int, torch.Tensor] = {}
                        for layer in range(controller.n_layers):
                            old_k[layer] = torch.stack([
                                getattr(states[p], "k")[layer].float()
                                + config.alpha * (getattr(states[p], "k")[layer].float() - getattr(histories[p][-2], "k")[layer].float())
                                for p in old_positions
                            ])
                            old_v[layer] = torch.stack([
                                getattr(states[p], "v")[layer].float()
                                + config.alpha * (getattr(states[p], "v")[layer].float() - getattr(histories[p][-2], "v")[layer].float())
                                for p in old_positions
                            ])
                        reproduced = _counterfactual(model, controller, x, attention_mask, old_positions, old_k, old_v)
                        vanilla_old = logits[0, old_positions].detach().float().cpu()
                        for row, position in enumerate(old_positions):
                            gen_position = position - prompt.shape[1]
                            target = int(reference_generated[gen_position])
                            delta = _summarize_logits(reproduced[row], target)["logp"] - _summarize_logits(vanilla_old[row], target)["logp"]
                            replication_max_error = max(replication_max_error, abs(delta - old[position]))
                            replication_overlap += 1
                    valid_next = [p for p in m_positions if not bool(transfer[0, p])]
                    valid_next.sort(key=lambda p: (-unresolved[p], p))
                    selected = valid_next[:config.n_mask] if len(valid_next) >= 4 else []
                    if selected:
                        source_for = {p: m_positions[(m_positions.index(p) + 1) % len(m_positions)] for p in selected}
                        replacements = {name: {"k": {}, "v": {}} for name in ALL_CONDITIONS}
                        temporal_inputs: dict[int, dict[str, Any]] = {p: {} for p in selected}
                        for layer in range(controller.n_layers):
                            for kind in ("k", "v"):
                                velocities = {
                                    p: getattr(states[p], kind)[layer].float() - getattr(histories[p][-2], kind)[layer].float()
                                    for p in m_positions
                                }
                                total = torch.stack(list(velocities.values())).sum(0)
                                per_condition = {name: [] for name in ALL_CONDITIONS}
                                for position in selected:
                                    source = source_for[position]
                                    current = getattr(states[position], kind)[layer].float()
                                    velocity = velocities[position]
                                    common_loo = (total - velocity) / (len(m_positions) - 1)
                                    residual = velocity - common_loo
                                    source_velocity = velocities[source]
                                    source_common_loo = (total - source_velocity) / (len(m_positions) - 1)
                                    source_residual = source_velocity - source_common_loo
                                    shuffled = _match_direction(source_residual, residual)
                                    random = matched_random(
                                        residual.unsqueeze(0),
                                        _seed(config.seed, "common", sample_id, block, step, position, layer, kind),
                                    )[0].float()
                                    target_norm = float(residual.norm())
                                    denominator = max(target_norm, torch.finfo(torch.float32).tiny)
                                    max_norm_relative_error = max(
                                        max_norm_relative_error,
                                        abs(float(shuffled.norm()) - target_norm) / denominator,
                                        abs(float(random.norm()) - target_norm) / denominator,
                                    )
                                    directions = {
                                        "full_velocity": velocity, "common_only": common_loo,
                                        "real_residual": residual, "shuffled_residual": shuffled,
                                        "random_residual": random,
                                    }
                                    for name, direction in directions.items():
                                        per_condition[name].append(current + config.alpha * direction)
                                    for band, layers in LAYER_BANDS.items():
                                        if layer in layers:
                                            per_condition[f"common_layers_{band}"].append(current + config.alpha * common_loo)
                                            per_condition[f"residual_layers_{band}"].append(current + config.alpha * residual)
                                    temporal_inputs[position].setdefault(f"{kind}_raw", {})[layer] = velocity
                                    ordinary_mean = total / len(m_positions)
                                    temporal_inputs[position].setdefault(f"{kind}_mean", {})[layer] = ordinary_mean
                                    temporal_inputs[position].setdefault(f"{kind}_residual", {})[layer] = velocity - ordinary_mean
                                    temporal_inputs[position].setdefault(f"{kind}_shuffle_raw", {})[layer] = source_velocity
                                    temporal_inputs[position].setdefault(f"{kind}_shuffle_residual", {})[layer] = source_velocity - ordinary_mean
                                for name, rows in per_condition.items():
                                    if rows:
                                        replacements[name][kind][layer] = torch.stack(rows)

                        vanilla_rows = logits[0, selected].detach().float().cpu()
                        if verify_vanilla_probe:
                            vanilla_probe = _counterfactual(model, controller, x, attention_mask, selected)
                            max_vanilla_error = max(max_vanilla_error, float((vanilla_probe - vanilla_rows).abs().max()))
                        condition_logits = {
                            name: _counterfactual(model, controller, x, attention_mask, selected, values["k"], values["v"])
                            for name, values in replacements.items()
                        }
                        modified_positions_valid &= all(
                            all(tensor.shape[0] == len(selected) for tensor in values["k"].values())
                            and all(tensor.shape[0] == len(selected) for tensor in values["v"].values())
                            for values in replacements.values()
                        )
                        for row, position in enumerate(selected):
                            gen_position = position - prompt.shape[1]
                            target = int(reference_generated[gen_position])
                            if gen_position >= content_end or target in special_token_ids:
                                continue
                            vanilla = _summarize_logits(vanilla_rows[row], target)
                            conditions = {"vanilla": vanilla}
                            for name in ALL_CONDITIONS:
                                conditions[name] = _condition_result(condition_logits[name][row], target, vanilla)
                            record = {
                                "sample_id": sample_id, "block_index": block, "step_in_block": step,
                                "progress": schedule[step], "absolute_position": position,
                                "generation_position": gen_position, "unresolved_steps": unresolved[position],
                                "common_set_size": len(m_positions), "shuffle_source_position": source_for[position],
                                "future_vanilla_target": target, "conditions": conditions,
                            }
                            records.append(record)
                            packed = {key: [values[layer] for layer in range(controller.n_layers)] for key, values in temporal_inputs[position].items()}
                            pending.append({
                                "record": record, "position": position, "block_index": block, "next_step": step + 1,
                                "n_layers": controller.n_layers, **packed,
                            })

                x0 = torch.argmax(logits, dim=-1)
                x0 = torch.where(mask_index, x0, x)
                x[transfer] = x0[transfer]
    finally:
        controller.close()

    generated = x[0, prompt.shape[1]:].detach().cpu()
    sanity = {
        "reference_equals_common_mode_traced": bool(torch.equal(generated, reference_generated)),
        "vanilla_probe_max_abs_logit_error": max_vanilla_error if verify_vanilla_probe else None,
        "norm_match_max_relative_error": max_norm_relative_error,
        "decomposition_max_relative_error": max_decomposition_error,
        "only_selected_positions_modified": bool(modified_positions_valid),
        "projection_layers": controller.n_layers,
        "probe_records": len(records), "probe_states": len(probe_states), "schedule": schedule,
        "temporal_geometry_complete": all("temporal_geometry" in record for record in records),
        "forward_replication_overlap": replication_overlap,
        "forward_replication_max_abs_delta_logp_error": replication_max_error if replication_overlap else None,
    }
    return x, records, probe_states, sanity
