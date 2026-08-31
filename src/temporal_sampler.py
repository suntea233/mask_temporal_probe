from __future__ import annotations

import hashlib
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import ProbeConfig
from .projection import ProjectedState, ProjectionController, matched_random


@dataclass
class StepState:
    k: dict[int, torch.Tensor]
    v: dict[int, torch.Tensor]


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base, remainder = mask_num // steps, mask_num % steps
    result = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for row in range(mask_num.size(0)):
        result[row, : remainder[row]] += 1
    return result


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _schedule(steps_per_block: int, history: int, fractions: tuple[float, ...]) -> dict[int, str]:
    result: dict[int, str] = {}
    used: set[int] = set()
    for fraction in fractions:
        target = fraction * steps_per_block
        choices = [s for s in range(history + 1, steps_per_block + 1) if s not in used]
        step = min(choices, key=lambda s: (abs(s - target), s))
        used.add(step)
        result[step] = f"{round(100 * fraction):d}%"
    return result


def _split_state(captured: ProjectedState, positions: list[int]) -> dict[int, StepState]:
    states: dict[int, StepState] = {}
    for row, position in enumerate(positions):
        states[position] = StepState(
            k={layer: tensor[row].clone() for layer, tensor in captured.k.items()},
            v={layer: tensor[row].clone() for layer, tensor in captured.v.items()},
        )
    return states


def _mean_previous(history: deque[StepState], kind: str, layer: int, h: int) -> torch.Tensor:
    previous = list(history)[-(h + 1):-1]
    return torch.stack([getattr(item, kind)[layer].float() for item in previous]).mean(0)


def _counterfactual(model, controller: ProjectionController, x: torch.Tensor, attention_mask, positions, k=None, v=None):
    with controller.mode(positions, k=k, v=v):
        return model(x, attention_mask=attention_mask).logits[0, positions].detach().float().cpu()


def _summarize_logits(logits: torch.Tensor, target: int) -> dict[str, Any]:
    logp = F.log_softmax(logits.float(), dim=-1)
    probs = logp.exp()
    top_prob, top_token = probs.max(dim=-1)
    entropy = -(probs * logp).sum(dim=-1)
    return {
        "logp": float(logp[target]),
        "top1_token": int(top_token),
        "top1_probability": float(top_prob),
        "hit": bool(int(top_token) == target),
        "entropy": float(entropy),
    }


@torch.inference_mode()
def traced_generate(
    model,
    prompt: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: ProbeConfig,
    *,
    sample_id: int,
    special_token_ids: set[int],
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    """Behavior-preserving official sampler plus read-only causal probe forwards."""
    if prompt.shape[0] != 1 or config.cfg_scale != 0 or config.temperature != 0:
        raise ValueError("The fixed experiment requires batch=1, cfg=0, and temperature=0")
    device = model.device
    x = torch.full((1, prompt.shape[1] + config.gen_length), config.mask_id, dtype=torch.long, device=device)
    x[:, : prompt.shape[1]] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, config.gen_length), dtype=attention_mask.dtype, device=device),
        ], dim=-1)
    if config.gen_length % config.block_length or config.steps % (config.gen_length // config.block_length):
        raise ValueError("Invalid block/step divisibility")
    blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // blocks
    schedule = _schedule(steps_per_block, config.history, config.progress_fractions)
    controller = ProjectionController(model)
    pending: list[dict[str, Any]] = []
    max_vanilla_error = 0.0
    random_norm_max_error = 0.0
    histories: dict[int, deque[StepState]]
    logit_histories: dict[int, deque[torch.Tensor]]

    try:
        for block in range(blocks):
            block_start = prompt.shape[1] + block * config.block_length
            block_end = block_start + config.block_length
            block_masks = x[:, block_start:block_end] == config.mask_id
            transfer_counts = get_num_transfer_tokens(block_masks, steps_per_block)
            histories = defaultdict(lambda: deque(maxlen=config.history + 1))
            logit_histories = defaultdict(lambda: deque(maxlen=config.history + 1))
            unresolved = defaultdict(int)

            for step_zero in range(steps_per_block):
                step = step_zero + 1
                mask_index = x == config.mask_id
                active = [int(p) for p in torch.nonzero(mask_index[0, block_start:block_end]).flatten().add(block_start)]
                with controller.mode(active, capture=True):
                    logits = model(x, attention_mask=attention_mask).logits
                states = _split_state(controller.state(), active)
                for row, position in enumerate(active):
                    unresolved[position] += 1
                    histories[position].append(states[position])
                    logit_histories[position].append(logits[0, position].detach().float().cpu().clone())

                # Continue every previously opened observational trajectory
                # until the position's untouched vanilla reveal step.
                active_rows = {position: row for row, position in enumerate(active)}
                for item in pending:
                    position = item["absolute_position"]
                    if item["block_index"] == block and position in active_rows and item["last_trajectory_step"] < step:
                        item["trajectory_logits"].append(logits[0, position].detach().float().cpu().clone())
                        item["trajectory_steps"].append(step)
                        item["last_trajectory_step"] = step

                eligible = [p for p in active if len(histories[p]) == config.history + 1 and unresolved[p] >= config.history + 1]
                if step in schedule and eligible:
                    eligible.sort(key=lambda p: (-unresolved[p], p))
                    selected = eligible[: config.n_mask]
                    if len(eligible) >= 2:
                        shuffle_source = {p: eligible[(eligible.index(p) + 1) % len(eligible)] for p in selected}
                        real_k: dict[int, torch.Tensor] = {}
                        real_v: dict[int, torch.Tensor] = {}
                        shuffle_k: dict[int, torch.Tensor] = {}
                        shuffle_v: dict[int, torch.Tensor] = {}
                        random_k: dict[int, torch.Tensor] = {}
                        random_v: dict[int, torch.Tensor] = {}
                        drift_k, drift_v = [], []

                        for layer in range(controller.n_layers):
                            rk, rv, sk, sv, nk, nv = [], [], [], [], [], []
                            for position in selected:
                                current = histories[position][-1]
                                hist_k = _mean_previous(histories[position], "k", layer, config.history)
                                hist_v = _mean_previous(histories[position], "v", layer, config.history)
                                source = shuffle_source[position]
                                shuffled_hist_k = _mean_previous(histories[source], "k", layer, config.history)
                                shuffled_hist_v = _mean_previous(histories[source], "v", layer, config.history)
                                dk = hist_k - current.k[layer].float()
                                dv = hist_v - current.v[layer].float()
                                rand_k = matched_random(dk.unsqueeze(0), _seed(config.seed, sample_id, block, step, position, layer, "k"))[0]
                                rand_v = matched_random(dv.unsqueeze(0), _seed(config.seed, sample_id, block, step, position, layer, "v"))[0]
                                random_norm_max_error = max(
                                    random_norm_max_error,
                                    abs(float(rand_k.float().norm() - dk.norm())),
                                    abs(float(rand_v.float().norm() - dv.norm())),
                                )
                                rk.append(current.k[layer].float() + config.alpha * dk)
                                rv.append(current.v[layer].float() + config.alpha * dv)
                                sk.append((1 - config.alpha) * current.k[layer].float() + config.alpha * shuffled_hist_k)
                                sv.append((1 - config.alpha) * current.v[layer].float() + config.alpha * shuffled_hist_v)
                                nk.append(current.k[layer].float() + config.alpha * rand_k.float())
                                nv.append(current.v[layer].float() + config.alpha * rand_v.float())
                            real_k[layer], real_v[layer] = torch.stack(rk), torch.stack(rv)
                            shuffle_k[layer], shuffle_v[layer] = torch.stack(sk), torch.stack(sv)
                            random_k[layer], random_v[layer] = torch.stack(nk), torch.stack(nv)

                        vanilla_probe = _counterfactual(model, controller, x, attention_mask, selected)
                        vanilla_main = logits[0, selected].detach().float().cpu()
                        max_vanilla_error = max(max_vanilla_error, float((vanilla_probe - vanilla_main).abs().max()))
                        condition_logits = {
                            "vanilla": vanilla_main,
                            "real": _counterfactual(model, controller, x, attention_mask, selected, real_k, real_v),
                            "shuffle": _counterfactual(model, controller, x, attention_mask, selected, shuffle_k, shuffle_v),
                            "random": _counterfactual(model, controller, x, attention_mask, selected, random_k, random_v),
                        }
                        logit_fusion = []
                        for position in selected:
                            history_logits = list(logit_histories[position])
                            logit_fusion.append((1 - config.alpha) * history_logits[-1] + config.alpha * torch.stack(history_logits[:-1]).mean(0))
                        condition_logits["logit_history"] = torch.stack(logit_fusion)

                        for row, position in enumerate(selected):
                            current_state, previous_state = histories[position][-1], histories[position][-2]
                            drift_k = [float((current_state.k[l].float() - previous_state.k[l].float()).norm()) for l in range(controller.n_layers)]
                            drift_v = [float((current_state.v[l].float() - previous_state.v[l].float()).norm()) for l in range(controller.n_layers)]
                            pending.append({
                                "sample_id": sample_id,
                                "block_index": block,
                                "step_in_block": step,
                                "progress": schedule[step],
                                "absolute_position": position,
                                "generation_position": position - prompt.shape[1],
                                "unresolved_steps": unresolved[position],
                                "shuffle_source_position": shuffle_source[position],
                                "condition_logits": {name: value[row].clone() for name, value in condition_logits.items()},
                                "trajectory_logits": [value.clone() for value in logit_histories[position]],
                                "trajectory_steps": list(range(step - config.history, step + 1)),
                                "probe_trajectory_length": config.history + 1,
                                "last_trajectory_step": step,
                                "drift_k_mean_layer_l2": float(np.mean(drift_k)),
                                "drift_v_mean_layer_l2": float(np.mean(drift_v)),
                                "drift_kv_mean_layer_l2": float(np.mean(drift_k + drift_v)),
                            })

                logits_with_noise = logits
                x0 = torch.argmax(logits_with_noise, dim=-1)
                if config.remasking != "low_confidence":
                    raise NotImplementedError(config.remasking)
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

    generated = x[0, prompt.shape[1]:]
    eos_positions = torch.nonzero(generated == getattr(model.config, "eos_token_id", 126081)).flatten()
    content_end = int(eos_positions[0]) if eos_positions.numel() else config.gen_length
    records: list[dict[str, Any]] = []
    for item in pending:
        gen_position = item["generation_position"]
        target = int(generated[gen_position])
        if gen_position >= content_end or target in special_token_ids:
            continue
        summaries = {name: _summarize_logits(value, target) for name, value in item.pop("condition_logits").items()}
        trajectory_logits = item.pop("trajectory_logits")
        trajectory_steps = item.pop("trajectory_steps")
        probe_trajectory_length = item.pop("probe_trajectory_length")
        item.pop("last_trajectory_step")
        trajectory = []
        for step, value in zip(trajectory_steps, trajectory_logits):
            point = _summarize_logits(value, target)
            point["step_in_block"] = step
            trajectory.append(point)
        historical = trajectory[:probe_trajectory_length]
        vanilla_logp = summaries["vanilla"]["logp"]
        for name in ("real", "shuffle", "random", "logit_history"):
            summaries[name]["delta_logp"] = summaries[name]["logp"] - vanilla_logp
            summaries[name]["w2r"] = bool(not summaries["vanilla"]["hit"] and summaries[name]["hit"])
            summaries[name]["r2w"] = bool(summaries["vanilla"]["hit"] and not summaries[name]["hit"])
        top_tokens = [point["top1_token"] for point in trajectory]
        item.update({
            "future_vanilla_target": target,
            "conditions": summaries,
            "prediction_trajectory": trajectory,
            "top1_already_future_target_by_offset": {str(offset - config.history): historical[offset]["hit"] for offset in range(config.history + 1)},
            "top1_stable": len(set(top_tokens)) == 1,
            "top1_stable_confidence_rises": len(set(top_tokens)) == 1 and trajectory[-1]["top1_probability"] > trajectory[0]["top1_probability"],
            "top1_change_count": sum(a != b for a, b in zip(top_tokens, top_tokens[1:])),
        })
        records.append(item)
    sanity = {
        "vanilla_probe_max_abs_logit_error": max_vanilla_error,
        "random_norm_max_abs_error": random_norm_max_error,
        "projection_layers": controller.n_layers,
        "probe_records": len(records),
        "schedule": schedule,
    }
    return x, records, sanity
