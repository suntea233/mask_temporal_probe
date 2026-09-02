from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import ProbeConfig
from .projection import ProjectionController, matched_random
from .temporal_sampler import _schedule, _seed, _summarize_logits, get_num_transfer_tokens


CANDIDATE_LAYERS = (20, 24, 26, 28, 31)
LATENT_CONDITIONS = ("previous", "early", "shuffle", "random", "endpoint")
SCENARIO_BATCH_SIZE = 8
MATURITY_EPSILON = 1e-8


def _strict_future_endpoint(reveal_step: int, probe_step: int) -> bool:
    return reveal_step > probe_step


def _record_sanity(records: list[dict[str, Any]], hidden_isolated: bool, hard_isolated: bool) -> dict[str, bool]:
    nonempty = bool(records)
    return {
        "previous_same_position_step_layer": nonempty and all(r["previous_step"] == r["step_in_block"] - 1 for r in records),
        "early_is_first_unresolved": nonempty and all(r["early_step"] == 1 for r in records),
        "endpoint_same_position_layer_pre_reveal": nonempty and all(
            r["endpoint_step"] == r["reveal_step"] and r["endpoint_horizon"] > 0 for r in records
        ),
        "shuffle_same_block_different_position": nonempty and all(
            r["absolute_position"] != r["shuffle_source_position"] for r in records
        ),
        "only_target_modified": nonempty and hidden_isolated,
        "hard_only_target_token_changed": nonempty and hard_isolated,
        "downstream_targets_nonempty": nonempty and all(r["downstream_count"] > 0 for r in records),
    }


def _prediction_rows(logits: torch.Tensor, positions: list[int], reference: torch.Tensor, prompt_length: int) -> list[dict[str, Any]]:
    rows = logits[0, positions].float()
    probabilities = F.softmax(rows, -1); log_probabilities = F.log_softmax(rows, -1)
    top_probability, top_token = probabilities.max(-1); top2 = probabilities.topk(2, dim=-1).values
    targets = reference[torch.tensor([p - prompt_length for p in positions])].long().to(rows.device)
    target_logp = log_probabilities.gather(1, targets[:, None]).squeeze(1)
    entropy = -(probabilities * log_probabilities).sum(-1)
    return [{
        "top1_token": int(top_token[row]), "top1_probability": float(top_probability[row]),
        "top1_top2_margin": float(top2[row, 0] - top2[row, 1]), "entropy": float(entropy[row]),
        "future_target_logp": float(target_logp[row]), "hit": bool(int(top_token[row]) == int(targets[row])),
    } for row in range(len(positions))]


def _transfer(logits: torch.Tensor, x: torch.Tensor, mask_index: torch.Tensor, block_end: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = logits.argmax(-1); probabilities = F.softmax(logits, -1)
    confidence = probabilities.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    confidence[:, block_end:] = -np.inf; confidence = torch.where(mask_index, confidence, -np.inf)
    transfer = torch.zeros_like(x, dtype=torch.bool); _, indices = torch.topk(confidence[0], k=count); transfer[0, indices] = True
    return torch.where(mask_index, x0, x), transfer


def _condition_result(logits: torch.Tensor, target: int, vanilla: dict[str, Any], downstream_gain: float) -> dict[str, Any]:
    result = _summarize_logits(logits, target)
    result.update({
        "self_delta_logp": result["logp"] - vanilla["logp"], "downstream_gain": downstream_gain,
        "w2r": bool(not vanilla["hit"] and result["hit"]), "r2w": bool(vanilla["hit"] and not result["hit"]),
    })
    return result


def _downstream_gain(logits: torch.Tensor, positions: list[int], targets: list[int], baseline_logp: list[float]) -> float:
    if not positions:
        raise ValueError("Downstream gain requires at least one eligible non-target position")
    if not (len(positions) == len(targets) == len(baseline_logp)):
        raise ValueError("Downstream positions, targets, and baselines must have equal length")
    rows = logits[positions].float(); target_tensor = torch.tensor(targets, device=rows.device)
    logp = F.log_softmax(rows, -1).gather(1, target_tensor[:, None]).squeeze(1).cpu()
    return float((logp - torch.tensor(baseline_logp)).mean())


def _run_hidden_scenarios(
    model, controller: ProjectionController, snapshot: torch.Tensor, attention_mask: torch.Tensor | None,
    layer: int, scenarios: list[dict[str, Any]], records: list[dict[str, Any]],
) -> bool:
    device = model.device
    target_isolation_verified = True
    for start in range(0, len(scenarios), SCENARIO_BATCH_SIZE):
        chunk = scenarios[start:start + SCENARIO_BATCH_SIZE]; batch = len(chunk)
        x_batch = snapshot.to(device).repeat(batch, 1)
        mask_batch = None if attention_mask is None else attention_mask.repeat(batch, 1)
        positions = [scenario["absolute_position"] for scenario in chunk]
        replacements = torch.stack([scenario["replacement"] for scenario in chunk])
        target_isolation_verified &= all(
            torch.equal(x_batch[row].cpu(), snapshot[0])
            and scenario["absolute_position"] == records[scenario["record_index"]]["absolute_position"]
            for row, scenario in enumerate(chunk)
        )
        with controller.mode(positions, batch_indices=list(range(batch)), h={layer: replacements}):
            logits = model(x_batch, attention_mask=mask_batch).logits
        for row, scenario in enumerate(chunk):
            record = records[scenario["record_index"]]
            downstream = _downstream_gain(
                logits[row], record["other_positions"], record["other_targets"], record["other_vanilla_logp"],
            )
            result = _condition_result(
                logits[row, scenario["absolute_position"]].detach().float().cpu(),
                record["future_vanilla_target"], record["vanilla"], downstream,
            )
            record["layers"][str(layer)][scenario["condition"]] = result
    return target_isolation_verified


def _run_hard(
    model, snapshots: torch.Tensor, attention_mask: torch.Tensor | None, records: list[dict[str, Any]], indices: list[int],
) -> bool:
    if not indices:
        return False
    device = model.device
    target_isolation_verified = True
    for start in range(0, len(indices), SCENARIO_BATCH_SIZE):
        chunk = indices[start:start + SCENARIO_BATCH_SIZE]; batch = len(chunk)
        x_batch = snapshots.to(device).repeat(batch, 1)
        for row, index in enumerate(chunk):
            record = records[index]; x_batch[row, record["absolute_position"]] = record["hard_token"]
            changed = torch.nonzero(x_batch[row].cpu() != snapshots[0]).flatten().tolist()
            target_isolation_verified &= changed == [record["absolute_position"]]
        mask_batch = None if attention_mask is None else attention_mask.repeat(batch, 1)
        logits = model(x_batch, attention_mask=mask_batch).logits
        for row, index in enumerate(chunk):
            record = records[index]
            record["hard_downstream_gain"] = _downstream_gain(
                logits[row], record["other_positions"], record["other_targets"], record["other_vanilla_logp"],
            )
    return target_isolation_verified


@torch.inference_mode()
def unified_latent_generate(
    model, prompt: torch.Tensor, attention_mask: torch.Tensor | None, config: ProbeConfig, *,
    sample_id: int, reference_generated: torch.Tensor, special_token_ids: set[int], verify_resume: bool,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    if prompt.shape[0] != 1 or config.temperature != 0 or config.cfg_scale != 0:
        raise ValueError("Unified latent probe requires batch=1, temperature=0, cfg=0")
    device = model.device; prompt_length = prompt.shape[1]
    x = torch.full((1, prompt_length + config.gen_length), config.mask_id, dtype=torch.long, device=device); x[:, :prompt_length] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((1, config.gen_length), dtype=attention_mask.dtype, device=device)], -1)
    blocks = config.gen_length // config.block_length; steps_per_block = config.steps // blocks
    schedule = _schedule(steps_per_block, config.history, config.progress_fractions)
    reference_generated = reference_generated.cpu(); eos_id = getattr(model.config, "eos_token_id", 126081)
    eos_positions = torch.nonzero(reference_generated == eos_id).flatten(); content_end = int(eos_positions[0]) if eos_positions.numel() else config.gen_length
    controller = ProjectionController(model); records: list[dict[str, Any]] = []
    max_resume_error = 0.0; resume_checked = False; max_random_norm_relative_error = 0.0
    hidden_target_isolation_verified = True; hard_target_isolation_verified = True
    try:
        for block in range(blocks):
            block_start = prompt_length + block * config.block_length; block_end = block_start + config.block_length
            block_positions = list(range(block_start, block_end)); transfer_counts = get_num_transfer_tokens(x[:, block_start:block_end] == config.mask_id, steps_per_block)
            hidden_by_step: dict[int, dict[int, torch.Tensor]] = {}; predictions: dict[int, list[dict[str, Any]]] = {}
            reveal_steps: dict[int, int] = {}; probe_specs: list[dict[str, Any]] = []
            for step_zero in range(steps_per_block):
                step = step_zero + 1
                with controller.mode(block_positions, capture_h=True, capture_kv=False):
                    logits = model(x, attention_mask=attention_mask).logits
                captured = controller.hidden_state(); hidden_by_step[step] = {layer: captured[layer] for layer in CANDIDATE_LAYERS}
                predictions[step] = _prediction_rows(logits, block_positions, reference_generated, prompt_length)
                active_rows = torch.nonzero(x[0, block_start:block_end] == config.mask_id).flatten().tolist()
                if step in schedule and step > 1 and len(active_rows) >= 2:
                    selected = active_rows[:config.n_mask]
                    shuffle = {row: active_rows[(active_rows.index(row) + 1) % len(active_rows)] for row in selected}
                    probe_specs.append({"step": step, "progress": schedule[step], "selected": selected, "shuffle": shuffle, "active": active_rows, "snapshot": x.detach().cpu().clone()})
                mask_index = x == config.mask_id; x0, transfer = _transfer(logits, x, mask_index, block_end, int(transfer_counts[0, step_zero]))
                for absolute in torch.nonzero(transfer[0, block_start:block_end]).flatten().add(block_start).tolist(): reveal_steps[absolute - block_start] = step
                x[transfer] = x0[transfer]

            for probe in probe_specs:
                step = probe["step"]; local_indices = []
                for row in probe["selected"]:
                    gen_position = block * config.block_length + row; target = int(reference_generated[gen_position])
                    if gen_position >= content_end or target in special_token_ids or row not in reveal_steps:
                        continue
                    # The endpoint condition is explicitly a future-information
                    # oracle.  If this position reveals in the probe forward,
                    # its pre-reveal endpoint is identical to the current state
                    # and is therefore not an intervention.
                    if not _strict_future_endpoint(reveal_steps[row], step):
                        continue
                    absolute = block_start + row; current_prediction = predictions[step][row]; early_entropy = predictions[1][row]["entropy"]
                    relative_entropy = current_prediction["entropy"] / (early_entropy + MATURITY_EPSILON)
                    other_rows = [other for other in probe["active"] if other != row and block * config.block_length + other < content_end and int(reference_generated[block * config.block_length + other]) not in special_token_ids]
                    # Downstream utility is undefined without another eligible
                    # unresolved position.  Do not encode missing measurements
                    # as an artificial zero.
                    if not other_rows:
                        continue
                    record = {
                        "sample_id": sample_id, "block_index": block, "step_in_block": step, "progress": probe["progress"],
                        "absolute_position": absolute, "generation_position": gen_position, "shuffle_source_position": block_start + probe["shuffle"][row],
                        "future_vanilla_target": target, "reveal_step": reveal_steps[row],
                        "endpoint_horizon": reveal_steps[row] - step,
                        "previous_step": step - 1, "early_step": 1, "endpoint_step": reveal_steps[row],
                        "entropy": current_prediction["entropy"], "early_entropy": early_entropy,
                        "relative_entropy": relative_entropy, "maturity": 1 - relative_entropy,
                        "top1_probability": current_prediction["top1_probability"], "top1_top2_margin": current_prediction["top1_top2_margin"],
                        "hard_token": current_prediction["top1_token"], "hard_future_token_consistent": current_prediction["top1_token"] == target,
                        "vanilla": {"logp": current_prediction["future_target_logp"], "hit": current_prediction["hit"], "top1": current_prediction["top1_token"]},
                        "other_positions": [block_start + other for other in other_rows],
                        "other_targets": [int(reference_generated[block * config.block_length + other]) for other in other_rows],
                        "other_vanilla_logp": [predictions[step][other]["future_target_logp"] for other in other_rows],
                        "downstream_count": len(other_rows),
                        "layers": {str(layer): {} for layer in CANDIDATE_LAYERS}, "hard_downstream_gain": None,
                    }
                    records.append(record); local_indices.append(len(records) - 1)

                if not local_indices:
                    continue
                # Exact current-state resume check, once per sample, using one target and all candidate layers.
                if verify_resume and not resume_checked:
                    baseline = model(probe["snapshot"].to(device), attention_mask=attention_mask).logits
                    first = records[local_indices[0]]; first_row = first["absolute_position"] - block_start
                    for layer in CANDIDATE_LAYERS:
                        current = hidden_by_step[step][layer][first_row].unsqueeze(0)
                        with controller.mode([first["absolute_position"]], h={layer: current}):
                            resumed = model(probe["snapshot"].to(device), attention_mask=attention_mask).logits
                        max_resume_error = max(max_resume_error, float((resumed - baseline).abs().max()))
                    resume_checked = True

                for layer in CANDIDATE_LAYERS:
                    scenarios = []
                    for index in local_indices:
                        record = records[index]; row = record["absolute_position"] - block_start
                        source_row = record["shuffle_source_position"] - block_start
                        current = hidden_by_step[step][layer][row].float(); previous = hidden_by_step[step - 1][layer][row].float()
                        early = hidden_by_step[1][layer][row].float(); shuffled = hidden_by_step[step - 1][layer][source_row].float()
                        endpoint = hidden_by_step[reveal_steps[row]][layer][row].float(); displacement = previous - current
                        random = current + matched_random(displacement.unsqueeze(0), _seed(config.seed, "unified-random", sample_id, block, step, row, layer))[0].float()
                        denominator = max(float(displacement.norm()), torch.finfo(torch.float32).tiny)
                        max_random_norm_relative_error = max(max_random_norm_relative_error, abs(float((random-current).norm()) - float(displacement.norm())) / denominator)
                        values = {"previous": previous, "early": early, "shuffle": shuffled, "random": random, "endpoint": endpoint}
                        for condition, replacement in values.items():
                            scenarios.append({"record_index": index, "absolute_position": record["absolute_position"], "condition": condition, "replacement": replacement})
                    hidden_target_isolation_verified &= _run_hidden_scenarios(
                        model, controller, probe["snapshot"], attention_mask, layer, scenarios, records,
                    )
                hard_target_isolation_verified &= _run_hard(model, probe["snapshot"], attention_mask, records, local_indices)
    finally:
        controller.close()

    # Remove counterfactual-only convenience arrays from persisted records after all forwards.
    for record in records:
        record.pop("other_positions"); record.pop("other_targets"); record.pop("other_vanilla_logp")
    generated = x[0, prompt_length:].detach().cpu()
    sanity = {
        "reference_equals_unified_traced": bool(torch.equal(generated, reference_generated)), "projection_layers": controller.n_layers,
        "resume_current_max_abs_logit_error": max_resume_error if verify_resume else None, "resume_checked": resume_checked if verify_resume else None,
        "random_norm_max_relative_error": max_random_norm_relative_error, "observations": len(records), "schedule": schedule,
    }
    sanity.update(_record_sanity(records, hidden_target_isolation_verified, hard_target_isolation_verified))
    return x, records, sanity
