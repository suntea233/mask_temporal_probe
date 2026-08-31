from __future__ import annotations

import argparse
import json
import os

import torch

from .common_mode_sampler import common_mode_generate
from .config import ProbeConfig
from .run_experiment import PROJECT, load_assets, prompt_for, seed_everything


def _directionality_lookup(sample_id: int) -> dict[tuple[int, int], dict[int, float]]:
    path = PROJECT / "traces/directionality_probe/formal" / f"sample_{sample_id:04d}.json"
    payload = json.loads(path.read_text())
    result: dict[tuple[int, int], dict[int, float]] = {}
    for r in payload["observations"]:
        result.setdefault((r["block_index"], r["step_in_block"]), {})[r["absolute_position"]] = r["conditions"]["forward_hidden_last"]["delta_logp"]
    return result


def run(start: int, count: int, mode: str) -> dict:
    config = ProbeConfig()
    if mode == "debug" and not 5 <= count <= 10:
        raise ValueError("Common-mode debug requires 5–10 samples")
    if mode == "formal":
        gate = PROJECT / "results/common_mode_debug_sanity.json"
        if not gate.exists() or not json.loads(gate.read_text())["all_passed"]:
            raise RuntimeError("Formal common-mode run refused until its debug gate passes")
    if start < 0 or count <= 0 or start + count > config.samples:
        raise ValueError("Requested range must be within samples 0:200")
    previous_gate = PROJECT / "results/directionality_debug_sanity.json"
    if not previous_gate.exists() or not json.loads(previous_gate.read_text())["all_passed"]:
        raise RuntimeError("Previous directionality sanity gate is absent or failed")

    seed_everything(config.seed)
    model, tokenizer, dataset = load_assets(config)
    special_ids = set(tokenizer.all_special_ids)
    output_dir = PROJECT / "traces/common_mode_probe" / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses = []
    for sample_id in range(start, start + count):
        sample_path = output_dir / f"sample_{sample_id:04d}.json"
        if sample_path.exists():
            statuses.append(json.loads(sample_path.read_text())["sanity"])
            continue
        reference_path = PROJECT / "traces/mask_temporal_state_probe/formal" / f"sample_{sample_id:04d}.json"
        reference = json.loads(reference_path.read_text())
        if not reference["sanity"]["passed"]:
            raise RuntimeError(f"Original sample {sample_id} failed sanity")
        prompt, attention_mask, formatted = prompt_for(tokenizer, dataset[sample_id]["question"], model.device)
        reference_generated = torch.tensor(reference["official_output_token_ids"], dtype=torch.long)
        seed_everything(config.seed + sample_id)
        cpu_before = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state().clone()
        previous = _directionality_lookup(sample_id)
        output, records, probe_states, sanity = common_mode_generate(
            model, prompt, attention_mask, config, sample_id=sample_id,
            reference_generated=reference_generated, special_token_ids=special_ids,
            verify_vanilla_probe=mode == "debug",
            forward_reference=previous if mode == "debug" else None,
        )
        sanity.update({
            "sample_id": sample_id,
            "cpu_rng_unchanged": bool(torch.equal(cpu_before, torch.random.get_rng_state())),
            "cuda_rng_unchanged": bool(torch.equal(cuda_before, torch.cuda.get_rng_state())),
            "shuffle_different_position": all(r["absolute_position"] != r["shuffle_source_position"] for r in records),
            "common_set_at_least_four": all(r["common_set_size"] >= 4 for r in records),
            "targets_have_next_state": all("temporal_geometry" in r for r in records),
        })
        sanity["passed"] = bool(
            sanity["reference_equals_common_mode_traced"]
            and sanity["vanilla_probe_max_abs_logit_error"] in (None, 0.0)
            and sanity["norm_match_max_relative_error"] <= 2e-6
            and sanity["decomposition_max_relative_error"] <= 2e-5
            and sanity["only_selected_positions_modified"]
            and sanity["projection_layers"] == 32
            and sanity["cpu_rng_unchanged"] and sanity["cuda_rng_unchanged"]
            and sanity["shuffle_different_position"] and sanity["common_set_at_least_four"]
            and sanity["targets_have_next_state"]
            and (
                mode != "debug"
                or (sanity["forward_replication_overlap"] > 0
                    and sanity["forward_replication_max_abs_delta_logp_error"] <= 2e-5)
            )
        )
        payload = {
            "sample_id": sample_id, "question": dataset[sample_id]["question"],
            "formatted_prompt": formatted, "reference_answer": reference["reference_answer"],
            "decoded_output": reference["decoded_output"],
            "reference_output_token_ids": reference_generated.tolist(),
            "common_mode_output_token_ids": output[0, prompt.shape[1]:].cpu().tolist(),
            "sanity": sanity, "observations": records, "probe_states": probe_states,
        }
        temporary = sample_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, sample_path)
        statuses.append(sanity)

    result = {
        "mode": mode, "start": start, "count": count, "completed": len(statuses),
        "all_passed": len(statuses) == count and all(s.get("passed", False) for s in statuses),
        "samples": statuses, "config": config.as_dict(),
    }
    destination = PROJECT / "results" / (
        "common_mode_debug_sanity.json" if mode == "debug"
        else f"common_mode_status_{start:04d}_{start + count:04d}.json"
    )
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("debug", "formal"), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    count = args.count if args.count is not None else (5 if args.mode == "debug" else 200)
    result = run(args.start, count, args.mode)
    print(json.dumps({k: result[k] for k in ("mode", "start", "count", "all_passed")}, indent=2))


if __name__ == "__main__":
    main()
