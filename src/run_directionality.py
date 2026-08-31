from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .config import ProbeConfig
from .directionality_sampler import directionality_generate
from .run_experiment import PROJECT, load_assets, prompt_for, seed_everything


def run(start: int, count: int, mode: str) -> dict:
    config = ProbeConfig()
    debug_gate = PROJECT / "results/debug_sanity.json"
    if not debug_gate.exists() or not json.loads(debug_gate.read_text())["all_passed"]:
        raise RuntimeError("Original probe sanity gate is absent or failed")
    if mode == "debug" and not (5 <= count <= 10):
        raise ValueError("Follow-up debug requires 5–10 samples")
    if mode == "formal":
        followup_gate = PROJECT / "results/directionality_debug_sanity.json"
        if not followup_gate.exists() or not json.loads(followup_gate.read_text())["all_passed"]:
            raise RuntimeError("Directionality formal run refused until its debug gate passes")
    if start < 0 or count <= 0 or start + count > config.samples:
        raise ValueError("Requested range must be within samples 0:200")
    seed_everything(config.seed)
    model, tokenizer, dataset = load_assets(config)
    special_ids = set(tokenizer.all_special_ids)
    output_dir = PROJECT / "traces/directionality_probe" / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses = []
    for sample_id in range(start, start + count):
        sample_path = output_dir / f"sample_{sample_id:04d}.json"
        if sample_path.exists():
            statuses.append(json.loads(sample_path.read_text())["sanity"])
            continue
        reference_path = PROJECT / "traces/mask_temporal_state_probe/formal" / f"sample_{sample_id:04d}.json"
        reference = json.loads(reference_path.read_text())
        if not reference["sanity"]["passed"] or not reference["sanity"]["official_equals_traced"]:
            raise RuntimeError(f"Original reference sample {sample_id} failed sanity")
        prompt, attention_mask, formatted = prompt_for(tokenizer, dataset[sample_id]["question"], model.device)
        reference_generated = torch.tensor(reference["official_output_token_ids"], dtype=torch.long)
        seed_everything(config.seed + sample_id)
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state().clone()
        output, records, sanity = directionality_generate(
            model, prompt, attention_mask, config,
            sample_id=sample_id,
            reference_generated=reference_generated,
            special_token_ids=special_ids,
            verify_vanilla_probe=mode == "debug",
        )
        sanity.update({
            "sample_id": sample_id,
            "cpu_rng_unchanged": bool(torch.equal(cpu_rng_before, torch.random.get_rng_state())),
            "cuda_rng_unchanged": bool(torch.equal(cuda_rng_before, torch.cuda.get_rng_state())),
            "shuffle_different_position": all(r["absolute_position"] != r["shuffle_source_position"] for r in records),
            "history_windows_valid": all(r["unresolved_steps"] >= config.history + 1 for r in records),
            "geometry_complete": all("geometry" in r for r in records),
        })
        sanity["passed"] = bool(
            sanity["reference_equals_followup_traced"]
            and (sanity["vanilla_probe_max_abs_logit_error"] in (None, 0.0))
            and sanity["random_shuffle_norm_max_relative_error"] <= 2e-6
            and sanity["projection_layers"] == 32
            and sanity["cpu_rng_unchanged"] and sanity["cuda_rng_unchanged"]
            and sanity["shuffle_different_position"] and sanity["history_windows_valid"]
            and sanity["geometry_complete"]
        )
        payload = {
            "sample_id": sample_id,
            "question": dataset[sample_id]["question"],
            "formatted_prompt": formatted,
            "reference_output_token_ids": reference_generated.tolist(),
            "followup_output_token_ids": output[0, prompt.shape[1]:].cpu().tolist(),
            "sanity": sanity,
            "observations": records,
        }
        temporary = sample_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, sample_path)
        statuses.append(sanity)
    aggregate = {
        "mode": mode,
        "start": start,
        "count": count,
        "completed": len(statuses),
        "all_passed": len(statuses) == count and all(status.get("passed", False) for status in statuses),
        "samples": statuses,
        "config": config.as_dict(),
    }
    if mode == "debug":
        destination = PROJECT / "results/directionality_debug_sanity.json"
    else:
        destination = PROJECT / "results" / f"directionality_status_{start:04d}_{start + count:04d}.json"
    destination.write_text(json.dumps(aggregate, indent=2) + "\n")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("debug", "formal"), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    count = args.count if args.count is not None else (5 if args.mode == "debug" else 200)
    result = run(args.start, count, args.mode)
    print(json.dumps({key: result[key] for key in ("mode", "start", "count", "all_passed")}, indent=2))


if __name__ == "__main__":
    main()
