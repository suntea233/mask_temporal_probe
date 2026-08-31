from __future__ import annotations

import argparse
import json
import os

import torch

from .config import ProbeConfig
from .endpoint_geometry_sampler import endpoint_geometry_generate
from .run_experiment import PROJECT, load_assets, prompt_for, seed_everything


def run(start: int, count: int, mode: str) -> dict:
    config = ProbeConfig()
    if mode == "debug" and not 5 <= count <= 10:
        raise ValueError("Endpoint geometry debug requires 5–10 samples")
    if mode == "formal":
        gate = PROJECT / "results/endpoint_geometry_debug_sanity.json"
        if not gate.exists() or not json.loads(gate.read_text())["all_passed"]:
            raise RuntimeError("Formal endpoint run refused until its debug gate passes")
    if start < 0 or count <= 0 or start + count > config.samples:
        raise ValueError("Requested range must be within samples 0:200")
    seed_everything(config.seed)
    model, tokenizer, dataset = load_assets(config)
    special_ids = set(tokenizer.all_special_ids)
    output_dir = PROJECT / "traces/mask_endpoint_geometry" / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses = []
    for sample_id in range(start, start + count):
        path = output_dir / f"sample_{sample_id:04d}.json"
        if path.exists():
            statuses.append(json.loads(path.read_text())["sanity"]); continue
        reference = json.loads((PROJECT / "traces/mask_temporal_state_probe/formal" / f"sample_{sample_id:04d}.json").read_text())
        if not reference["sanity"]["passed"] or not reference["sanity"]["official_equals_traced"]:
            raise RuntimeError(f"Reference sample {sample_id} failed")
        prompt, attention_mask, formatted = prompt_for(tokenizer, dataset[sample_id]["question"], model.device)
        reference_generated = torch.tensor(reference["official_output_token_ids"], dtype=torch.long)
        seed_everything(config.seed + sample_id)
        cpu_before = torch.random.get_rng_state().clone(); cuda_before = torch.cuda.get_rng_state().clone()
        output, records, sanity = endpoint_geometry_generate(
            model, prompt, attention_mask, config, sample_id=sample_id,
            reference_generated=reference_generated, special_token_ids=special_ids,
        )
        sanity.update({
            "sample_id": sample_id,
            "cpu_rng_unchanged": bool(torch.equal(cpu_before, torch.random.get_rng_state())),
            "cuda_rng_unchanged": bool(torch.equal(cuda_before, torch.cuda.get_rng_state())),
            "absolute_positions_stay_in_block": all(
                prompt.shape[1] + r["block_index"] * config.block_length <= r["absolute_position"] < prompt.shape[1] + (r["block_index"] + 1) * config.block_length
                for r in records
            ),
            "shuffle_sources_in_same_block": all(
                prompt.shape[1] + r["block_index"] * config.block_length <= r["shuffle_source_position"] < prompt.shape[1] + (r["block_index"] + 1) * config.block_length
                for r in records
            ),
        })
        sanity["passed"] = bool(
            sanity["reference_equals_endpoint_traced"] and sanity["projection_layers"] == 32
            and sanity["early_mask"] and sanity["pre_mask"] and sanity["post_token"]
            and sanity["pre_is_reveal_forward"] and sanity["post_is_first_later_forward"]
            and sanity["shuffle_same_sample_block_different_position"]
            and sanity["cpu_rng_unchanged"] and sanity["cuda_rng_unchanged"]
            and sanity["absolute_positions_stay_in_block"] and sanity["shuffle_sources_in_same_block"]
        )
        payload = {
            "sample_id": sample_id, "question": dataset[sample_id]["question"], "formatted_prompt": formatted,
            "reference_answer": reference["reference_answer"], "decoded_output": reference["decoded_output"],
            "reference_output_token_ids": reference_generated.tolist(),
            "endpoint_output_token_ids": output[0, prompt.shape[1]:].cpu().tolist(),
            "sanity": sanity, "positions": records,
        }
        temporary = path.with_suffix(".json.tmp"); temporary.write_text(json.dumps(payload), encoding="utf-8"); os.replace(temporary, path)
        statuses.append(sanity)
    result = {
        "mode": mode, "start": start, "count": count, "completed": len(statuses),
        "all_passed": len(statuses) == count and all(s.get("passed", False) for s in statuses),
        "samples": statuses, "config": config.as_dict(),
    }
    destination = PROJECT / "results" / ("endpoint_geometry_debug_sanity.json" if mode == "debug" else f"endpoint_geometry_status_{start:04d}_{start + count:04d}.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("debug", "formal"), required=True)
    parser.add_argument("--start", type=int, default=0); parser.add_argument("--count", type=int)
    args = parser.parse_args(); count = args.count if args.count is not None else (5 if args.mode == "debug" else 200)
    result = run(args.start, count, args.mode)
    print(json.dumps({k: result[k] for k in ("mode", "start", "count", "all_passed")}, indent=2))


if __name__ == "__main__": main()
