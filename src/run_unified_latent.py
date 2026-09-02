from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch

from .config import ProbeConfig
from .run_experiment import PROJECT, load_assets, prompt_for, seed_everything
from .unified_latent_sampler import unified_latent_generate


def source_fingerprint() -> str:
    """Bind resumable traces and the debug gate to this probe implementation."""
    digest = hashlib.sha256()
    for relative in (
        "src/config.py", "src/projection.py", "src/unified_latent_sampler.py",
        "src/run_unified_latent.py",
    ):
        path = PROJECT / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def reference_path(sample_id: int):
    original = PROJECT / "traces/mask_temporal_state_probe/formal" / f"sample_{sample_id:04d}.json"
    if original.exists():
        return original
    vanilla = PROJECT / "traces/vanilla_reference/formal" / f"sample_{sample_id:04d}.json"
    if vanilla.exists():
        return vanilla
    raise RuntimeError(
        f"Reference sample {sample_id} is absent; run src.run_vanilla_reference first"
    )


def run(start: int, count: int, mode: str) -> dict:
    config = ProbeConfig()
    fingerprint = source_fingerprint()
    if mode == "debug" and not 5 <= count <= 10:
        raise ValueError("Unified latent debug requires 5–10 samples")
    if mode == "formal":
        gate = PROJECT / "results/unified_latent_debug_sanity.json"
        gate_payload = json.loads(gate.read_text()) if gate.exists() else {}
        if not gate_payload.get("all_passed") or gate_payload.get("source_fingerprint") != fingerprint:
            raise RuntimeError("Formal unified latent run refused until debug passes")
    if start < 0 or count <= 0 or start + count > config.samples:
        raise ValueError("Requested range must be within samples 0:200")
    seed_everything(config.seed); model, tokenizer, dataset = load_assets(config); special_ids = set(tokenizer.all_special_ids)
    output_dir = PROJECT / "traces/unified_latent_state_probe" / mode; output_dir.mkdir(parents=True, exist_ok=True)
    statuses = []
    for sample_id in range(start, start + count):
        path = output_dir / f"sample_{sample_id:04d}.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if existing.get("source_fingerprint") != fingerprint:
                raise RuntimeError(f"Existing sample {sample_id} was produced by different probe source")
            statuses.append(existing["sanity"]); continue
        reference = json.loads(reference_path(sample_id).read_text())
        if not reference["sanity"]["passed"]: raise RuntimeError(f"Reference sample {sample_id} failed")
        prompt, attention_mask, formatted = prompt_for(tokenizer, dataset[sample_id]["question"], model.device)
        reference_generated = torch.tensor(reference["official_output_token_ids"], dtype=torch.long)
        seed_everything(config.seed + sample_id); cpu_before = torch.random.get_rng_state().clone(); cuda_before = torch.cuda.get_rng_state().clone()
        output, records, sanity = unified_latent_generate(
            model, prompt, attention_mask, config, sample_id=sample_id,
            reference_generated=reference_generated, special_token_ids=special_ids, verify_resume=mode == "debug",
        )
        sanity.update({
            "sample_id": sample_id, "cpu_rng_unchanged": bool(torch.equal(cpu_before, torch.random.get_rng_state())),
            "cuda_rng_unchanged": bool(torch.equal(cuda_before, torch.cuda.get_rng_state())),
            "observations_nonempty": len(records) > 0,
            "all_conditions_complete": all(
                all(all(condition in r["layers"][str(layer)] for condition in ("previous","early","shuffle","random","endpoint")) for layer in (20,24,26,28,31))
                and r["hard_downstream_gain"] is not None for r in records
            ),
        })
        sanity["passed"] = bool(
            sanity["reference_equals_unified_traced"] and sanity["projection_layers"] == 32
            and sanity["resume_current_max_abs_logit_error"] in (None, 0.0)
            and sanity["random_norm_max_relative_error"] <= 2e-6
            and sanity["previous_same_position_step_layer"] and sanity["early_is_first_unresolved"]
            and sanity["endpoint_same_position_layer_pre_reveal"] and sanity["shuffle_same_block_different_position"]
            and sanity["only_target_modified"] and sanity["hard_only_target_token_changed"]
            and sanity["downstream_targets_nonempty"] and sanity["observations_nonempty"]
            and (mode != "debug" or sanity["resume_checked"])
            and sanity["cpu_rng_unchanged"] and sanity["cuda_rng_unchanged"] and sanity["all_conditions_complete"]
        )
        payload = {
            "source_fingerprint": fingerprint,
            "sample_id": sample_id, "question": dataset[sample_id]["question"], "formatted_prompt": formatted,
            "reference_answer": reference["reference_answer"], "decoded_output": reference["decoded_output"],
            "reference_output_token_ids": reference_generated.tolist(), "unified_output_token_ids": output[0, prompt.shape[1]:].cpu().tolist(),
            "sanity": sanity, "observations": records,
        }
        temporary = path.with_suffix(".json.tmp"); temporary.write_text(json.dumps(payload), encoding="utf-8"); os.replace(temporary, path)
        statuses.append(sanity)
    result = {"mode":mode,"start":start,"count":count,"completed":len(statuses),"all_passed":len(statuses)==count and all(s.get("passed",False) for s in statuses),"samples":statuses,"config":config.as_dict(),"source_fingerprint":fingerprint}
    destination = PROJECT / "results" / ("unified_latent_debug_sanity.json" if mode == "debug" else f"unified_latent_status_{start:04d}_{start+count:04d}.json")
    destination.write_text(json.dumps(result,indent=2)+"\n"); return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=("debug","formal"),required=True); parser.add_argument("--start",type=int,default=0); parser.add_argument("--count",type=int)
    args=parser.parse_args(); count=args.count if args.count is not None else (5 if args.mode=="debug" else 200)
    result=run(args.start,count,args.mode); print(json.dumps({k:result[k] for k in ("mode","start","count","all_passed")},indent=2))


if __name__ == "__main__": main()
