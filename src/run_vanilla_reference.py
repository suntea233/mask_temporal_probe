from __future__ import annotations

import argparse
import hashlib
import json
import os

from .config import ProbeConfig
from .run_experiment import PROJECT, load_assets, official_generate, prompt_for, seed_everything


def reference_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in (
        "src/config.py", "src/run_experiment.py", "src/run_vanilla_reference.py",
        "vendor/LLaDA/generate.py",
    ):
        path = PROJECT / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(start: int, count: int) -> dict:
    config = ProbeConfig()
    if start < 0 or count <= 0 or start + count > config.samples:
        raise ValueError("Requested range must be within samples 0:200")
    fingerprint = reference_fingerprint()
    seed_everything(config.seed)
    model, tokenizer, dataset = load_assets(config)
    output_dir = PROJECT / "traces/vanilla_reference/formal"
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses = []

    for sample_id in range(start, start + count):
        path = output_dir / f"sample_{sample_id:04d}.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if existing.get("reference_fingerprint") != fingerprint:
                raise RuntimeError(f"Existing reference sample {sample_id} was produced by different source")
            statuses.append(existing["sanity"])
            continue
        row = dataset[sample_id]
        prompt, attention_mask, formatted = prompt_for(tokenizer, row["question"], model.device)
        seed_everything(config.seed + sample_id)
        output = official_generate(
            model, prompt, attention_mask=attention_mask,
            steps=config.steps, gen_length=config.gen_length, block_length=config.block_length,
            temperature=config.temperature, cfg_scale=config.cfg_scale,
            remasking=config.remasking, mask_id=config.mask_id,
        )
        generated = output[0, prompt.shape[1]:].cpu()
        sanity = {
            "sample_id": sample_id,
            "passed": bool(generated.numel() == config.gen_length and not (generated == config.mask_id).any()),
            "generated_tokens": int(generated.numel()),
            "no_unresolved_mask": bool(not (generated == config.mask_id).any()),
        }
        payload = {
            "reference_fingerprint": fingerprint,
            "sample_id": sample_id,
            "question": row["question"],
            "formatted_prompt": formatted,
            "reference_answer": row["answer"],
            "official_output_token_ids": generated.tolist(),
            "decoded_output": tokenizer.decode(generated, skip_special_tokens=True),
            "sanity": sanity,
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, path)
        statuses.append(sanity)

    result = {
        "start": start, "count": count, "completed": len(statuses),
        "all_passed": len(statuses) == count and all(item.get("passed", False) for item in statuses),
        "reference_fingerprint": fingerprint, "config": config.as_dict(), "samples": statuses,
    }
    destination = PROJECT / "results" / f"vanilla_reference_status_{start:04d}_{start + count:04d}.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    result = run(args.start, args.count)
    print(json.dumps({key: result[key] for key in ("start", "count", "all_passed")}, indent=2))


if __name__ == "__main__":
    main()
