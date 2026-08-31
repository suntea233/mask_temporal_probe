from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer

from .config import ProbeConfig
from .temporal_sampler import traced_generate


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "vendor/LLaDA"))
from generate import generate as official_generate  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_assets(config: ProbeConfig):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required: the formal 8B experiment is not permitted to fall back to CPU")
    model = AutoModel.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=True,
        local_files_only=True,
    )
    # Exact cached Arrow artifact from the pinned GSM8K revision. Loading the
    # shared HF cache directly would try to create a lock in a read-only path.
    dataset = Dataset.from_file(str(PROJECT / "data/gsm8k-test.arrow"))
    return model, tokenizer, dataset


def prompt_for(tokenizer, question: str, device: torch.device):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        tokenize=False,
    )
    encoded = tokenizer(text, return_tensors="pt")
    return encoded.input_ids.to(device), encoded.attention_mask.to(device), text


def run(start: int, count: int, mode: str) -> dict:
    config = ProbeConfig()
    if mode == "debug" and not (5 <= count <= 10):
        raise ValueError("Debug gate must use 5–10 samples")
    if mode == "formal":
        gate_path = PROJECT / "results/debug_sanity.json"
        if not gate_path.exists() or not json.loads(gate_path.read_text())["all_passed"]:
            raise RuntimeError("Formal run refused: debug_sanity.json is absent or failed")
        if start < 0 or count <= 0 or start + count > config.samples:
            raise ValueError("Formal shard must be a non-empty subset of samples 0:200")
    seed_everything(config.seed)
    model, tokenizer, dataset = load_assets(config)
    output_dir = PROJECT / "traces/mask_temporal_state_probe" / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses = []
    special_ids = set(tokenizer.all_special_ids)

    for sample_id in range(start, start + count):
        sample_path = output_dir / f"sample_{sample_id:04d}.json"
        if sample_path.exists():
            statuses.append(json.loads(sample_path.read_text())["sanity"])
            continue
        row = dataset[sample_id]
        prompt, attention_mask, formatted = prompt_for(tokenizer, row["question"], model.device)
        seed_everything(config.seed + sample_id)
        official = official_generate(
            model, prompt, attention_mask=attention_mask,
            steps=config.steps, gen_length=config.gen_length, block_length=config.block_length,
            temperature=config.temperature, cfg_scale=config.cfg_scale,
            remasking=config.remasking, mask_id=config.mask_id,
        )
        rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state().clone()
        traced, records, sanity = traced_generate(
            model, prompt, attention_mask, config, sample_id=sample_id,
            special_token_ids=special_ids,
        )
        rng_after = torch.random.get_rng_state()
        cuda_rng_after = torch.cuda.get_rng_state()
        sanity.update({
            "sample_id": sample_id,
            "official_equals_traced": bool(torch.equal(official, traced)),
            "cpu_rng_unchanged_by_traced_probe": bool(torch.equal(rng_before, rng_after)),
            "cuda_rng_unchanged_by_traced_probe": bool(torch.equal(cuda_rng_before, cuda_rng_after)),
            "history_windows_valid": all(r["unresolved_steps"] >= config.history + 1 for r in records),
            "shuffle_different_position": all(r["absolute_position"] != r["shuffle_source_position"] for r in records),
            "same_block_positions": all(
                prompt.shape[1] + r["block_index"] * config.block_length <= r["absolute_position"]
                < prompt.shape[1] + (r["block_index"] + 1) * config.block_length for r in records
            ),
        })
        sanity["passed"] = bool(
            sanity["official_equals_traced"]
            and sanity["vanilla_probe_max_abs_logit_error"] == 0.0
            and sanity["random_norm_max_abs_error"] <= 1e-4
            and sanity["cpu_rng_unchanged_by_traced_probe"]
            and sanity["cuda_rng_unchanged_by_traced_probe"]
            and sanity["history_windows_valid"]
            and sanity["shuffle_different_position"]
            and sanity["same_block_positions"]
            and sanity["projection_layers"] == 32
        )
        payload = {
            "sample_id": sample_id,
            "question": row["question"],
            "formatted_prompt": formatted,
            "reference_answer": row["answer"],
            "official_output_token_ids": official[0, prompt.shape[1]:].cpu().tolist(),
            "traced_output_token_ids": traced[0, prompt.shape[1]:].cpu().tolist(),
            "decoded_output": tokenizer.decode(traced[0, prompt.shape[1]:], skip_special_tokens=True),
            "sanity": sanity,
            "observations": records,
        }
        temporary = sample_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, sample_path)
        statuses.append(sanity)

    aggregate = {
        "mode": mode,
        "requested_samples": list(range(start, start + count)),
        "completed": len(statuses),
        "all_passed": len(statuses) == count and all(s.get("passed", False) for s in statuses),
        "samples": statuses,
        "config": config.as_dict(),
    }
    if mode == "debug":
        destination = PROJECT / "results/debug_sanity.json"
    elif start == 0 and count == config.samples:
        destination = PROJECT / "results/formal_run_status.json"
    else:
        destination = PROJECT / "results" / f"formal_run_status_{start:04d}_{start + count:04d}.json"
    destination.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["debug", "formal"], required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    count = args.count if args.count is not None else (5 if args.mode == "debug" else 200)
    result = run(args.start, count, args.mode)
    print(json.dumps({k: result[k] for k in ("mode", "completed", "all_passed")}, indent=2))


if __name__ == "__main__":
    main()
