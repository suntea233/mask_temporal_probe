# Unresolved-MASK Temporal State Probe

This project is a phenomenon probe for whether the discarded K/V trajectory of an unresolved LLaDA `[MASK]` predicts the token chosen later by the untouched vanilla trajectory beyond current logits and historical logits. It does **not** implement caching, remasking changes, token editing, reveal-policy changes, beam search, training, or a decoding method.

## Fixed experiment

- Official repository: `ML-GSAI/LLaDA`, commit `9182493720ed723ef8031210d85959364e51cbe0`
- Model: `GSAI-ML/LLaDA-8B-Instruct`, revision `08b83a6feb34df1a6011b80c3c00c7563e963b07`
- Dataset: GSM8K `main` test, revision `740312add88f781978c0658806c59bc2815b9866`, samples 0–199
- Local test Arrow SHA-256: `45965b000311d1550e5619b60b5bf31cf76edebfd8b8eddc62a876fbf8c9be95`
- Generation: `steps=128`, `gen_length=256`, `block_length=32`, `temperature=0`, `cfg_scale=0`, `remasking=low_confidence`
- Probe: `H=4`, `alpha=0.25`, up to four active-block unresolved positions at approximately 25%, 50%, and 75%
- Probe schedule: steps 5, 8, and 12 of each 16-step block. Step 5 is the nearest legal point to 25% that has four completed previous forward states.
- K is captured after `k_proj` and before RoPE; V is captured after `v_proj`.

Histories reset when a block becomes active. This makes every window same-sample, same-active-block, same-absolute-position, and previous-step-only. The rolling buffers live only in memory. Raw JSON stores scalar/token trajectory summaries and outputs, never full K/V states.

At a probe, a second no-intervention forward checks vanilla logits exactly. Real, shuffled, and norm-matched random replacements are injected by hooks into the official checkpoint's projection modules. All conditions begin from a read-only copy of the same current token tensor. Random controls use private deterministic generators. Only the selected unresolved rows are replaced.

The future vanilla target is filled in only after the untouched trajectory completes. Historical-logit fusion uses exactly `t-H...t`; separately, observational prediction summaries continue from `t-H` through the position's vanilla reveal step so pre-reveal top-1 changes are measurable. Positions at or after the first generated EOS and positions whose eventual token is special are discarded.

## Environment

The isolated environment is `.conda/`, cloned from a compatible local package cache and pinned by `requirements.txt`. Recreate it on another server with a Python 3.10 environment and:

```bash
pip install -r requirements.txt
```

The model revision must be available in the Hugging Face cache; runners use offline/local-only loading to prevent revision drift. The exact cached GSM8K test Arrow artifact is vendored under `data/` because this server's shared dataset cache cannot create locks.

## Run order

```bash
scripts/capture_environment.sh
scripts/run_debug.sh 5
scripts/run_formal.sh
```

`run_debug.sh` first executes the official sampler and then the traced sampler for each sample. The formal runner refuses to start unless 5–10 debug samples pass every gate. Formal execution may use non-overlapping GPU shards, but analysis still refuses fewer than all 200 sample files.

The sanity gate checks official/traced token equality, exact vanilla probe logits, 32 projection layers, complete unresolved windows, same-block absolute positions, different shuffle positions, random norm matching, and unchanged CPU/CUDA RNG states. Sample files are written atomically and are resumable.

## Outputs

- Per-sample raw records: `traces/mask_temporal_state_probe/{debug,formal}/sample_XXXX.json`
- Environment: `results/environment.json`
- Debug gate: `results/debug_sanity.json`
- Formal report: `results/report_mask_temporal_state_probe.txt`
- Machine-readable summary: `results/mask_temporal_state_probe_summary.json`
- Exactly three main figures in `figures/`

The analysis uses sample-level clustered bootstrap resampling (10,000 draws), never position-level IID confidence intervals. Decision A requires the lower 95% bootstrap bound to be positive for Real vs Vanilla, Real vs Shuffle, Real vs Random, and Real vs Logit History. Decision B requires a positive Logit History lower bound when A is not met. Otherwise it chooses C.

## Completed run

The experiment completed on two NVIDIA A800-SXM4-40GB GPUs using the `dllm` Conda environment. Five debug samples passed before the formal run. The 200 formal samples were executed as two non-overlapping GPU shards; all samples passed sanity, and every official output exactly matched its traced output. See the report and JSON summary for the metrics and decision.

## Follow-up mechanism diagnostics

Two fixed, no-training follow-ups reuse the same official vanilla trajectories:

- Temporal directionality/layer localization: `results/report_directionality_probe.txt`
- Common-mode vs position-specific dynamics: `results/report_common_mode_probe.txt`
- MASK endpoint geometry and reveal transition: `results/report_mask_endpoint_geometry.txt`

The common-mode probe estimates block-wide velocity from every eligible unresolved MASK, uses leave-one-out means for causal targets, and separates full, common, real-residual, shuffled-residual, and random-residual interventions. Its 5-sample debug gate exactly reproduced the earlier simultaneous-target forward-velocity condition, and all 200 formal samples preserved the reference output. Raw records are under `traces/common_mode_probe/`; its machine-readable summary is `results/common_mode_probe_summary.json`, with exactly three figures under `figures/common_mode/`.

The endpoint probe is read-only: it captures `H` as the post-block residual-stream output and K/V at the existing pre-RoPE projection coordinates. It tracks each generated position's first MASK state, final pre-reveal MASK state, and first post-reveal token state, then stores only scalar endpoint and trajectory geometry. Its raw records are under `traces/mask_endpoint_geometry/`; results are `results/report_mask_endpoint_geometry.txt`, `results/mask_endpoint_geometry_summary.json`, and five figures under `figures/mask_endpoint_geometry/`.

## Current experiment: Unified Layer × Entropy × State Probe

The current experiment probes layers `{20, 24, 26, 28, 31}` and jointly asks
where a latent state can be consumed, when its utility depends on maturity, and
how MASK/LATENT/HARD affect other unresolved positions. It includes direct
Previous Carry, Early, same-block Shuffled Previous, norm-matched Random, a
strict future Endpoint Oracle, and an independent Hard input-token condition.
No counterfactual changes the vanilla trajectory.

Implementation is in `src/unified_latent_sampler.py` and
`src/run_unified_latent.py`; analysis is in
`analysis/analyze_unified_latent.py`. The mandatory debug gate has not yet been
run on this source snapshot because both GPUs on the original server were
occupied by an unrelated job. Do not start the formal run until the new
server's 5-sample gate passes.

See [RUN_UNIFIED_PROBE.md](RUN_UNIFIED_PROBE.md) for the exact cross-server
setup, resumable run commands, two-GPU sharding, expected outputs, and
scientific guardrails.
