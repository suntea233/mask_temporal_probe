# Running the Unified Layer × Entropy × State Probe elsewhere

This repository is a portable snapshot of the probe code and completed small
reports. It deliberately excludes model weights, the local Conda environment,
and raw per-sample traces.

## Pinned inputs

- LLaDA repository commit: `9182493720ed723ef8031210d85959364e51cbe0`
- Model: `GSAI-ML/LLaDA-8B-Instruct`
- Model revision: `08b83a6feb34df1a6011b80c3c00c7563e963b07`
- GSM8K revision: `740312add88f781978c0658806c59bc2815b9866`
- Included Arrow SHA-256: `45965b000311d1550e5619b60b5bf31cf76edebfd8b8eddc62a876fbf8c9be95`
- Python: 3.10; reference run used PyTorch 2.7.1+cu128

The runner passes both `revision=...` and `local_files_only=True`. It cannot
silently use a newer checkpoint.

## 1. Clone

```bash
git clone --recurse-submodules git@github.com:suntea233/mask_temporal_probe.git
cd mask_temporal_probe
git submodule update --init --recursive
test "$(git -C vendor/LLaDA rev-parse HEAD)" = \
  9182493720ed723ef8031210d85959364e51cbe0
sha256sum data/gsm8k-test.arrow
```

If HTTPS is preferable, replace the clone URL with the HTTPS URL shown on the
GitHub repository page.

## 2. Create the environment

The environment is intentionally not committed. On a CUDA 12.8-compatible
server:

```bash
conda create -n dllm python=3.10 -y
conda activate dllm
python -m pip install -r requirements.txt
python -m pytest -q
```

The original machine used two A800 40GB cards, but the runner loads one
LLaDA-8B model on one visible GPU. A single GPU with roughly 24GB or more free
memory is recommended. `SCENARIO_BATCH_SIZE` controls only counterfactual
execution batching; lowering it after an OOM does not change the scientific
conditions.

## 3. Prepare the exact model revision

Download or copy the Hugging Face cache for exactly:

```text
GSAI-ML/LLaDA-8B-Instruct@08b83a6feb34df1a6011b80c3c00c7563e963b07
```

For example, on a machine with `huggingface-cli` and network access:

```bash
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct \
  --revision 08b83a6feb34df1a6011b80c3c00c7563e963b07
```

Do not upgrade the remote model implementation independently of the pinned
dependencies.

## 4. Run the mandatory debug gate

Choose one idle GPU and run 5 samples:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m src.run_unified_latent --mode debug --start 0 --count 5
python - <<'PY'
import json
gate = json.load(open("results/unified_latent_debug_sanity.json"))
assert gate["all_passed"], gate
print("debug gate passed")
PY
```

The gate checks trajectory equality, exact unchanged-hidden resume, temporal
provenance, same-block non-self shuffle, norm matching, target-only edits, hard
token isolation, and unchanged RNG state. Formal execution refuses to start if
this gate is absent or failed.

## 5. Run all 200 samples

The run is atomic and resumable: an existing completed `sample_XXXX.json` is
loaded rather than overwritten.

```bash
python -m src.run_unified_latent --mode formal --start 0 --count 200
python analysis/analyze_unified_latent.py
```

To use two GPUs, run non-overlapping shards from two shells after the shared
debug gate passes:

```bash
# shell 1
CUDA_VISIBLE_DEVICES=0 python -m src.run_unified_latent \
  --mode formal --start 0 --count 100

# shell 2
CUDA_VISIBLE_DEVICES=1 python -m src.run_unified_latent \
  --mode formal --start 100 --count 100

# after both complete
python analysis/analyze_unified_latent.py
```

Do not overlap sample ranges. The analysis refuses to run unless all 200
formal sample files exist and pass sanity.

## 6. Expected new outputs

```text
results/unified_latent_debug_sanity.json
results/unified_latent_status_*.json
results/report_unified_latent_state_probe.txt
results/unified_latent_state_probe_summary.json
traces/unified_latent_state_probe/{debug,formal}/sample_XXXX.json
figures/unified_latent_state_probe/01_layer_self_delta.png
figures/unified_latent_state_probe/02_previous_transitions.png
figures/unified_latent_state_probe/03_maturity_previous_curves.png
figures/unified_latent_state_probe/04_previous_heatmap.png
figures/unified_latent_state_probe/05_endpoint_heatmap.png
figures/unified_latent_state_probe/06_mask_latent_hard.png
figures/unified_latent_state_probe/07_preferred_state.png
```

Preserve `traces/` on the compute server or archive it separately; Git ignores
raw traces intentionally.

## Scientific guardrails

The vanilla trajectory is untouched. Previous, Early, Shuffle, Random,
Endpoint Oracle, and Hard are discarded counterfactual forwards. The endpoint
condition is future-information oracle analysis, not an inference method. The
experiment does not train parameters, tune a decoding rule, or implement a
final decoder.
