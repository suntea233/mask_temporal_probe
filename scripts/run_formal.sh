#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.conda/bin/python -m src.run_experiment --mode formal --start 0 --count 200
.conda/bin/python analysis/analyze.py
