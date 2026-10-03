#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH=/root/.local/bin:$PATH
export UV_PYTHON=/root/autodl-tmp/python/bin/python3
export UV_PYTHON_DOWNLOADS=never
mkdir -p outputs/stage-logs
exec > >(tee -a outputs/stage-logs/large-batch.log) 2>&1
uv run python scripts/run_sweep.py \
  --base-config configs/tinystories-experiment.json \
  --manifest configs/sweeps/tinystories-large-batch-final.json \
  --output-root outputs/large-batch
