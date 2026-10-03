#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH=/root/.local/bin:$PATH
export UV_PYTHON=/root/autodl-tmp/python/bin/python3
export UV_PYTHON_DOWNLOADS=never
mkdir -p outputs/stage-logs
exec > >(tee -a outputs/stage-logs/lr-fine.log) 2>&1
uv run python scripts/run_sweep.py \
  --base-config configs/tinystories-experiment.json \
  --manifest configs/sweeps/tinystories-lr-fine.json \
  --output-root outputs/lr-fine
uv run python scripts/plot_metrics.py outputs/lr-fine/[0-9][0-9][0-9]-* \
  --output outputs/lr-fine/lr-fine.pdf
