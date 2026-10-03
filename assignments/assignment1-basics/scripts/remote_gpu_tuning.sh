#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH=/root/.local/bin:$PATH
export UV_PYTHON=/root/autodl-tmp/python/bin/python3
export UV_PYTHON_DOWNLOADS=never

mkdir -p outputs/stage-logs
exec > >(tee -a outputs/stage-logs/gpu-tuning.log) 2>&1

echo "[$(date --iso-8601=seconds)] GPU pilot"
uv run python scripts/train.py \
  --config configs/tinystories-experiment.json \
  --set output_dir=outputs/pilot \
  --set max_steps=100 \
  --set eval_interval=20 \
  --set save_checkpoints=false

echo "[$(date --iso-8601=seconds)] LR and batch tuning"
uv run python scripts/run_sweep.py \
  --base-config configs/tinystories-experiment.json \
  --manifest configs/sweeps/tinystories-tuning.json \
  --output-root outputs/tuning

uv run python scripts/plot_metrics.py outputs/tuning/[0-9][0-9][0-9]-* \
  --output outputs/tuning/tuning.pdf
echo "[$(date --iso-8601=seconds)] GPU tuning complete"
