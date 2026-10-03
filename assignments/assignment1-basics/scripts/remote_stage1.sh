#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH=/root/.local/bin:$PATH
export UV_PYTHON=/root/autodl-tmp/python/bin/python3
export UV_PYTHON_DOWNLOADS=never

mkdir -p outputs/stage-logs
exec > >(tee -a outputs/stage-logs/stage1.log) 2>&1

echo "[$(date --iso-8601=seconds)] stage1: TinyStories tokenizer"
uv run python scripts/prepare_data.py \
  --train-text data/raw/TinyStoriesV2-GPT4-train.txt \
  --valid-text data/raw/TinyStoriesV2-GPT4-valid.txt \
  --output-dir data/tinystories \
  --vocab-size 10000 \
  --workers 16

echo "[$(date --iso-8601=seconds)] stage1: OpenWebText tokenizer"
uv run python scripts/prepare_data.py \
  --train-text data/raw/owt_train.txt \
  --valid-text data/raw/owt_valid.txt \
  --output-dir data/owt \
  --vocab-size 32000 \
  --workers 16

echo "[$(date --iso-8601=seconds)] stage1: GPU pilot"
uv run python scripts/train.py \
  --config configs/tinystories-experiment.json \
  --set output_dir=outputs/pilot \
  --set max_steps=100 \
  --set eval_interval=20 \
  --set compile_model=false \
  --set save_checkpoints=false

echo "[$(date --iso-8601=seconds)] stage1: LR and batch tuning"
uv run python scripts/run_sweep.py \
  --base-config configs/tinystories-experiment.json \
  --manifest configs/sweeps/tinystories-tuning.json \
  --output-root outputs/tuning

uv run python scripts/plot_metrics.py outputs/tuning/[0-9][0-9][0-9]-* \
  --output outputs/tuning/tuning.pdf
echo "[$(date --iso-8601=seconds)] stage1 complete"
