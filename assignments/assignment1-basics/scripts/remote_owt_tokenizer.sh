#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH=/root/.local/bin:$PATH
export UV_PYTHON=/root/autodl-tmp/python/bin/python3
export UV_PYTHON_DOWNLOADS=never

mkdir -p outputs/stage-logs
exec > >(tee -a outputs/stage-logs/owt-tokenizer.log) 2>&1
echo "[$(date --iso-8601=seconds)] OWT tokenizer start"
uv run python scripts/prepare_data.py \
  --train-text data/raw/owt_train.txt \
  --valid-text data/raw/owt_valid.txt \
  --output-dir data/owt \
  --vocab-size 32000 \
  --bpe-workers 16 \
  --bpe-chunks 256 \
  --workers 16
echo "[$(date --iso-8601=seconds)] OWT tokenizer complete"
