#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH=/root/.local/bin:$PATH
export UV_PYTHON=/root/autodl-tmp/python/bin/python3
export UV_PYTHON_DOWNLOADS=never

RUN=outputs/final-seeds/000-baseline-seed-42
OUT=outputs/final-seeds/generations.txt
: > "$OUT"
for temperature in 0.7 1.0 1.3; do
  {
    echo "===== temperature=$temperature, top_p=0.95 ====="
    uv run python scripts/generate.py \
      --run-dir "$RUN" \
      --tokenizer-dir data/tinystories \
      --prompt "Once upon a time, in a small village" \
      --max-new-tokens 256 \
      --temperature "$temperature" \
      --top-p 0.95
    echo
  } >> "$OUT"
done
