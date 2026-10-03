#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH=/root/.local/bin:$PATH
export UV_PYTHON=/root/autodl-tmp/python/bin/python3
export UV_PYTHON_DOWNLOADS=never

mkdir -p outputs/stage-logs
exec > >(tee -a outputs/stage-logs/complete-remaining.log) 2>&1

wait_for_artifact() {
  local artifact="$1"
  local process_pattern="$2"
  local label="$3"
  local deadline=$((SECONDS + 12 * 60 * 60))
  while [[ ! -s "$artifact" ]]; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for $label: $artifact" >&2
      return 1
    fi
    if ! pgrep -f "$process_pattern" >/dev/null; then
      echo "$label stopped without producing $artifact" >&2
      return 1
    fi
    echo "[$(date --iso-8601=seconds)] waiting for $label"
    sleep 60
  done
}

wait_for_gpu_idle() {
  while pgrep -f "scripts/train.py" >/dev/null; do
    echo "[$(date --iso-8601=seconds)] waiting for current GPU training job"
    sleep 60
  done
}

wait_for_artifact data/owt/metrics.json \
  "prepare_data.py.*data/raw/owt_train.txt" "OWT tokenizer"

if [[ ! -s outputs/tokenizer-experiments.json ]]; then
  uv run python scripts/tokenizer_experiments.py \
    --tinystories-text data/raw/TinyStoriesV2-GPT4-valid.txt \
    --owt-text data/raw/owt_valid.txt \
    --tinystories-tokenizer data/tinystories \
    --owt-tokenizer data/owt \
    --output outputs/tokenizer-experiments.json
fi

wait_for_artifact outputs/no-rmsnorm-full/summary.json \
  "train.py.*output_dir=outputs/no-rmsnorm-full" "full-budget no-RMSNorm"

wait_for_gpu_idle
rm -f outputs/owt-tiny-full/checkpoint-[0-9]*.pt
if [[ ! -s outputs/owt-lr-tuning/sweep-status.jsonl ]] ||
   [[ "$(wc -l < outputs/owt-lr-tuning/sweep-status.jsonl)" -lt 4 ]]; then
  uv run python scripts/run_sweep.py \
    --base-config configs/owt-experiment.json \
    --manifest configs/sweeps/owt-lr.json \
    --output-root outputs/owt-lr-tuning
fi

BEST_LR="$(
  uv run python - <<'PY'
import json
from pathlib import Path

best = min(
    (
        json.loads((path / "summary.json").read_text())["best_val_loss"],
        json.loads((path / "config.json").read_text())["max_lr"],
    )
    for path in Path("outputs/owt-lr-tuning").glob("[0-9][0-9][0-9]-*")
    if (path / "summary.json").exists()
)
print(best[1])
PY
)"
MIN_LR="$(uv run python -c "print(float('$BEST_LR') / 10)")"
echo "Selected OWT max_lr=$BEST_LR min_lr=$MIN_LR"

if [[ ! -s outputs/owt-final/summary.json ]]; then
  uv run python scripts/train.py --config configs/owt-experiment.json \
    --set max_lr="$BEST_LR" \
    --set min_lr="$MIN_LR"
fi

if [[ ! -s outputs/owt-final/generations.txt ]]; then
  : > outputs/owt-final/generations.txt
  for temperature in 0.7 1.0 1.3; do
    {
      echo "===== temperature=$temperature, top_p=0.95 ====="
      uv run python scripts/generate.py \
        --run-dir outputs/owt-final \
        --tokenizer-dir data/owt \
        --prompt "The development of language models" \
        --max-new-tokens 256 \
        --temperature "$temperature" \
        --top-p 0.95
      echo
    } >> outputs/owt-final/generations.txt
  done
fi
rm -f outputs/owt-final/checkpoint-[0-9]*.pt

LEADERBOARD_STEPS="$(
  uv run python - <<'PY'
import json
from pathlib import Path

best_path = min(
    (
        json.loads((path / "summary.json").read_text())["best_val_loss"],
        path,
    )
    for path in Path("outputs/owt-lr-tuning").glob("[0-9][0-9][0-9]-*")
    if (path / "summary.json").exists()
)[1]
summary = json.loads((best_path / "summary.json").read_text())
seconds_per_step = summary["elapsed_seconds"] / summary["steps"]
# Keep a five-minute margin under the 45-minute local wall-clock envelope.
print(max(1000, min(40000, int(2400 / seconds_per_step))))
PY
)"
echo "Local leaderboard proxy: tied embeddings, max_steps=$LEADERBOARD_STEPS"
if [[ ! -s outputs/owt-leaderboard-tied/summary.json ]]; then
  uv run python scripts/train.py --config configs/owt-experiment.json \
    --set output_dir=outputs/owt-leaderboard-tied \
    --set max_steps="$LEADERBOARD_STEPS" \
    --set warmup_steps=500 \
    --set max_lr="$BEST_LR" \
    --set min_lr="$MIN_LR" \
    --set tie_embeddings=true
fi
rm -f outputs/owt-leaderboard-tied/checkpoint-[0-9]*.pt

uv run python scripts/plot_metrics.py outputs/owt-final \
  --output outputs/owt-final/owt-training.pdf
uv run python scripts/plot_metrics.py outputs/owt-leaderboard-tied \
  --output outputs/owt-leaderboard-tied/leaderboard-training.pdf
echo "[$(date --iso-8601=seconds)] all remaining experiments complete"
