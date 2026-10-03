#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
export PYTHONPATH="$ROOT"
mkdir -p outputs/report-extension/logs

run_group() {
  local name="$1"
  local manifest="$2"
  local output="$3"
  echo "[$(date --iso-8601=seconds)] starting ${name}"
  "$PYTHON" scripts/run_sweep.py \
    --base-config configs/tinystories-experiment.json \
    --manifest "$manifest" \
    --output-root "$output" 2>&1 | tee "outputs/report-extension/logs/${name}.log"
  echo "[$(date --iso-8601=seconds)] completed ${name}"
}

{
  date --iso-8601=seconds
  git rev-parse HEAD 2>/dev/null || true
  uname -a
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  "$PYTHON" -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda}")'
} > outputs/report-extension/environment.txt

run_group batch-probe configs/sweeps/rtx6000d-batch-probe.json outputs/rtx6000d-batch-probe
run_group compile-precision configs/sweeps/compile-precision.json outputs/compile-precision
run_group optimizer-gradient configs/sweeps/optimizer-gradient.json outputs/optimizer-gradient
run_group microbatch-equivalence configs/sweeps/microbatch-equivalence.json outputs/microbatch-equivalence
run_group batch512-full configs/sweeps/batch512-full.json outputs/batch512-full
run_group tied-full configs/sweeps/tied-full.json outputs/tied-full

echo "[$(date --iso-8601=seconds)] all report-extension experiments complete"
