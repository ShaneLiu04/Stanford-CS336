#!/usr/bin/env bash
set -euo pipefail

# Run from assignment2-systems on a Linux CUDA host. Credentials are deliberately
# absent: transfer this repository using SSH/Git credential helpers.
export PATH="${HOME}/.local/bin:${PATH}"
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-never}"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user uv
fi

mkdir -p report/results/raw report/results/figures report/results/tables profiles
mkdir -p report/results/raw/logs

{
  date -u +"timestamp_utc=%Y-%m-%dT%H:%M:%SZ"
  git rev-parse HEAD
  uname -a
  python3 --version
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} > report/results/raw/environment.txt

uv sync
uv run pytest -v tests 2>&1 | tee report/results/raw/logs/pytest.log

# Operator sweep first: it is inexpensive and provides the most direct
# comparison between standard attention and the assignment implementations.
uv run python scripts/benchmark_systems.py attention \
  --batch-size 1 \
  --sequence-lengths 128 256 512 1024 2048 4096 8192 16384 \
  --head-dimensions 16 32 64 128 \
  --dtypes fp32 bf16 \
  --mode forward_backward \
  --output report/results/raw/benchmark.csv

# End-to-end runs are ordered by size. OOM is recorded rather than aborting.
for model in small medium large xl; do
  uv run python scripts/benchmark_systems.py model \
    --model-sizes "${model}" \
    --batch-size 4 \
    --sequence-lengths 512 \
    --dtypes fp32 bf16 \
    --mode forward_backward \
    --output report/results/raw/benchmark.csv
done

uv run python scripts/benchmark_correctness.py
uv run python scripts/benchmark_checkpointing.py \
  --model-size large \
  --batch-size 2 \
  --sequence-lengths 512 1024 2048 \
  --segments 0 4 12 36
uv run python scripts/build_report_assets.py --results report/results
