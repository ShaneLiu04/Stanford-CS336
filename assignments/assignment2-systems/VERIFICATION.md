# Verification record

## Automated correctness

| Environment | Scope | Result |
| --- | --- | --- |
| Windows, PyTorch 2.11 CPU | Complete suite | 10 passed, 4 CUDA cases skipped |
| AutoDL RTX 6000D, CUDA 12.8 | PyTorch/Triton attention | 6 passed |
| AutoDL CPU/Gloo | DDP, FSDP, sharded optimizer | 8 passed |

Archived outputs:

- [`report/results/raw/logs/pytest-cuda.log`](report/results/raw/logs/pytest-cuda.log)
- [`report/results/raw/logs/pytest-distributed.log`](report/results/raw/logs/pytest-distributed.log)
- [`report/results/raw/test_summary.csv`](report/results/raw/test_summary.csv)

## Reproduction

```bash
uv sync
uv run pytest -v tests
uv run ruff check cs336_systems tests/adapters.py scripts
uv run python scripts/build_report_assets.py --results report/results
cd report && make
```

`report/results/manifest.sha256` verifies the archived raw metrics, figures,
tables and machine-readable summaries. Multi-GPU NCCL performance is not
claimed because the available AutoDL instance contains one GPU.

This is a **non-course-submission** self-study record by
[ShaneLiu04](https://github.com/ShaneLiu04).
