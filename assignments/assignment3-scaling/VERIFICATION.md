# Verification

- Official source: `stanford-cs336/assignment3-scaling`, SHA-256
  `da6a464c3f5e5039d3f20e137dfcab3031a3d8e7ef0ac2f376f96e295d4afd27`.
- Official analysis: 72 records, 9 compute tiers.
- Offline proxy: 22 completed RTX 6000D TinyStories runs across four compute
  tiers; no failed/OOM runs.
- No Stanford A3 API key was used and no `/final_submission` was made.
- PostgreSQL-dependent staff API tests are outside the offline analysis scope.

Validation commands:

```bash
uv run ruff check scripts
uv run python scripts/analyze_scaling.py
uv run python scripts/analyze_proxy.py
cd report && make
```

Non-course-submission study record by
[ShaneLiu04](https://github.com/ShaneLiu04).
