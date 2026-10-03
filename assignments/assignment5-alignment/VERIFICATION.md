# Verification

- Main GRPO suite: 19 passed.
- SFT packing / metrics / DPO supplement: 7 passed.
- Total: 26 passed, 0 failed.
- RTX 6000D proxy: 7 objectives × 4 seeds × 160 steps = 4480 metric rows.
- No SUNET_ID, Modal job, OLMo leaderboard, or 70B judge was used.

```bash
python -m pytest -q tests/test_grpo.py tests/test_data.py tests/test_metrics.py tests/test_dpo.py
python scripts/run_proxy_alignment.py
python scripts/build_report_assets.py
```

Non-course-submission study by
[ShaneLiu04](https://github.com/ShaneLiu04).
