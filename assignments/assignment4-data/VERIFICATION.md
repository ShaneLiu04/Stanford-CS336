# Verification

- Windows offline suite: **21 passed, 0 failed**.
- Controlled corpus: 400 documents across five quality/corruption groups.
- Filter matrix: 12 Gopher × safety × quality-threshold configurations.
- Report: 15 generated PNG/PDF figure groups.
- Modal 2500-WET and 8×B200 training were not run; no SUNET_ID was used.

```bash
python -m pytest -q tests
PYTHONPATH=. python scripts/run_offline_experiments.py
python scripts/build_report_assets.py
```

Non-course-submission self-study record by
[ShaneLiu04](https://github.com/ShaneLiu04).
