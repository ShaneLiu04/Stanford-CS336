# Verification

Date: 2026-09-27

## Unit tests

- 23 official model, numerical utility, optimizer, serialization, data and BPE tests pass on Windows.
- One extension test verifies optional embedding weight tying and reduced parameter count.
- 23 tokenizer correctness/parity tests passed with Python UTF-8 mode.
- Two tokenizer RSS-limit tests are Linux-only and were skipped locally.
- BPE speed, deterministic merges, special-token boundaries, model snapshots and checkpoint round trips pass.

Commands:

```powershell
$env:PYTHONUTF8 = "1"
uv run pytest tests/test_model.py tests/test_nn_utils.py tests/test_optimizer.py `
  tests/test_serialization.py tests/test_data.py tests/test_train_bpe.py -q

# Windows has no stdlib `resource`; inject an empty module because the two
# resource-dependent tests are platform-skipped.
uv run python -c "import sys,types,pytest; sys.modules['resource']=types.ModuleType('resource'); raise SystemExit(pytest.main(['tests/test_tokenizer.py','-q']))"
```

## Pipeline checks

- Three-step CPU smoke training completed with validation, JSONL metrics and checkpoint writes.
- TinyStories 5 MiB sample BPE/tokenizer benchmark is recorded in `report/results/sample_bpe.json`.
- `scripts/build_report_assets.py` reconstructs 27 report figures and a 75-run experiment log from raw JSONL.
- The archived Linux test log records 47 passed and 1 xpassed.

## Archived GPU experiments

- RTX 4080 SUPER: 50/50 original sweep and standalone runs completed.
- RTX 6000D: 25/25 extension runs completed (batch capacity, compile ×
  precision, microbatch equivalence, optimizer dynamics, batch-512 and tied
  multi-seed full-budget runs).
- TinyStories batch-32 three-seed best validation loss: 1.3711 ± 0.0024.
- Full-budget batch 128 / 256 best validation loss: 1.3313 / 1.3247.
- Full-budget no-RMSNorm best validation loss: 7.512.
- OWT 32K tokenizer: 10,695 s BPE time, 46.45 GiB peak RSS, 4.371 bytes/token.
- OWT main / tied local proxy best validation loss: 4.116 / 4.097.
- RTX 6000D BF16 compile/eager throughput: 396K / 197K tokens/s.
- Batch-512 three-seed loss: 1.335 ± 0.002; tied full-budget three-seed loss:
  1.357 ± 0.005.
- The 200-document tokenizer comparison includes bootstrap 95% confidence
  intervals; the generation panel uses four checkpoints and fixed prompts.
- Raw configs, environments, metrics, summaries, and generated samples are under `report/results/raw`.

## Remaining limitations

The company proxy required Windows SSPI/NTLM authentication before opening
the SSH CONNECT tunnel. All important non-checkpoint artifacts were synchronized
after the remote pipeline emitted `all remaining experiments complete`; no
credentials were written to the repository.

- The 38.96-minute tied run is a local RTX 4080 SUPER proxy, not an official
  B200 leaderboard submission or ranking.
- The TinyStories tokenizer JSON contains `bpe_training_seconds: 0.0`; this
  missing timing is explicitly not interpreted as a real measurement.
