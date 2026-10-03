# Initial verification

Date: 2026-09-26

## Passed

- `scripts/check-repo.ps1`: no nested Git repositories, files over 50 MiB, common secret filenames, or unaccounted missing upstream licenses.
- Python 3.12 `compileall`: all imported `.py` files compile.
- Python 3.12 `tomllib`: all 12 imported TOML files parse.
- Secret-pattern scan: no common GitHub/OpenAI token prefixes, private-key headers, or literal password assignments found.
- Import inventory: nine licensed official snapshots plus one source-only Spring 2026 Assignment 5 notice.

## Expected warnings

`compileall` reports invalid-escape `SyntaxWarning`s in upstream Spring 2025 Assignment 1 adapters and Assignment 5 grader regex/docstrings. They do not prevent compilation and are retained unchanged so the official snapshot remains auditable.

## Not run

Full assignment tests were not run because this machine currently has Python 3.12 but neither `uv` nor `pytest` installed. The official student templates intentionally contain `NotImplementedError` placeholders, so many tests are expected to fail before implementation. Each assignment should be tested in its own `uv` environment as work begins.
