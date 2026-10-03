# Changelog

All changes we make to the assignment code or PDF will be documented in this file.

## [unreleased] - yyyy-mm-dd

### Added

### Changed

### Fixed

## [2.0.1] - 2026-06-03

### Added
- both: additional prompt files and writeup clarifications to clarify prompting
- both: modal utils and shared volume with safety models and datasets

### Changed

### Fixed
- both: updated alpaca eval command to work with new package versions; downloaded gpt4-turbo reference data

## [2.0.0] - 2026-05-20

### Added
- both: added MaxRL, GSPO
- both: added few-shot r1_zero prompt
- both: 4 random seeds (up from 1 random seed)
- handout: added math and word problems

### Changed
- handout: changed from Qwen-2.5-Math-1.5B on MATH to Olmo-2-1B on GSM8K

### Fixed
- code: Updated package versions and code to work with B200s (vllm_utils.py)

## [1.0.1] - 2025-05-31

### Added
- code: for optional assignment, update alpaca_eval to Llama 3.3 70B Instruct judge
- handout: add optional assignment on safety, instruction tuning, and RLHF

### Changed

### Fixed
- code: change masked normalize constant to not equal seqlen, more SFT test coverage

## [1.0.0] - 2025-05-23

### Added
- code: 2025 assignment on SFT, Expert Iteration, and GRPO with verified rewards on MATH
- handout: 2025 assignment on SFT, Expert Iteration, and GRPO with verified rewards on MATH

### Changed
- N/A

### Fixed
- N/A

## [0.0.3] - 2024-06-05

### Added

### Changed

- code: fix AlpacaEval auto-evaluator to use local Llama 3 70B instruct.
- code: add missing `evaluate_safety.py` script to `scripts`
- code: add Llama 3 tokenizer as a fixture.
- code: fix DPO loss test
- code: make SFT dataset test stricter by comparing against expected output to help folks catch bugs.
- code: include prompts as text files in `cs336_alignment/prompts`
- handout: fix typo in code example for writing AlpacaEval outputs.
- handout: provide more instructions on interpreting AlpacaEval annotations file.
- handout: give better default DPO hyperparameters
- handout: clarify prompt to use for the DPO loss (AlpacaEval prompt) and mention EOS token
- handout: clarify that arrows in the prompts are line continuations, not line breaks
- handout: mention that we provide the prompts as text files at `cs336_alignment/prompts`

### Fixed

## 0.0.2 - 2024-05-30

### Added

- code: add MMLU, GSM8K, AlpacaEval, and SimpleSafetyTests data to `./data`.

### Changed

### Fixed


## 0.0.1 - 2024-05-30

### Added

- handout: explicitly set CUDA_HOME in FlashAttention-2 installation instructions. 
- code: explicitly set CUDA_HOME in FlashAttention-2 installation instructions.

### Changed

### Fixed


## 0.0.0 - 2024-05-30

Initial release