# Changelog

All changes we make to the assignment code or PDF will be documented in this file.

## [26.0.5] - 2026-05-08

- code: make training dependencies optional

## [26.0.4] - 2026-05-05

- code: rename test_and_make_submission.sh -> make_submission.sh and remove pytest

## [26.0.3] - 2026-05-05

- code: add test_and_make_submission.sh

## [26.0.2] - 2026-05-04

- code: remove max runtime from training config to allow 48 hour final submission

## [26.0.1] - 2026-05-01

- writeup: add AI policy

## [26.0.0] - 2026-04-29

- change from offline to online model training api

## [1.0.0] - 2024-04-30
- handout: document end point for previous runs
- api: add endpoint for previous runs
- api: use xgboost regressor instead of sklearn tree regressor

### Added

- handout: added suggestion to plan out your scaling runs beforehand, since the
  training API will refuse further requests past the 2e18 budget.

### Changed

### Fixed

## [0.0.0] - 2024-05-02

Initial release.
