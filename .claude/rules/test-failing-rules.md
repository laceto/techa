# Test Failing Rules

## Context
Tests are red. This overrides `test-rules.md`. Your only job is to make failing tests pass.

## Process
1. Read the full failure output — do not skim.
2. Identify the specific assertion that failed and which source line it points to.
3. Trace the source function; find the mismatch between expectation and implementation.
4. Make the minimal change to source code that satisfies the contract.
5. Run the failing test again: `python -m pytest path/to/test.py::test_name`.
6. If it passes, run the full suite: `python -m pytest techa/`.
7. Repeat until all green.

## Hard rules
- Do NOT modify tests to make them pass.
- Do NOT add features or refactor while fixing.
- Do NOT change more code than required.
- The test defines the contract. If the test is wrong, flag it — but do not silently modify it.

## Common failure patterns in this codebase
- **NaN propagation**: `ols_slope_r2` returns `(0.0, 0.0)` on NaN input — check the guard at the top of the function.
- **Wilder convergence**: results differ from expected when history is shorter than `3×period` — the function should log a WARNING; check the threshold.
- **Signal validation**: `assess_ma_volume` strict int-only check — a float column from parquet may trigger this.
- **Import path**: `ta.` vs `techa.` mismatch causes `ModuleNotFoundError` — match the existing imports in the file.
- **TA-Lib missing**: `techa.indicators` requires the TA-Lib C library. If tests for `techa.ma`/`techa.breakout` fail with `ModuleNotFoundError: talib`, the wrong test path is being run.

## When done
→ All tests pass: `python -m pytest techa/`
→ STOP. Do not refactor.
