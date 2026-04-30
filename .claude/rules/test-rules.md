# Test Rules

## Context
You are writing tests for the `techa` library. Tests live in `{subpackage}/tests/` and run with pytest.

## Commands
```bash
python -m pytest techa/                          # all tests
python -m pytest techa/ma/tests/                 # subpackage
python -m pytest techa/ma/tests/test_trend_quality.py  # single file
python -m pytest techa/ma/tests/test_trend_quality.py::test_rsi_all_gains  # single test
```

## Test structure
- One `describe`-style class or module per public function.
- Test names: `test_<function>_<scenario>` (e.g. `test_rsi_all_gains`, `test_adx_too_short_raises`).
- Arrange-Act-Assert with no intermediate logic.
- Use synthetic `pd.Series` / `pd.DataFrame` built inline — no fixture files.

## What to test
Every public function needs:
1. **Happy path**: expected return type, expected value for a hand-verified input.
2. **Invariants**: range assertions (e.g. `0 <= rsi <= 100`, `0 <= r2 <= 1`).
3. **Error cases**: every `ValueError` the docstring documents.
4. **Edge cases** specific to this domain:
   - Constant series (RSI=50, ADX≈0, OLS slope=0)
   - All-gains / all-losses series (RSI=100 / RSI=0)
   - NaN in input (expect `(0.0, 0.0)` from `ols_slope_r2`, `float("nan")` from `last_valid`)
   - Flat price series (TR=0 → ADX=0)

## Wilder smoothing invariants (for `techa.ma` tests)
- Constant series stays constant through `_wilder_ewm`.
- `period=1` in `_wilder_running_sum` is identity (output = input).
- `period > len(arr)` returns empty array, not an error.

## Signal column tests
- Always test that non-`{-1, 0, 1}` values raise `ValueError`.
- Test that NaN in signal raises `ValueError`.
- Test that float `0.5` (non-integer) raises `ValueError`.

## Do not modify source code
If a test reveals a bug, create a comment `# BUG: ...` and note it. Do not fix bugs while writing tests.

## When done
→ Run all tests: `python -m pytest techa/`
→ If ALL pass → STOP.
→ If ANY fail → load `test-failing-rules.md`.
