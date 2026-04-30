# Coding Rules — `techa.indicators`

## Context
You are working in `techa/indicators/`. This subpackage computes last-bar technical indicator snapshots from **raw OHLCV** DataFrames using TA-Lib.

## Commands
```bash
python -m pytest techa/indicators/tests/   # run indicator tests
python -m pytest techa/indicators/tests/test_snapshot.py::test_name  # single test
```

## Architecture constraints
- `_adapter.py` is the **only** file that calls `.to_numpy()` or `.values` on OHLCV data. Every domain module receives `np.ndarray[float64]` from `to_numpy_ohlcv()`.
- `last_valid(arr)` reads `arr[-1]`, returning `float("nan")` when ta-lib's lookback is not satisfied. Use it for every ta-lib output array — never index raw ta-lib output directly.
- `MIN_BARS = 30` is enforced once in `snapshot.py`. Domain modules assume it.
- `snapshot.py` is a thin orchestrator — add no indicator logic there. Add it to the correct domain module: `trend.py`, `momentum.py`, `volatility.py`, or `volume.py`.

## Import path
Use `techa.` everywhere in this subpackage — `from techa.indicators._adapter import last_valid`.

## Adding a new indicator
1. Pick the right domain module (trend/momentum/volatility/volume).
2. Use `talib.<FUNC>(arr)` and wrap the result with `last_valid()`.
3. Add the key to the flat `dict` returned by `compute_<domain>()`.
4. Update the schema comment in `indicators/PLAN.md` if the public key changes.

## Naming conventions
- Keys: lowercase with underscores + unit suffix where ambiguous (`atr_pct`, `hist_vol_20d`, `slope_sma20`).
- Metric type first in compound names: `slope_sma20`, not `sma20_slope`.
- Period explicit in name when multiple periods exist: `roc_10d`, `roc_20d`.

## OLS slopes
Always return R² alongside every slope. Use `techa.utils.ols_slope_r2`. Never expose a slope without its R².

## `nan_to_none`
`build_snapshot(ohlcv, nan_to_none=True)` replaces `float("nan")` with `None` for JSON-safe output. Implement this only in `snapshot.py`, not in domain modules.

## When done
→ STOP. Do not write tests here. Do not modify the router `CLAUDE.md`.
→ Next: load `test-rules.md` if tests need to be written.
