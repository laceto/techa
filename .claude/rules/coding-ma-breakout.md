# Coding Rules — `techa.ma` and `techa.breakout`

## Context
You are working in `techa/ma/` or `techa/breakout/`. These subpackages analyse **relative-price** data loaded from a pre-computed parquet (`data/results/it/analysis_results.parquet`). They do NOT use TA-Lib — all smoothing is implemented manually.

## Commands
```bash
python -m pytest techa/ma/tests/
python -m pytest techa/breakout/tests/
python -m pytest techa/ma/tests/test_trend_quality.py::test_name  # single test
```

## Import path — critical
These subpackages **still use the old `ta.` package name**. Match the existing imports in the file you are editing:
```python
from ta.utils import ols_slope_r2          # correct for ma/ and breakout/
from ta.ma.trend_quality import assess_ma_trend
from ta.breakout.range_quality import assess_range
```
Do **not** change existing imports to `techa.` — that rename is in progress and must be done as a separate task.

## Relative prices
All price data is ticker price / FTSEMIB.MI benchmark. Column names are prefixed `r`: `rclose`, `rhigh`, `rema_short_50`, `rbo_N_age`, etc. Raw OHLC is not present in the parquet. Slope and volatility estimates are directly comparable across tickers because the normalisation removes price-level effects.

## Wilder smoothing
Two distinct variants — choose the right one:
- `_wilder_running_sum`: coefficient=1 on new value, seed=sum. Used for TR, +DM, −DM.
- `_wilder_ewm`: coefficient=1/period on new value, seed=mean. Used for RSI avg_gain/avg_loss and DX→ADX.

Both are overflow-safe via the batched implementation in `trend_quality.py`. Do not re-implement Wilder smoothing; reuse these helpers.

## OLS slopes
Always return R² alongside every slope. Use `techa.utils.ols_slope_r2`. R² < 0.3 = slope is noise-dominated; R² ≥ 0.7 = reliable.

## Float comparisons
Use tolerance, not `== 0`. Pattern: `if val < 1e-10`. See `_wilder_running_sum` and `compute_ma_gap_pct` for the established idiom.

## Lookahead protection
All functions return a snapshot of bar t using data up to and including bar t. When called in a rolling backtesting loop, the caller must `signals.shift(1)`. Document this in every new public function's docstring.

## Signal column values
MA signal columns (`rema_50100`, etc.) hold only `{-1, 0, 1}`. `assess_ma_volume` validates this strictly. Any continuous score, error sentinel, or NaN raises `ValueError`. If you add a function that consumes a signal column, apply the same validation.

## Snapshot builders (`ma_snapshot.py`, `bo_snapshot.py`)
These are thin orchestrators — no indicator math lives here. They:
1. Call `select_columns(df_ticker)` to narrow the DataFrame.
2. Call domain primitives for enrichment.
3. Serialize the last bar to a JSON-safe dict.

## When done
→ STOP. Do not write tests here.
→ Next: load `test-rules.md` if tests need to be written.
