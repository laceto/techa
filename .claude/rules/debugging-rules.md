# Debugging Rules

## Context
You are investigating a bug, unexpected value, or silent failure in the `techa` library.

## Common failure modes by area

### NaN propagation
- `ols_slope_r2`: any NaN in `values` → `(0.0, 0.0)` (intentional safe degradation). Check callers that treat `0.0` slope as a real value.
- `last_valid(arr)`: returns `float("nan")` when ta-lib's lookback is not satisfied. Callers must handle `nan` before doing arithmetic.
- `_wilder_running_sum` / `_wilder_ewm`: NaN in input array propagates to all subsequent output values. Trace the NaN source (usually a zero or missing price bar).

### Wilder smoothing divergence
- The library implements Wilder's smoothing from scratch — it may differ from ta-lib's output for short series (seed value sensitivity). Check whether `len(series) >= 3 * period`.
- `_wilder_running_sum` (seed=sum, coefficient=1) vs `_wilder_ewm` (seed=mean, coefficient=1/period) are NOT interchangeable. Make sure the right variant is used at each step of ADX.

### Signal column corruption
- Parquet round-trips can silently widen `int8` signal columns to `float64`. `assess_ma_volume` catches this via explicit validation; other callers may not. If a flip is not detected, check the dtype of the signal column.

### Relative-price calculation errors
- `compute_ma_gap_pct` raises `ValueError` for `rclose <= 0`. A zero relative price means the ticker traded at the same price as FTSEMIB.MI — extremely rare but valid data for some synthetic tests. Guard before calling.
- `rclose_safe = df["rclose"].replace(0, np.nan)` is the established pattern for zero-rclose guard in gap series.

## Trace process
1. Reproduce with the smallest possible synthetic DataFrame.
2. Print intermediate arrays at each step (Wilder sum, DI series, ADX series).
3. Compare against a known-good reference (e.g., ta-lib output for the same input).
4. Identify which step first diverges.

## When done
→ Document the root cause (one sentence).
→ Create a task: "Fix [issue description]".
→ Do NOT implement the fix here unless explicitly asked.
