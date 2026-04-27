# techa/indicators — Rebuild Plan

Redesign based on core-architect + api-product-designer review.
Mark each item `[x]` when implementation is complete.

---

## Module Structure

| Status | File | Responsibility |
|--------|------|----------------|
| [x] | `_adapter.py` | `to_numpy_ohlcv()`, `last_valid()`, `MIN_BARS` — single DataFrame→numpy conversion point |
| [x] | `trend.py` | `compute_trend(o, h, l, c)` → dict — talib: SMA, EMA, ADX, PLUS_DI, MINUS_DI |
| [x] | `momentum.py` | `compute_momentum(c, h, l)` → dict — talib: RSI, MACD, STOCH, ROC |
| [x] | `volatility.py` | `compute_volatility(h, l, c)` → dict — talib: ATR, NATR, BBANDS; manual HV |
| [x] | `volume.py` | `compute_volume(c, v)` → dict — talib: OBV, SMA for vol_ma |
| [x] | `snapshot.py` | `build_snapshot()`, `build_snapshot_from_parquet()` — thin orchestrator (replaces compute.py) |
| [x] | `grouping.py` | `assign_vol_regime()`, `assign_user_groups()` — group assignment only |
| [x] | `aggregation.py` | `aggregate_group()`, `build_group_df()`, constants — aggregation only |
| [x] | `group_snapshot.py` | `GroupSnapshot`, `build_group_snapshot()`, `build_ticker_snapshot()` — thin orchestrator |
| [x] | `__init__.py` | Updated public exports |
| [x] | ~~`compute.py`~~ | Delete — replaced by snapshot.py + domain modules |

---

## Public API Changes

| Status | Change | Detail |
|--------|--------|--------|
| [x] | Rename `compute_last_bar` → `build_snapshot` | Keep deprecated alias with `DeprecationWarning` |
| [x] | Rename `compute.py` → `snapshot.py` | Matches sibling pattern: `ma_snapshot.py`, `bo_snapshot.py` |
| [x] | Add `build_snapshot_from_parquet(ticker, data_path, *, ticker_col, date_col)` | No default path — caller must supply |
| [x] | Return `GroupSnapshot` dataclass from `build_group_snapshot` | Replace bare tuple `(ticker_df, group_df)` |
| [x] | Rename `build_ticker_table` → `build_ticker_snapshot` | Keep deprecated alias |
| [x] | Add `include_vol_regime: bool = True` to `build_group_snapshot` | Lets caller opt out of auto vol_regime grouping |
| [x] | Add `nan_to_none: bool = False` to `build_snapshot` | Makes output JSON-serialisable without custom encoder |

---

## Snapshot Schema Changes

### Keys removed (redundant booleans — always derivable from floats)

| Status | Key removed | Derive instead from |
|--------|-------------|---------------------|
| [x] | `rsi_overbought` | `snap["rsi"] > 70` |
| [x] | `rsi_oversold` | `snap["rsi"] < 30` |
| [x] | `above_sma20` | `snap["dist_sma20_pct"] > 0` |
| [x] | `above_sma50` | `snap["dist_sma50_pct"] > 0` |
| [x] | `above_sma200` | `snap["dist_sma200_pct"] > 0` |
| [x] | `macd_bullish` | `snap["macd_hist"] > 0` |
| [x] | `high_volume` | `snap["vol_vs_ma20"] > 1.5` |

### Keys added

| Status | Key | Type | Notes |
|--------|-----|------|-------|
| [x] | `rsi_zone` | `str` | `"overbought"` / `"neutral"` / `"oversold"` / `"n/a"` |
| [x] | `bb_upper` | `float` | Bollinger upper band (was computed but not exposed) |
| [x] | `bb_mid` | `float` | Bollinger middle band (= SMA20) |
| [x] | `bb_lower` | `float` | Bollinger lower band |

### Keys renamed

| Status | Old key | New key | Reason |
|--------|---------|---------|--------|
| [x] | `hv20` | `hist_vol_20d` | Spells out concept; period explicit |
| [x] | `vol_ratio` | `vol_vs_ma20` | States what the ratio is |
| [x] | `roc10` | `roc_10d` | Consistent underscore + unit pattern |
| [x] | `roc20` | `roc_20d` | Same |
| [x] | `chg1d` | `chg_1d` | Underscore for readability |
| [x] | `chg5d` | `chg_5d` | Same |
| [x] | `sma20_slope` | `slope_sma20` | Metric type first; normalised to %/bar |
| [x] | `sma20_slope_r2` | `slope_sma20_r2` | Same |

---

## Bugs Fixed

| Status | Bug | Fix |
|--------|-----|-----|
| [x] | Pandas RSI/ADX diverge from ta-lib Wilder smoothing | Replace with `talib.RSI`, `talib.ADX` |
| [x] | `vol_ma20` uses `min_periods=5` — produces meaningless ratios for short series | Use `talib.SMA` with strict lookback via `last_valid()` |
| [x] | Multi-group membership encoded as comma-joined string — corrupts labels containing commas | Use list column; explode directly |
| [x] | Group assignment and aggregation in same module — unrelated concerns change together | Split into `grouping.py` + `aggregation.py` |
| [x] | No input validation before computation — opaque `KeyError` mid-stack | Validate OHLCV columns at top of `build_snapshot` |

---

## ta-lib Integration Pattern

All indicator functions receive `np.ndarray[float64]` from `_adapter.to_numpy_ohlcv()`.
`last_valid(arr)` reads `arr[-1]`, returning `float("nan")` if NaN (lookback not satisfied).
`build_snapshot` enforces `MIN_BARS = 30` hard minimum.

```
build_snapshot(ohlcv_df)
  └─ to_numpy_ohlcv()          # validate + convert once
  ├─ compute_trend(o,h,l,c)    # talib.SMA/EMA/ADX/PLUS_DI/MINUS_DI
  ├─ compute_momentum(c,h,l)   # talib.RSI/MACD/STOCH/ROC
  ├─ compute_volatility(h,l,c) # talib.ATR/NATR/BBANDS + manual HV
  └─ compute_volume(c,v)       # talib.OBV + talib.SMA for vol_ma
```

---

## Full Output Schema (target)

```python
{
    # Identity
    "price":            float,

    # Trend
    "sma20":            float | nan,
    "sma50":            float | nan,
    "sma200":           float | nan,
    "ema20":            float | nan,
    "ema50":            float | nan,
    "dist_sma20_pct":   float,          # (price - sma20) / sma20 * 100
    "dist_sma50_pct":   float,
    "dist_sma200_pct":  float,
    "slope_sma20":      float,          # OLS slope, price-normalised %/bar
    "slope_sma20_r2":   float,          # [0, 1]
    "adx":              float | nan,    # [0, 100]; > 25 = trending
    "di_plus":          float | nan,
    "di_minus":         float | nan,
    "golden_cross":     bool,           # sma50 > sma200

    # Momentum
    "rsi":              float | nan,    # [0, 100]
    "rsi_zone":         str,            # "overbought"|"neutral"|"oversold"|"n/a"
    "macd":             float | nan,
    "macd_signal":      float | nan,
    "macd_hist":        float | nan,    # positive = bullish
    "stoch_k":          float | nan,    # [0, 100]
    "stoch_d":          float | nan,
    "roc_10d":          float | nan,    # 10-bar rate of change (%)
    "roc_20d":          float | nan,
    "chg_1d":           float,          # 1-bar % change
    "chg_5d":           float,          # 5-bar % change

    # Volatility
    "atr":              float | nan,    # ATR in price units
    "atr_pct":          float | nan,    # NATR — ATR / close * 100
    "bb_upper":         float | nan,
    "bb_mid":           float | nan,
    "bb_lower":         float | nan,
    "bb_width":         float | nan,    # (upper - lower) / mid
    "bb_pct_b":         float | nan,    # position in band; > 1 = above upper
    "hist_vol_20d":     float | nan,    # annualised HV (%)

    # Volume
    "volume":           float,          # raw last-bar volume
    "vol_vs_ma20":      float | nan,    # volume / 20-bar mean volume
    "obv":              float | nan,
}
```
