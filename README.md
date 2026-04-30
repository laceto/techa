# techa

Technical analysis primitives for trader assistants.

## Subpackages

| Package | Backed by | Input | Purpose |
|---|---|---|---|
| `techa.indicators` | TA-Lib | Raw OHLCV DataFrame | Last-bar indicator snapshot (trend, momentum, volatility, volume) |
| `techa.patterns` | TA-Lib + mplfinance | Raw OHLCV DataFrame | Candlestick pattern scanner and visualizer (61 patterns) |
| `techa.ma` | Manual Wilder | Relative-price parquet | Moving-average crossover analytics |
| `techa.breakout` | Manual | Relative-price parquet | Range breakout analytics |

> **Note:** `techa.ma` and `techa.breakout` currently use the old `ta.` import alias internally. Raw OHLCV is not present in their parquet input.

---

## `techa.indicators`

Last-bar technical indicator snapshot from raw OHLCV history.

```python
from techa.indicators import build_snapshot, build_snapshot_from_parquet

snap = build_snapshot(ohlcv_df)               # dict of floats
snap = build_snapshot(ohlcv_df, nan_to_none=True)  # JSON-safe (NaN → None)

snap = build_snapshot_from_parquet(
    ticker="SIE.DE",
    data_path="data/ohlcv.parquet",
    ticker_col="symbol",
    date_col="date",
)
```

**Input:** DataFrame with columns `open/high/low/close/volume` (case-insensitive), sorted ascending, minimum 30 bars.

**Output schema (selected keys):**

| Key | Type | Notes |
|---|---|---|
| `price` | float | Last close |
| `rsi` | float\|nan | [0, 100] |
| `rsi_zone` | str | `"overbought"` / `"neutral"` / `"oversold"` / `"n/a"` |
| `adx` | float\|nan | > 25 = trending |
| `macd_hist` | float\|nan | positive = bullish |
| `atr_pct` | float\|nan | NATR — ATR / close × 100 |
| `bb_upper/mid/lower` | float\|nan | Bollinger bands |
| `hist_vol_20d` | float\|nan | Annualised historical volatility (%) |
| `vol_vs_ma20` | float\|nan | Last-bar volume / 20-bar mean |
| `slope_sma20` | float\|nan | OLS slope of SMA20 (%/bar) |
| `slope_sma20_r2` | float | R² for slope; < 0.3 = noise |

Group snapshots across many tickers:

```python
from techa.indicators import build_group_snapshot, build_ticker_snapshot

gs = build_group_snapshot(ohlcv_dict)   # GroupSnapshot(tickers=df, groups=df)
df = build_ticker_snapshot(ohlcv_dict)  # per-ticker last-bar table
```

---

## `techa.patterns`

Detect and visualise all 61 TA-Lib candlestick patterns.

```python
from techa.patterns import scan_patterns, scan_last_bar, plot_pattern, explore_patterns

# Scan full history — returns tidy DataFrame
hits = scan_patterns(ohlcv_df)
hits = scan_patterns(ohlcv_df, patterns=["CDLENGULFING", "CDLDOJI"])
hits = scan_patterns(ohlcv_df, signal_filter="bull")  # +100 only

# hits columns: date | talib_name | display_name | signal (+100 or -100)

# Nightly multi-ticker scan — patterns that fired on each ticker's last bar
report = scan_last_bar({"STMMI.MI": ohlcv_stmmi, "PRY.MI": ohlcv_pry})
report = scan_last_bar(ohlcv_by_ticker, signal_filter="bear")

# report columns: ticker | date | display_name | signal

# Plot one pattern
plot_pattern(ohlcv_df, "CDLENGULFING", symbol="SIE.DE")
plot_pattern(ohlcv_df, "CDLENGULFING", output="save", output_dir="charts/")

# Explore all patterns that fired
explore_patterns(ohlcv_df, symbol="SIE.DE", output="save", output_dir="charts/")
explore_patterns(ohlcv_df, signal_filter="bear", patterns=["CDLEVENINGSTAR", "CDLSHOOTINGSTAR"])
```

**`plot_pattern` keyword arguments:**

| Argument | Default | Description |
|---|---|---|
| `signal_filter` | `"all"` | `"all"` / `"bull"` / `"bear"` |
| `window_bars` | `5` | Trading bars on each side of each occurrence |
| `max_occurrences` | `5` | Zoom windows shown (most-recent first, capped at 5) |
| `show_volume` | `True` | Volume bars on overview and zoom panels |
| `output` | `"show"` | `"show"` (interactive) or `"save"` (PNG) |
| `output_dir` | `"."` | Destination for saved PNGs |
| `figsize` | `(20, 10)` | Figure size in inches |
| `symbol` | `""` | Ticker label in chart titles |

**CLI:**

```bash
python -m techa.patterns SIE.DE
python -m techa.patterns SIE.DE 2020-01-01 2024-01-01
python -m techa.patterns SIE.DE 2020-01-01 2024-01-01 save
```

---

## Development

```bash
python -m pytest techa/indicators/tests/   # indicator tests
python -m pytest techa/ma/tests/           # MA tests
python -m pytest techa/breakout/tests/     # breakout tests
python -m pytest techa/                    # all tests
```

See `.claude/rules/` for task-specific coding rules loaded by `CLAUDE.md`.
