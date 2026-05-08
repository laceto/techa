# techa

Technical analysis primitives for trader assistants.

## Subpackages

| Package | Backed by | Input | Purpose |
|---|---|---|---|
| `techa.indicators` | TA-Lib | Raw OHLCV DataFrame | Last-bar indicator snapshot (trend, momentum, volatility, volume) |
| `techa.patterns` | TA-Lib + mplfinance | Raw OHLCV DataFrame | Candlestick pattern scanner and visualizer (61 patterns) |
| `techa.ma` | Manual Wilder | Relative-price parquet | Moving-average crossover analytics |
| `techa.breakout` | Manual | Relative-price parquet | Range breakout analytics |
| `techa.agents` | LangGraph + OpenAI | OHLCV / parquet | AI-powered multi-agent technical analysis and pattern reports |

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

## `techa.agents`

LangGraph multi-agent system for AI-powered technical analysis. Both agents use OpenAI structured output (`gpt-4.1-nano`). Set `OPENAI_API_KEY` before use.

### `techa.agents.ta` — Technical Analysis Agent

Single-ticker analysis: MA crossovers, breakouts, and indicator context.

```python
from techa.agents.ta import create_manager

# Parquet mode (default) — reads from data/results/it/analysis_results.parquet
graph = create_manager("A2A.MI", analysis_date="2024-06-30")
result = graph.invoke(graph._initial_state)
print(result["final_output"])

# Live mode — downloads OHLCV via yfinance
graph = create_manager("ENI.MI", data_source="live", benchmark="FTSEMIB.MI")
result = graph.invoke(graph._initial_state)
print(result["final_output"])
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `symbol` | required | Ticker to analyse (e.g. `"A2A.MI"`) |
| `analysis_date` | `None` | ISO date ceiling; `None` → latest available bar |
| `data_source` | `"parquet"` | `"parquet"` or `"live"` |
| `benchmark` | `"FTSEMIB.MI"` | Benchmark ticker (for relative-price computation in live mode) |
| `fx` | `None` | Optional FX ticker for currency conversion (e.g. `"EURUSD=X"`) |
| `checkpointer` | `None` | LangGraph checkpointer for persistence / resumption |

### `techa.agents.patterns` — Candlestick Pattern Scan Agent

Multi-ticker scan: last-bar candlestick hits plus recent pattern history, structured per-ticker analysis.

```python
from techa.agents.patterns import create_pattern_agent

# Parquet mode — reads ropen/rhigh/rlow/rclose from analysis_results.parquet
graph = create_pattern_agent(["A2A.MI", "ENI.MI"], analysis_date="2024-06-30")
result = graph.invoke(graph._initial_state)
print(result["final_output"])

# Live mode — downloads raw OHLCV via yfinance
graph = create_pattern_agent(
    ["A2A.MI", "ENI.MI"],
    data_source="live",
    signal_filter="bear",
    lookback_bars=30,
)
result = graph.invoke(graph._initial_state)
print(result["final_output"])
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `tickers` | required | List of tickers to scan (e.g. `["A2A.MI", "ENI.MI"]`) |
| `analysis_date` | `None` | ISO date ceiling; `None` → latest available bar (parquet mode only) |
| `data_source` | `"parquet"` | `"parquet"` or `"live"` |
| `signal_filter` | `"all"` | `"all"`, `"bull"` (+100 only), or `"bear"` (-100 only) |
| `lookback_days` | `365` | Calendar days of OHLCV history to download (live mode only) |
| `lookback_bars` | `20` | Trading bars of recent pattern history sent to the model (≈ 1 month) |
| `benchmark` | `"FTSEMIB.MI"` | Accepted for API consistency; not used by pattern nodes |
| `fx` | `None` | Accepted for API consistency; not used by pattern nodes |
| `checkpointer` | `None` | LangGraph checkpointer for persistence / resumption |

**LLM output fields (structured via Pydantic):**

| Field | Description |
|---|---|
| `description` | 3-5 sentence narrative: tickers fired, net bull/bear balance, notable confluences |
| `total_hits` / `bullish_count` / `bearish_count` | Last-bar hit counts |
| `recent_activity` | Per-ticker: recent hit counts, pattern names, activity trend (increasing / stable / decreasing) |
| `ticker_summaries` | Per-ticker: patterns fired, net bias, conviction (high / medium / low), verdict |
| `top_actionable` | Tickers with ≥ 2 same-direction patterns (medium/high conviction) |
| `watchlist` | Tickers with single-bar or mixed signals |
| `summary` | One-sentence net market message |

---

## Development

```bash
python -m pytest techa/indicators/tests/   # indicator tests
python -m pytest techa/ma/tests/           # MA tests
python -m pytest techa/breakout/tests/     # breakout tests
python -m pytest techa/                    # all tests
```

See `.claude/rules/` for task-specific coding rules loaded by `CLAUDE.md`.
