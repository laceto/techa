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
| `techa.agents.orchestrator` | LangGraph + OpenAI + LangChain | OHLCV / parquet | Single-ticker orchestrator: loads OHLCV once and fans out in parallel to `indicators`, `patterns`, and `ta` agents; final synthesis via `gpt-4o` |

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
| `golden_cross` | bool | SMA50 > SMA200 |
| `dist_sma20/50/200_pct` | float\|nan | % distance from each MA |
| `slope_sma20` | float\|nan | OLS slope of SMA20 (%/bar) |
| `slope_sma20_r2` | float | R² for slope; < 0.3 = noise |
| `macd_hist` | float\|nan | positive = bullish |
| `stoch_k/d` | float\|nan | Slow stochastic |
| `roc_10d/20d` | float\|nan | Rate of change (%) |
| `atr_pct` | float\|nan | NATR — normalised ATR (%) |
| `bb_upper/mid/lower` | float\|nan | Bollinger bands |
| `bb_pct_b` | float\|nan | Position within bands [0, 1] |
| `hist_vol_20d` | float\|nan | Annualised historical volatility (%) |
| `obv` | float | On-balance volume |
| `ad` | float | Chaikin A/D line |
| `adosc` | float | Chaikin A/D oscillator |

> RSI, ADX/DI, and vol_vs_ma20 are computed on relative-price data and available via `techa.ma` and `techa.breakout` snapshots.

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

LangGraph multi-agent system for AI-powered technical analysis. All agents use OpenAI structured output (`gpt-4.1-nano`). Set `OPENAI_API_KEY` before use.

### `techa.agents.indicators` — Indicator Analysis Agent

Single-ticker analysis from raw OHLCV: trend (MA alignment), momentum (MACD/stochastic/ROC), and volatility & volume flow (ATR/BB/Chaikin). Three workers run in parallel via `Send`; synthesis via `gpt-4o`.

```python
from techa.agents.indicators import create_indicator_agent

# Live mode (default) — downloads OHLCV via yfinance
graph = create_indicator_agent("PST.MI")
result = graph.invoke(graph._initial_state)
print(result["final_output"])

# Parquet mode — reads ropen/rhigh/rlow/rclose from analysis_results.parquet
graph = create_indicator_agent("ENI.MI", data_source="parquet", analysis_date="2024-06-30")
result = graph.invoke(graph._initial_state)
print(result["final_output"])
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `symbol` | required | Ticker to analyse (e.g. `"PST.MI"`) |
| `analysis_date` | `None` | ISO date ceiling; `None` → latest bar (parquet mode only) |
| `data_source` | `"live"` | `"live"` (yfinance) or `"parquet"` (relative-price OHLCV) |
| `lookback_days` | `365` | Calendar days of OHLCV history to download (live mode only) |
| `checkpointer` | `None` | LangGraph checkpointer for persistence / resumption |

**Structured output per worker:**

| Worker | Pydantic model | Key fields |
|---|---|---|
| `trend` | `TrendAnalysis` | `sma_alignment`, `slope_direction`, `slope_quality`, `golden_cross`, `dist_sma20/50_pct`, `conviction`, `verdict` |
| `momentum` | `MomentumAnalysis` | `macd_bias`, `macd_hist`, `stoch_condition`, `stoch_k`, `momentum_direction`, `roc_20d`, `chg_5d`, `conviction`, `verdict` |
| `volatility` | `VolatilityAnalysis` | `volatility_regime`, `atr_pct`, `hist_vol_20d`, `bb_position`, `bb_pct_b`, `bb_squeeze`, `volume_flow`, `conviction`, `verdict` |

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
| `relative` | `False` | If `True`, signals use relative prices (stock / benchmark). If `False` (default, matches config.json), absolute. Live mode only. |
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

### `techa.agents.orchestrator` — Orchestrator Agent

Single-ticker orchestrator: loads OHLCV once and fans out **in parallel** to three agents — `indicators` (trend / momentum / volatility), `patterns` (candlestick scan), and `ta` (MA crossovers + breakout). A final `gpt-4o` call synthesises all assessments into a structured markdown brief.

```python
from techa.agents.orchestrator import create_orchestrator

# Live mode (default) — downloads OHLCV via YFinanceDataHandler
graph = create_orchestrator("PST.MI")
result = graph.invoke(graph._initial_state)
print(result["final_output"])

# Parquet mode — reads ropen/rhigh/rlow/rclose from analysis_results.parquet
graph = create_orchestrator("A2A.MI", data_source="parquet", analysis_date="2024-06-30")
result = graph.invoke(graph._initial_state)
print(result["final_output"])
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `symbol` | required | Ticker to analyse (e.g. `"PST.MI"`) |
| `data_source` | `"live"` | `"live"` (yfinance) or `"parquet"` (relative-price OHLCV) |
| `analysis_date` | `None` | ISO date ceiling; `None` → latest bar (parquet mode only) |
| `lookback_days` | `365` | Calendar days of OHLCV history to fetch (live mode only) |
| `benchmark` | `"FTSEMIB.MI"` | Benchmark ticker for relative-price computation (ta runner) |
| `fx` | `None` | Optional FX ticker for currency conversion (e.g. `"EURUSD=X"`) |
| `relative` | `False` | If `True`, ta runner uses relative prices (stock / benchmark). If `False` (default, matches config.json), absolute. Live mode only. |
| `checkpointer` | `None` | LangGraph checkpointer for persistence / resumption |

**`final_output` report sections:**

1. **Position Recommendation** — LONG / SHORT / NEUTRAL with conviction level and holding horizon
2. **Signal Confluence Scorecard** — 5-dimension table across Trend / Momentum / Volatility / Patterns / TA
3. **Indicators Deep-Dive** — Trend (MA alignment, slope), Momentum (MACD, stochastic, ROC), Volatility & Volume Flow (ATR, BB, Chaikin)
4. **Candlestick Patterns** — last-bar hits, recent activity, confluence with indicators
5. **TA Deep-Dive** — MA crossovers (EMA/SMA triple confluence), breakout analysis (range quality, vol compression)
6. **Entry & Exit Plan** — specific entry trigger, stop-loss, first target
7. **Bottom Line** — net conviction and the single signal to monitor

---

## Development

```bash
python -m pytest techa/indicators/tests/   # indicator tests
python -m pytest techa/ma/tests/           # MA tests
python -m pytest techa/breakout/tests/     # breakout tests
python -m pytest techa/                    # all tests
```

See `.claude/rules/` for task-specific coding rules loaded by `CLAUDE.md`.
