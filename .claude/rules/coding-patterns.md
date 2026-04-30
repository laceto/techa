# Coding Rules — `techa.patterns`

## Context
You are working in `techa/patterns/`. This subpackage scans raw OHLCV data for all 61 TA-Lib candlestick patterns and visualises matches with an overview chart + candlestick zoom windows.

## Commands
```bash
python -m techa.patterns SIE.DE 2020-01-01 2024-01-01        # interactive (show)
python -m techa.patterns SIE.DE 2020-01-01 2024-01-01 save   # write PNGs to cwd
python -c "from techa.patterns import scan_patterns; ..."     # programmatic
```

## Module responsibilities — do not cross boundaries

| File | Responsibility |
|---|---|
| `_registry.py` | Single source of truth: `PATTERNS` list and `PATTERN_DISPLAY` lookup dict. No imports from talib here. |
| `scanner.py` | `scan_patterns()`, `scan_last_bar()` — the only file that calls `getattr(talib, talib_name)`. Returns tidy DataFrames. |
| `explorer.py` | `plot_pattern()`, `explore_patterns()` — pure visualization. Calls `scan_patterns()`; no direct talib calls. |
| `__main__.py` | CLI entry point. Calls `yfinance.download()` + `explore_patterns()`. No pattern logic here. |

## Import path
Use `techa.patterns` everywhere — `from techa.patterns.scanner import scan_patterns`.

## Adding a new pattern
1. Add `("Display Name", "CDLNEWPATTERN")` to `PATTERNS` in `_registry.py`.
2. That is all — `scan_patterns` and `explore_patterns` pick it up automatically.

## Adding a new visualization feature
Add it to `explorer.py`. Keep all matplotlib/mplfinance code out of `scanner.py`.

## Signal values
TA-Lib pattern functions return integers: `+100` (bullish), `-100` (bearish), `0` (no signal). The scanner strips zeros and exposes only `+100` / `-100`. Never treat the raw talib array as boolean.

## OHLCV input contract
- Column names are case-insensitive (lowercased internally via `.str.lower()`).
- Required columns: `open`, `high`, `low`, `close`. `volume` is optional.
- Index must be datetime, sorted ascending.
- No minimum bar count is enforced (patterns fire on short series; short series just produce fewer signals).

## Pattern identifiers
The public API uses TA-Lib function attribute names as pattern identifiers: `"CDLENGULFING"`, `"CDLDOJI"`, etc. Display names (`"Engulfing Pattern"`) are internal to `_registry.py` and returned as a column in the scan DataFrame.

## `scan_patterns` return schema
```python
pd.DataFrame(columns=["date", "talib_name", "display_name", "signal"])
# date         — datetime, matches ohlcv.index
# talib_name   — str, e.g. "CDLENGULFING"
# display_name — str, e.g. "Engulfing Pattern"
# signal       — int, +100 or -100
```
Sorted ascending by date. Empty DataFrame (correct schema) if no patterns fired.

## `scan_last_bar` return schema
```python
pd.DataFrame(columns=["ticker", "date", "display_name", "signal"])
# ticker       — str, key from ohlcv_by_ticker dict
# date         — datetime, the last bar date for that ticker
# display_name — str, e.g. "Engulfing Pattern"
# signal       — int, +100 or -100
```
Filters each ticker's scan to `date == hits["date"].max()` — patterns that fired on the most recent bar only. Empty DataFrame (correct schema) if nothing fired across all tickers. `talib_name` is intentionally excluded from output (internal identifier).

## `plot_pattern` layout
```
Row 0: Full-width close price line (all signals as faint dashes; zoomed ones numbered)
Row 1: Full-width volume bars [optional, show_volume=True]
Row 2: n_zooms candlestick zoom panels (most-recent occurrences, one column each)
Row 3: Per-zoom volume bars [optional]
```
Uses `layout="constrained"` — do not call `tight_layout()`.

## When done
→ STOP. Do not write tests here. Do not modify the router `CLAUDE.md`.
→ Next: load `test-rules.md` if tests need to be written.
