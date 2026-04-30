"""
techa/patterns — Candlestick pattern scanner and explorer (ta-lib backed).

Primary entry points
--------------------
scan_patterns(ohlcv, patterns=None, signal_filter="all") -> pd.DataFrame
    Detect patterns and return a tidy table of (date, talib_name, display_name, signal).
    signal is +100 (bullish) or -100 (bearish) as returned by TA-Lib.

plot_pattern(ohlcv, pattern, *, ...) -> None
    Price overview with signal markers + up to 5 candlestick zoom windows.

explore_patterns(ohlcv, patterns=None, signal_filter="all", *, ...) -> None
    Batch: calls plot_pattern for every pattern that fired at least once.

CLI
---
python -m techa.patterns TICKER [START] [END] [show|save]
"""

from techa.patterns.scanner import scan_patterns, scan_last_bar
from techa.patterns.explorer import plot_pattern, explore_patterns

__all__ = ["scan_patterns", "scan_last_bar", "plot_pattern", "explore_patterns"]
