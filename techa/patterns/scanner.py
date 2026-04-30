"""
techa/patterns/scanner.py — Candlestick pattern detection.

Public API
----------
scan_patterns(ohlcv, patterns=None, signal_filter="all") -> pd.DataFrame
    Detect all matching patterns and return a tidy (date, pattern, signal) table.

scan_last_bar(ohlcv_by_ticker, patterns=None, signal_filter="all") -> pd.DataFrame
    Multi-ticker convenience: patterns that fired on each ticker's last bar only.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import talib

from techa.patterns._registry import PATTERNS

__all__ = ["scan_patterns", "scan_last_bar"]


def _ohlcv_arrays(ohlcv: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = ohlcv.copy()
    df.columns = df.columns.str.lower()
    return (
        df["open"].to_numpy(dtype=np.float64),
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        df["close"].to_numpy(dtype=np.float64),
    )


def scan_patterns(
    ohlcv: pd.DataFrame,
    patterns: list[str] | None = None,
    signal_filter: Literal["all", "bull", "bear"] = "all",
) -> pd.DataFrame:
    """
    Detect candlestick patterns in OHLCV data.

    Args:
        ohlcv:         DataFrame with open/high/low/close columns (case-insensitive).
                       Index must be datetime, sorted ascending.
        patterns:      TA-Lib function names to check, e.g. ["CDLENGULFING", "CDLDOJI"].
                       None checks all 61 patterns.
        signal_filter: "all" returns both directions. "bull" returns only +100 signals.
                       "bear" returns only -100 signals.

    Returns:
        Tidy DataFrame with columns: date, talib_name, display_name, signal (+100 or -100).
        Sorted ascending by date. Empty DataFrame if no patterns fired.
    """
    registry = [
        (dn, tn) for dn, tn in PATTERNS
        if patterns is None or tn in patterns
    ]

    o, h, l, c = _ohlcv_arrays(ohlcv)

    rows = []
    for display_name, talib_name in registry:
        func = getattr(talib, talib_name)
        output = func(o, h, l, c)
        for date, sig in zip(ohlcv.index, output):
            if sig == 0:
                continue
            if signal_filter == "bull" and sig < 0:
                continue
            if signal_filter == "bear" and sig > 0:
                continue
            rows.append({
                "date": date,
                "talib_name": talib_name,
                "display_name": display_name,
                "signal": int(sig),
            })

    if not rows:
        return pd.DataFrame(columns=["date", "talib_name", "display_name", "signal"])

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def scan_last_bar(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    patterns: list[str] | None = None,
    signal_filter: Literal["all", "bull", "bear"] = "all",
) -> pd.DataFrame:
    """
    Scan multiple tickers for patterns that fired on each ticker's last bar.

    Intended for nightly runs: pass one OHLCV DataFrame per ticker; the function
    scans all patterns, filters to the most recent date of each ticker, and returns
    a combined table ready for alerts or reports.

    Args:
        ohlcv_by_ticker: Mapping of {ticker: ohlcv_df}. Each DataFrame must have
                         open/high/low/close columns (case-insensitive), datetime
                         index, sorted ascending.
        patterns:        TA-Lib function names to check. None checks all 61 patterns.
        signal_filter:   "all", "bull" (+100 only), or "bear" (-100 only).

    Returns:
        DataFrame with columns: ticker, date, display_name, signal.
        One row per (ticker, pattern) that fired on that ticker's last bar.
        Empty DataFrame (correct schema) if nothing fired across all tickers.
    """
    frames = []
    for ticker, ohlcv in ohlcv_by_ticker.items():
        hits = scan_patterns(ohlcv, patterns=patterns, signal_filter=signal_filter)
        if hits.empty:
            continue
        last_date = hits["date"].max()
        last = hits[hits["date"] == last_date][["date", "display_name", "signal"]].copy()
        last.insert(0, "ticker", ticker)
        frames.append(last)

    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "display_name", "signal"])

    return pd.concat(frames, ignore_index=True)
