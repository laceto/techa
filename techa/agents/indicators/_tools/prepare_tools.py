"""
agents/indicators/_tools/prepare_tools.py — Data loading for the IndicatorAnalysis agent.

Mode A (parquet): reads ropen/rhigh/rlow/rclose from analysis_results.parquet.
                  Indicator values are computed on relative prices vs benchmark.
Mode B (live):    downloads raw OHLCV via yfinance for absolute-price indicators.

Public API:
    load_ohlcv_from_parquet(path, symbol, analysis_date) -> (df, resolved_date)
    download_ohlcv_live(symbol, lookback_days)           -> (df, resolved_date)

Both return (ohlcv_df, ISO-date-string).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from techa.agents._common import RESULTS_PATH, HISTORY_BARS, _read_parquet_dated

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode A — parquet
# ---------------------------------------------------------------------------


def load_ohlcv_from_parquet(
    path: Path,
    symbol: str,
    analysis_date: str | None,
) -> tuple[pd.DataFrame, str]:
    """
    Load OHLCV for one symbol from analysis_results.parquet.

    Uses ropen/rhigh/rlow/rclose columns (relative prices vs benchmark).
    Indicator values will be relative-price-based rather than absolute.

    Args:
        path:          Path to analysis_results.parquet.
        symbol:        Ticker symbol to load.
        analysis_date: ISO date ceiling (inclusive); None → latest available bar.

    Returns:
        (ohlcv_df, resolved_date) where ohlcv_df has columns open/high/low/close
        and a datetime index, and resolved_date is the last bar's ISO date.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If symbol is not found in the parquet.
    """
    df_all = _read_parquet_dated(path, analysis_date)

    df = df_all[df_all["symbol"] == symbol].copy()
    if df.empty:
        sample = df_all["symbol"].unique()[:10].tolist()
        raise ValueError(
            f"Symbol '{symbol}' not found in {path}. "
            f"Sample available: {sample}."
        )

    df = df.sort_values("date").tail(HISTORY_BARS)
    resolved_date = df["date"].iloc[-1].strftime("%Y-%m-%d")

    rename = {k: v for k, v in {
        "ropen":  "open",
        "rhigh":  "high",
        "rlow":   "low",
        "rclose": "close",
    }.items() if k in df.columns}
    df = df.rename(columns=rename).set_index("date").sort_index()

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    log.info("[prepare] parquet: %s  %d bars  resolved=%s", symbol, len(df), resolved_date)
    return df[keep], resolved_date


# ---------------------------------------------------------------------------
# Mode B — live download
# ---------------------------------------------------------------------------


def download_ohlcv_live(
    symbol: str,
    lookback_days: int = 365,
) -> tuple[pd.DataFrame, str]:
    """
    Download raw OHLCV for one symbol via yfinance.

    Args:
        symbol:        Ticker symbol (e.g. "PST.MI").
        lookback_days: Calendar days of history to fetch (default 365).

    Returns:
        (ohlcv_df, resolved_date) where resolved_date is the last bar's ISO date.

    Raises:
        ValueError: If yfinance returns no data for the symbol.
    """
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=lookback_days)
    start    = start_dt.strftime("%Y-%m-%d")
    end      = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    log.info("[prepare] downloading OHLCV: %s  %s → %s", symbol, start, end)

    df = yf.download(
        symbol,
        start=start,
        end=end,
        multi_level_index=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(
            f"yfinance returned no data for '{symbol}' "
            f"in range {start} → {end}. Check the ticker symbol."
        )

    df.columns = df.columns.str.lower()
    resolved_date = df.index[-1].strftime("%Y-%m-%d")
    log.info("[prepare] live: %s  %d bars  resolved=%s", symbol, len(df), resolved_date)
    return df, resolved_date
