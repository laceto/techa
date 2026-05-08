"""
agents/patterns/_tools/prepare_tools.py — Data loading for the PatternScan agent.

Mode A (parquet): reads ropen/rhigh/rlow/rclose from analysis_results.parquet.
                  Candlestick pattern shapes are invariant to the relative-price normalisation.
Mode B (live):    downloads raw OHLCV via yfinance per ticker.

Public API:
    load_ohlcv_from_parquet(path, tickers, analysis_date) -> (ohlcv_by_ticker, resolved_date)
    download_ohlcv_live(tickers, lookback_days)           -> (ohlcv_by_ticker, scan_date)

Both return ({ticker: ohlcv_df}, ISO-date-string).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import yfinance as yf
from pathlib import Path

from techa.agents._common import RESULTS_PATH, HISTORY_BARS, _read_parquet_dated

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode A — parquet
# ---------------------------------------------------------------------------


def load_ohlcv_from_parquet(
    path: Path,
    tickers: list[str],
    analysis_date: str | None,
) -> tuple[dict, str]:
    """
    Load OHLCV for each ticker from analysis_results.parquet.

    Uses ropen/rhigh/rlow/rclose columns (relative OHLCV vs benchmark).
    Renames them to standard open/high/low/close and sets the date as the index.

    Args:
        path:          Path to analysis_results.parquet.
        tickers:       Ticker symbols to load.
        analysis_date: ISO date ceiling (inclusive); None → latest available bar.

    Returns:
        (ohlcv_by_ticker, resolved_date) where resolved_date is the latest
        bar date across all loaded tickers.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    df_all = _read_parquet_dated(path, analysis_date)  # raises FileNotFoundError if missing

    ohlcv_by_ticker: dict = {}
    resolved_dates:  list = []

    for ticker in tickers:
        df = df_all[df_all["symbol"] == ticker].copy()
        if df.empty:
            log.warning("[prepare] ticker %s not found in parquet — skipped", ticker)
            continue

        df = df.sort_values("date").tail(HISTORY_BARS)
        resolved_dates.append(df["date"].iloc[-1])

        rename = {k: v for k, v in {
            "ropen":  "open",
            "rhigh":  "high",
            "rlow":   "low",
            "rclose": "close",
        }.items() if k in df.columns}
        df = df.rename(columns=rename).set_index("date").sort_index()

        keep = [c for c in ("open", "high", "low", "close") if c in df.columns]
        ohlcv_by_ticker[ticker] = df[keep]

    resolved = (
        max(resolved_dates).strftime("%Y-%m-%d")
        if resolved_dates
        else str(analysis_date or "unknown")
    )
    return ohlcv_by_ticker, resolved


# ---------------------------------------------------------------------------
# Mode B — live download
# ---------------------------------------------------------------------------


def download_ohlcv_live(
    tickers: list[str],
    lookback_days: int = 365,
) -> tuple[dict, str]:
    """
    Download raw OHLCV for each ticker via yfinance.

    Args:
        tickers:       Ticker symbols to download (e.g. ["A2A.MI", "ENI.MI"]).
        lookback_days: Calendar days of history to fetch (default 365).

    Returns:
        (ohlcv_by_ticker, scan_date) where scan_date is today's ISO date.
        Tickers that fail to download are silently skipped (warning logged).
    """
    end_dt    = datetime.today()
    start_dt  = end_dt - timedelta(days=lookback_days)
    start     = start_dt.strftime("%Y-%m-%d")
    end       = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")  # end is exclusive in yfinance
    scan_date = end_dt.strftime("%Y-%m-%d")

    log.info("[prepare] downloading OHLCV for %d tickers (%s → %s)", len(tickers), start, end)

    ohlcv_by_ticker: dict = {}
    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                multi_level_index=False,
                progress=False,
            )
            if df.empty:
                log.warning("[prepare] no data returned for %s — skipped", ticker)
                continue
            ohlcv_by_ticker[ticker] = df
        except Exception as exc:
            log.warning("[prepare] download failed for %s: %s", ticker, exc)

    return ohlcv_by_ticker, scan_date
