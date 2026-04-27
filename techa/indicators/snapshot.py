"""
techa/indicators/snapshot.py — Last-bar indicator snapshot builder.

Responsibility: thin orchestrator.
    - Input validation and MIN_BARS enforcement
    - Delegates all indicator computation to domain modules (trend, momentum,
      volatility, volume) via to_numpy_ohlcv() from _adapter
    - Convenience loader build_snapshot_from_parquet for parquet-based workflows

Public API
----------
build_snapshot(ohlcv, *, nan_to_none=False) -> dict
    Compute a last-bar technical indicator snapshot from raw OHLCV history.

build_snapshot_from_parquet(ticker, data_path, *, ticker_col, date_col, nan_to_none) -> dict
    Load a parquet file, filter to ticker, and return build_snapshot().

compute_last_bar(ohlcv) -> dict
    Deprecated alias for build_snapshot().
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from techa.indicators._adapter import to_numpy_ohlcv, MIN_BARS
from techa.indicators.trend import compute_trend
from techa.indicators.momentum import compute_momentum
from techa.indicators.volatility import compute_volatility
from techa.indicators.volume import compute_volume

__all__ = ["build_snapshot", "build_snapshot_from_parquet", "compute_last_bar"]

log = logging.getLogger(__name__)


def build_snapshot(
    ohlcv: pd.DataFrame,
    *,
    nan_to_none: bool = False,
) -> dict:
    """
    Compute a last-bar technical indicator snapshot from raw OHLCV history.

    Args:
        ohlcv:       DataFrame with columns open/high/low/close/volume (case-insensitive).
                     Must be sorted ascending by date. Requires at least MIN_BARS rows.
        nan_to_none: Replace float NaN with None for JSON-serialisable output. Default False.

    Returns:
        Flat dict of scalars (float, str, bool). NaN appears as float("nan") by default,
        or None when nan_to_none=True.

    Raises:
        ValueError: Missing OHLCV columns, or fewer than MIN_BARS rows.
    """
    if len(ohlcv) < MIN_BARS:
        raise ValueError(
            f"build_snapshot requires at least {MIN_BARS} bars; got {len(ohlcv)}. "
            "Provide a longer history or lower MIN_BARS in _adapter.py."
        )

    o, h, l, c, v = to_numpy_ohlcv(ohlcv)

    result: dict = {"price": float(c[-1])}
    result.update(compute_trend(o, h, l, c))
    result.update(compute_momentum(c, h, l))
    result.update(compute_volatility(h, l, c))
    result.update(compute_volume(h, l, c, v))

    if nan_to_none:
        result = {
            k: (None if isinstance(val, float) and np.isnan(val) else val)
            for k, val in result.items()
        }

    return result


def build_snapshot_from_parquet(
    ticker: str,
    data_path: Union[str, Path],
    *,
    ticker_col: str = "symbol",
    date_col: str = "date",
    nan_to_none: bool = False,
) -> dict:
    """
    Load a parquet file, filter to ticker, sort by date, and return build_snapshot().

    Args:
        ticker:      Ticker symbol to filter on.
        data_path:   Path to parquet with raw OHLCV columns (open/high/low/close/volume).
        ticker_col:  Column holding the ticker identifier. Default "symbol".
        date_col:    Column holding the date. Default "date".
        nan_to_none: Passed through to build_snapshot. Default False.

    Returns:
        dict — same schema as build_snapshot().

    Raises:
        FileNotFoundError: data_path does not exist.
        ValueError: ticker not found, or OHLCV columns missing.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Parquet not found: {data_path}")

    df = pd.read_parquet(data_path)

    if ticker_col not in df.columns:
        raise ValueError(
            f"Ticker column '{ticker_col}' not found in {data_path}. "
            f"Available columns: {list(df.columns[:10])}."
        )

    df_ticker = df[df[ticker_col] == ticker].copy()
    if df_ticker.empty:
        sample = df[ticker_col].unique()[:10].tolist()
        raise ValueError(
            f"Ticker '{ticker}' not found in {data_path}. Sample available: {sample}."
        )

    if date_col in df_ticker.columns:
        df_ticker = df_ticker.sort_values(date_col).set_index(date_col)

    log.info("build_snapshot_from_parquet: %s (%d bars)", ticker, len(df_ticker))
    return build_snapshot(df_ticker, nan_to_none=nan_to_none)


def compute_last_bar(ohlcv: pd.DataFrame) -> dict:
    """Deprecated. Use build_snapshot()."""
    warnings.warn(
        "compute_last_bar is deprecated; use build_snapshot(ohlcv).",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_snapshot(ohlcv)
