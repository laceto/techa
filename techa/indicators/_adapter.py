"""
techa/indicators/_adapter.py — DataFrame-to-numpy conversion for ta-lib inputs.

All domain modules (trend, momentum, volatility, volume) receive np.ndarray[float64].
This module is the single conversion point; no other file in techa.indicators
should call .to_numpy() or .values on OHLCV data.

Public API
----------
to_numpy_ohlcv(ohlcv)  — validate columns, return (o, h, l, c, v) as float64 arrays.
last_valid(arr)         — last non-NaN element of a ta-lib output array.
MIN_BARS                — hard minimum bars required by build_snapshot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["to_numpy_ohlcv", "last_valid", "MIN_BARS"]

MIN_BARS = 30

_REQUIRED_COLS = {"open", "high", "low", "close", "volume"}


def to_numpy_ohlcv(ohlcv: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """
    Validate and convert an OHLCV DataFrame to five C-contiguous float64 arrays.

    Args:
        ohlcv: DataFrame with columns open/high/low/close/volume (case-insensitive).

    Returns:
        (open, high, low, close, volume) as np.ndarray[float64].

    Raises:
        ValueError: If any required column is missing.
    """
    df = ohlcv.copy()
    df.columns = df.columns.str.lower()
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"build_snapshot: missing required OHLCV columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}."
        )
    return (
        df["open"].to_numpy(dtype=np.float64),
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        df["close"].to_numpy(dtype=np.float64),
        df["volume"].to_numpy(dtype=np.float64),
    )


def last_valid(arr: np.ndarray) -> float:
    """
    Return the last non-NaN element of a ta-lib output array.

    ta-lib pads the head of output arrays with NaN for the lookback period.
    The length of the output equals the length of the input, so arr[-1] is
    always the fully-computed last bar — unless the input was shorter than
    the lookback, in which case arr[-1] is NaN.

    Returns float("nan") for an empty array or an all-NaN array.
    """
    if arr is None or len(arr) == 0:
        return float("nan")
    val = arr[-1]
    return float(val) if not np.isnan(val) else float("nan")
