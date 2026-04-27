"""
techa/indicators/volatility.py — Volatility indicators via ta-lib (+ manual HV).

Public API
----------
compute_volatility(h, l, c) -> dict
    All inputs are float64 arrays from _adapter.to_numpy_ohlcv().
    Returns a flat dict of last-bar volatility scalars.

ta-lib functions used: ATR, NATR, BBANDS.
Manual: historical volatility (annualised std of log returns) — no ta-lib equivalent.

Notes
-----
NATR replaces the previous manual atr_pct = atr / close * 100.
HV is computed over the last HV_PERIOD log returns; returns NaN if fewer bars available.
"""

from __future__ import annotations

import numpy as np
import talib

from techa.indicators._adapter import last_valid

__all__ = ["compute_volatility"]

_ATR_PERIOD = 14
_BB_PERIOD  = 20
_BB_NBDEV   = 2.0
_HV_PERIOD  = 20


def compute_volatility(
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
) -> dict:
    """
    Compute last-bar volatility indicators.

    Args:
        h, l, c: float64 arrays (high, low, close) from to_numpy_ohlcv().

    Returns:
        Flat dict of scalars. NaN where lookback is not satisfied.
    """
    atr  = talib.ATR(h, l, c, timeperiod=_ATR_PERIOD)
    natr = talib.NATR(h, l, c, timeperiod=_ATR_PERIOD)
    bb_upper, bb_mid, bb_lower = talib.BBANDS(
        c,
        timeperiod=_BB_PERIOD,
        nbdevup=_BB_NBDEV,
        nbdevdn=_BB_NBDEV,
        matype=0,
    )

    # Historical volatility: annualised std of log returns (no direct ta-lib equivalent)
    log_ret = np.diff(np.log(np.where(c > 0, c, np.nan)))
    if np.sum(~np.isnan(log_ret)) >= _HV_PERIOD:
        hv = float(np.nanstd(log_ret[-_HV_PERIOD:], ddof=1) * np.sqrt(252) * 100)
    else:
        hv = float("nan")

    upper_v = last_valid(bb_upper)
    mid_v   = last_valid(bb_mid)
    lower_v = last_valid(bb_lower)
    price   = float(c[-1])

    band     = upper_v - lower_v if not (np.isnan(upper_v) or np.isnan(lower_v)) else float("nan")
    bb_width = band / mid_v if not np.isnan(band) and mid_v and mid_v != 0.0 else float("nan")
    bb_pct_b = (price - lower_v) / band if not np.isnan(band) and band != 0.0 else float("nan")

    return {
        "atr":         last_valid(atr),
        "atr_pct":     last_valid(natr),
        "bb_upper":    upper_v,
        "bb_mid":      mid_v,
        "bb_lower":    lower_v,
        "bb_width":    bb_width,
        "bb_pct_b":    bb_pct_b,
        "hist_vol_20d": hv,
    }
