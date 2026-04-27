"""
techa/indicators/trend.py — Trend indicators via ta-lib.

Public API
----------
compute_trend(o, h, l, c) -> dict
    All inputs are float64 arrays from _adapter.to_numpy_ohlcv().
    Returns a flat dict of last-bar trend scalars.

ta-lib functions used: SMA, EMA, ADX, PLUS_DI, MINUS_DI.
Manual: OLS slope + R² on SMA20 (no ta-lib equivalent).
"""

from __future__ import annotations

import numpy as np
import talib

from techa.indicators._adapter import last_valid
from techa.utils import ols_slope_r2

__all__ = ["compute_trend"]

_SMA_PERIODS = (20, 50, 200)
_EMA_PERIODS = (20, 50)
_ADX_PERIOD = 14
_SLOPE_LOOKBACK = 10


def compute_trend(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
) -> dict:
    """
    Compute last-bar trend indicators.

    Args:
        o, h, l, c: float64 arrays (open, high, low, close) from to_numpy_ohlcv().

    Returns:
        Flat dict of scalars. NaN where lookback is not satisfied.
    """
    sma20  = talib.SMA(c, timeperiod=20)
    sma50  = talib.SMA(c, timeperiod=50)
    sma200 = talib.SMA(c, timeperiod=200)
    ema20  = talib.EMA(c, timeperiod=20)
    ema50  = talib.EMA(c, timeperiod=50)
    adx    = talib.ADX(h, l, c, timeperiod=_ADX_PERIOD)
    di_p   = talib.PLUS_DI(h, l, c, timeperiod=_ADX_PERIOD)
    di_m   = talib.MINUS_DI(h, l, c, timeperiod=_ADX_PERIOD)

    price = float(c[-1])
    s20, s50, s200 = last_valid(sma20), last_valid(sma50), last_valid(sma200)
    e20, e50 = last_valid(ema20), last_valid(ema50)

    def _pct_dist(ma: float) -> float:
        return (price - ma) / ma * 100 if not np.isnan(ma) and ma != 0.0 else float("nan")

    # Slope of SMA20 over the last _SLOPE_LOOKBACK bars, normalised to %/bar
    valid_sma20 = sma20[~np.isnan(sma20)]
    if len(valid_sma20) >= _SLOPE_LOOKBACK:
        window = valid_sma20[-_SLOPE_LOOKBACK:]
        raw_slope, r2 = ols_slope_r2(window)
        ref = float(window[-1])
        slope_pct = raw_slope / ref * 100 if ref != 0.0 else float("nan")
    else:
        slope_pct, r2 = float("nan"), float("nan")

    return {
        "sma20":          s20,
        "sma50":          s50,
        "sma200":         s200,
        "ema20":          e20,
        "ema50":          e50,
        "dist_sma20_pct":  _pct_dist(s20),
        "dist_sma50_pct":  _pct_dist(s50),
        "dist_sma200_pct": _pct_dist(s200),
        "slope_sma20":    slope_pct,
        "slope_sma20_r2": r2,
        "adx":            last_valid(adx),
        "di_plus":        last_valid(di_p),
        "di_minus":       last_valid(di_m),
        "golden_cross":   bool(s50 > s200) if not (np.isnan(s50) or np.isnan(s200)) else False,
    }
