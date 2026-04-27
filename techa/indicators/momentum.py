"""
techa/indicators/momentum.py — Momentum indicators via ta-lib.

Public API
----------
compute_momentum(c, h, l) -> dict
    All inputs are float64 arrays from _adapter.to_numpy_ohlcv().
    Returns a flat dict of last-bar momentum scalars.

ta-lib functions used: RSI, MACD, STOCH, STOCHF, ROC.

Notes
-----
RSI: Wilder's smoothing seeded with the plain mean of the first period (ta-lib canonical).
     Constant-price series returns all-NaN; rsi_zone will be "n/a".
STOCH:  Slow stochastic — fastk_period=14, slowk_period=3, slowd_period=3, both SMA.
STOCHF: Fast stochastic — fastk_period=5, fastd_period=3 (SMA). Reacts faster than STOCH
        but is noisier; useful for identifying short-term momentum turns.
ROC: (close[t] - close[t-n]) / close[t-n] * 100.
"""

from __future__ import annotations

import numpy as np
import talib

from techa.indicators._adapter import last_valid

__all__ = ["compute_momentum"]

_RSI_PERIOD   = 14
_MACD_FAST    = 12
_MACD_SLOW    = 26
_MACD_SIGNAL  = 9
_STOCH_K      = 14
_STOCH_D      = 3
_STOCHF_K     = 5
_STOCHF_D     = 3


def compute_momentum(
    c: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
) -> dict:
    """
    Compute last-bar momentum indicators.

    Args:
        c, h, l: float64 arrays (close, high, low) from to_numpy_ohlcv().

    Returns:
        Flat dict of scalars. NaN where lookback is not satisfied.
    """
    rsi_arr = talib.RSI(c, timeperiod=_RSI_PERIOD)
    macd_line, macd_sig, macd_hist = talib.MACD(
        c,
        fastperiod=_MACD_FAST,
        slowperiod=_MACD_SLOW,
        signalperiod=_MACD_SIGNAL,
    )
    stoch_k, stoch_d = talib.STOCH(
        h, l, c,
        fastk_period=_STOCH_K,
        slowk_period=_STOCH_D, slowk_matype=0,
        slowd_period=_STOCH_D, slowd_matype=0,
    )
    stochf_k, stochf_d = talib.STOCHF(
        h, l, c,
        fastk_period=_STOCHF_K,
        fastd_period=_STOCHF_D,
        fastd_matype=0,
    )
    roc10 = talib.ROC(c, timeperiod=10)
    roc20 = talib.ROC(c, timeperiod=20)

    rsi_v = last_valid(rsi_arr)

    if np.isnan(rsi_v):
        rsi_zone = "n/a"
    elif rsi_v >= 70:
        rsi_zone = "overbought"
    elif rsi_v <= 30:
        rsi_zone = "oversold"
    else:
        rsi_zone = "neutral"

    chg_1d = float((c[-1] / c[-2] - 1) * 100) if len(c) >= 2 else float("nan")
    chg_5d = float((c[-1] / c[-6] - 1) * 100) if len(c) >= 6 else float("nan")

    return {
        "rsi":        rsi_v,
        "rsi_zone":   rsi_zone,
        "macd":       last_valid(macd_line),
        "macd_signal": last_valid(macd_sig),
        "macd_hist":  last_valid(macd_hist),
        "stoch_k":      last_valid(stoch_k),
        "stoch_d":      last_valid(stoch_d),
        "stoch_fast_k": last_valid(stochf_k),
        "stoch_fast_d": last_valid(stochf_d),
        "roc_10d":      last_valid(roc10),
        "roc_20d":    last_valid(roc20),
        "chg_1d":     chg_1d,
        "chg_5d":     chg_5d,
    }
