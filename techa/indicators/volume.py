"""
techa/indicators/volume.py — Volume indicators via ta-lib.

Public API
----------
compute_volume(h, l, c, v) -> dict
    All inputs are float64 arrays from _adapter.to_numpy_ohlcv().
    Returns a flat dict of last-bar volume scalars.

ta-lib functions used: OBV, SMA (for vol_ma20), AD, ADOSC.

Notes
-----
OBV with all-zero volume returns all-zero (not NaN) — correct mathematical behaviour.
vol_vs_ma20: NaN when vol_ma20 is zero (no divide-by-zero).
AD (Chaikin A/D Line): accumulation/distribution based on close location within H-L range.
    All-zero volume produces all-zero AD — not NaN.
ADOSC (Chaikin A/D Oscillator): EMA(AD, fast) - EMA(AD, slow); default 3/10.
"""

from __future__ import annotations

import numpy as np
import talib

from techa.indicators._adapter import last_valid

__all__ = ["compute_volume"]

_VOL_MA_PERIOD = 20
_ADOSC_FAST    = 3
_ADOSC_SLOW    = 10


def compute_volume(
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
) -> dict:
    """
    Compute last-bar volume indicators.

    Args:
        h, l, c, v: float64 arrays (high, low, close, volume) from to_numpy_ohlcv().

    Returns:
        Flat dict of scalars. NaN where lookback is not satisfied.
    """
    obv    = talib.OBV(c, v)
    vol_ma = talib.SMA(v, timeperiod=_VOL_MA_PERIOD)
    ad     = talib.AD(h, l, c, v)
    adosc  = talib.ADOSC(h, l, c, v, fastperiod=_ADOSC_FAST, slowperiod=_ADOSC_SLOW)

    vol_last  = float(v[-1])
    vol_ma_v  = last_valid(vol_ma)
    vol_vs_ma = vol_last / vol_ma_v if not np.isnan(vol_ma_v) and vol_ma_v != 0.0 else float("nan")

    return {
        "volume":      vol_last,
        "vol_vs_ma20": vol_vs_ma,
        "obv":         last_valid(obv),
        "ad":          last_valid(ad),
        "adosc":       last_valid(adosc),
    }
