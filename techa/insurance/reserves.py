"""
techa/insurance/reserves.py — Insurance reserve adequacy and claims settlement KPIs.

Public API
----------
compute_reserves(arrays, *, trend_lookback) -> dict
    Input is dict[str, np.ndarray] from _adapter.to_numpy_financials().
    Returns a flat dict of reserve and claims settlement scalars.

KPIs computed
-------------
Current period:
  reserve_adequacy_ratio    — reserve_held / reserve_required  (1.0 = fully funded)
  reserve_adequacy_pct      — × 100
  reserve_surplus           — reserve_held − reserve_required  (£; negative = deficit)
  reserve_to_gwp_pct        — reserve_held / GWP × 100  (reserve depth relative to premium)
  claims_settlement_ratio   — claims_paid / claims_incurred  (0–1; 1 = all settled in period)
  claims_outstanding        — claims_incurred − claims_paid  (£; approximate open claims)
  claims_outstanding_ratio  — claims_outstanding / GWP × 100  (outstanding as % of premium)

Trends (OLS over last trend_lookback periods):
  reserve_adequacy_trend    — slope of reserve_adequacy_ratio (ratio/period)
  reserve_adequacy_trend_r2

Notes
-----
NaN is returned for any KPI whose required column is missing (reserve_held, reserve_required,
claims_paid). The caller should surface these as "unavailable" rather than treating them as zero.
"""

from __future__ import annotations

import numpy as np

from techa.insurance._adapter import last_valid, nan_div
from techa.utils import ols_slope_r2

__all__ = ["compute_reserves"]

_TREND_LOOKBACK = 8


def _series_trend(series: np.ndarray, lookback: int) -> tuple[float, float]:
    valid = series[~np.isnan(series)]
    if len(valid) < max(2, lookback):
        return float("nan"), float("nan")
    return ols_slope_r2(valid[-lookback:])


def compute_reserves(
    arrays: dict[str, np.ndarray],
    *,
    trend_lookback: int = _TREND_LOOKBACK,
) -> dict:
    """
    Compute reserve adequacy and claims settlement KPIs.

    Args:
        arrays:         Dict from to_numpy_financials().
        trend_lookback: Periods for OLS trend fits. Default 8.

    Returns:
        Flat dict of scalars. NaN where required column is missing.
    """
    gwp              = arrays["gwp"]
    claims_incurred  = arrays["claims_incurred"]
    claims_paid      = arrays["claims_paid"]
    reserve_held     = arrays["reserve_held"]
    reserve_required = arrays["reserve_required"]

    # ── Per-period adequacy ratio array ──────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        adequacy_ratios = np.where(
            reserve_required != 0,
            reserve_held / reserve_required,
            np.nan,
        )

    # ── Current-period scalars ────────────────────────────────────────────────
    gwp_last         = last_valid(gwp)
    claims_last      = last_valid(claims_incurred)
    paid_last        = last_valid(claims_paid)
    held_last        = last_valid(reserve_held)
    required_last    = last_valid(reserve_required)

    adequacy_ratio = last_valid(adequacy_ratios)
    adequacy_pct   = adequacy_ratio * 100.0 if not np.isnan(adequacy_ratio) else float("nan")
    surplus        = (
        held_last - required_last
        if not (np.isnan(held_last) or np.isnan(required_last))
        else float("nan")
    )
    reserve_to_gwp = nan_div(held_last, gwp_last) * 100.0

    # Claims settlement: fraction of incurred claims paid within the period
    settlement_ratio = nan_div(paid_last, claims_last)

    # Outstanding claims (approximate open liability; not the same as IBNR reserve)
    outstanding = (
        claims_last - paid_last
        if not (np.isnan(claims_last) or np.isnan(paid_last))
        else float("nan")
    )
    outstanding_ratio = nan_div(outstanding, gwp_last) * 100.0

    # ── Trend ────────────────────────────────────────────────────────────────
    adq_slope, adq_r2 = _series_trend(adequacy_ratios, trend_lookback)

    return {
        # Current period
        "reserve_adequacy_ratio":   adequacy_ratio,
        "reserve_adequacy_pct":     adequacy_pct,
        "reserve_surplus":          surplus,
        "reserve_to_gwp_pct":       reserve_to_gwp,
        "claims_settlement_ratio":  settlement_ratio,
        "claims_outstanding":       outstanding,
        "claims_outstanding_ratio": outstanding_ratio,
        # Trend
        "reserve_adequacy_trend":      adq_slope,
        "reserve_adequacy_trend_r2":   adq_r2,
    }
