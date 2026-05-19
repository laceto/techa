"""
techa/insurance/growth.py — Insurance portfolio growth and quality KPIs.

Public API
----------
compute_growth(arrays, *, periods_per_year, trend_lookback) -> dict
    Input is dict[str, np.ndarray] from _adapter.to_numpy_financials().
    Returns a flat dict of growth, CAGR, and portfolio quality scalars.

KPIs computed
-------------
Portfolio size (latest period):
  gwp_latest            — gross written premium (£) this period
  nwp_latest            — net written premium (£) this period

Period-over-period growth:
  premium_growth_pp     — (GWP_t / GWP_t-1 − 1) × 100  (%)
  claims_growth_pp      — (claims_t / claims_t-1 − 1) × 100  (%)

Year-over-year growth (requires ≥ periods_per_year + 1 history):
  premium_growth_yoy    — (GWP_t / GWP_t-N − 1) × 100  where N = periods_per_year
  claims_growth_yoy     — same basis

Long-run:
  gwp_cagr              — compound annual growth rate of GWP over full history (%)

Trend (OLS over last trend_lookback periods):
  gwp_trend             — OLS slope of GWP (£/period)
  gwp_trend_r2          — R²

Portfolio quality (optional — NaN when policy-count columns absent):
  avg_premium           — GWP / policies_in_force  (£/policy)
  lapse_rate            — lapsed_policies / prior_policies_in_force × 100  (%)
  new_business_ratio    — new_policies / policies_in_force × 100  (%)
"""

from __future__ import annotations

import numpy as np

from techa.insurance._adapter import last_valid, nan_div
from techa.utils import ols_slope_r2

__all__ = ["compute_growth"]

_TREND_LOOKBACK = 8


def _series_trend(series: np.ndarray, lookback: int) -> tuple[float, float]:
    valid = series[~np.isnan(series)]
    if len(valid) < max(2, lookback):
        return float("nan"), float("nan")
    return ols_slope_r2(valid[-lookback:])


def _pct_change(arr: np.ndarray, lag: int) -> float:
    """(arr[-1] / arr[-1-lag] - 1) * 100 or NaN if insufficient history / zeros."""
    valid = arr[~np.isnan(arr)]
    if len(valid) < lag + 1:
        return float("nan")
    prev = valid[-(lag + 1)]
    curr = valid[-1]
    if np.isnan(prev) or prev == 0.0:
        return float("nan")
    return float((curr / prev - 1.0) * 100.0)


def _cagr(arr: np.ndarray, periods_per_year: int) -> float:
    """
    Compound annual growth rate over the full valid history (%).
    Uses (end / start) ^ (periods_per_year / (n-1)) - 1.
    """
    valid = arr[~np.isnan(arr)]
    n = len(valid)
    if n < 2 or valid[0] == 0.0 or valid[0] < 0:
        return float("nan")
    ratio = valid[-1] / valid[0]
    if ratio <= 0:
        return float("nan")
    years = (n - 1) / periods_per_year
    if years <= 0:
        return float("nan")
    return float((ratio ** (1.0 / years) - 1.0) * 100.0)


def compute_growth(
    arrays: dict[str, np.ndarray],
    *,
    periods_per_year: int = 4,
    trend_lookback: int = _TREND_LOOKBACK,
) -> dict:
    """
    Compute portfolio growth and quality KPIs.

    Args:
        arrays:           Dict from to_numpy_financials().
        periods_per_year: Accounting frequency — 4 (quarterly, default) or 12 (monthly).
                          Used for YoY comparisons and CAGR annualisation.
        trend_lookback:   Periods for OLS trend fits. Default 8.

    Returns:
        Flat dict of scalars. NaN where insufficient history or missing data.
    """
    gwp              = arrays["gwp"]
    claims_incurred  = arrays["claims_incurred"]
    nwp              = arrays["nwp"]
    policies         = arrays["policies_in_force"]
    new_pol          = arrays["new_policies"]
    lapsed_pol       = arrays["lapsed_policies"]

    # ── Latest values ────────────────────────────────────────────────────────
    gwp_latest = last_valid(gwp)
    nwp_latest = last_valid(nwp)

    # ── Period-over-period growth ────────────────────────────────────────────
    premium_growth_pp = _pct_change(gwp, 1)
    claims_growth_pp  = _pct_change(claims_incurred, 1)

    # ── Year-over-year growth ────────────────────────────────────────────────
    premium_growth_yoy = _pct_change(gwp, periods_per_year)
    claims_growth_yoy  = _pct_change(claims_incurred, periods_per_year)

    # ── Long-run CAGR ────────────────────────────────────────────────────────
    gwp_cagr = _cagr(gwp, periods_per_year)

    # ── GWP trend ────────────────────────────────────────────────────────────
    gwp_slope, gwp_r2 = _series_trend(gwp, trend_lookback)

    # ── Portfolio quality (optional columns) ─────────────────────────────────
    pol_latest  = last_valid(policies)
    new_latest  = last_valid(new_pol)
    lapse_latest = last_valid(lapsed_pol)

    avg_premium = nan_div(gwp_latest, pol_latest)

    # Lapse rate: lapsed this period / in-force at start of period (prior period)
    valid_pol = policies[~np.isnan(policies)]
    prior_pol = float(valid_pol[-2]) if len(valid_pol) >= 2 else float("nan")
    lapse_rate = nan_div(lapse_latest, prior_pol) * 100.0

    new_business_ratio = nan_div(new_latest, pol_latest) * 100.0

    return {
        # Latest values
        "gwp_latest":          gwp_latest,
        "nwp_latest":          nwp_latest,
        # Period-over-period
        "premium_growth_pp":   premium_growth_pp,
        "claims_growth_pp":    claims_growth_pp,
        # Year-over-year
        "premium_growth_yoy":  premium_growth_yoy,
        "claims_growth_yoy":   claims_growth_yoy,
        # Long-run
        "gwp_cagr":            gwp_cagr,
        # Trend
        "gwp_trend":           gwp_slope,
        "gwp_trend_r2":        gwp_r2,
        # Portfolio quality
        "avg_premium":         avg_premium,
        "lapse_rate":          lapse_rate,
        "new_business_ratio":  new_business_ratio,
    }
