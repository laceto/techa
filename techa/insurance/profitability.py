"""
techa/insurance/profitability.py — Insurance profitability KPIs.

Public API
----------
compute_profitability(arrays, *, trend_lookback) -> dict
    Input is dict[str, np.ndarray] from _adapter.to_numpy_financials().
    Returns a flat dict of current-period and trend profitability scalars.

KPIs computed
-------------
Current period:
  loss_ratio             — net claims incurred / GWP
  expense_ratio          — expenses / GWP
  combined_ratio         — loss_ratio + expense_ratio
  underwriting_margin_pct — (1 − combined_ratio) × 100  (positive = profit)
  underwriting_profit    — GWP − claims_incurred − expenses  (£)
  reinsurance_cession_pct — (GWP − NWP) / GWP × 100  (% premium ceded to reinsurers)
  net_claims_ratio       — claims_incurred / NWP  (net-of-reinsurance loss ratio)

Trends (OLS over last trend_lookback periods):
  loss_ratio_trend       — slope of loss_ratio (ratio/period); positive = deteriorating
  loss_ratio_trend_r2    — R² of the slope fit; < 0.3 = noise-dominated
  combined_ratio_trend   — slope of combined_ratio
  combined_ratio_trend_r2
  expense_ratio_trend
  expense_ratio_trend_r2
"""

from __future__ import annotations

import numpy as np

from techa.insurance._adapter import last_valid, nan_div
from techa.utils import ols_slope_r2

__all__ = ["compute_profitability"]

_TREND_LOOKBACK = 8  # periods used for OLS trend (2 years at quarterly frequency)


def _series_trend(series: np.ndarray, lookback: int) -> tuple[float, float]:
    """Fit OLS over the last `lookback` non-NaN values. Returns (slope, r2)."""
    valid = series[~np.isnan(series)]
    if len(valid) < max(2, lookback):
        return float("nan"), float("nan")
    window = valid[-lookback:]
    return ols_slope_r2(window)


def compute_profitability(
    arrays: dict[str, np.ndarray],
    *,
    trend_lookback: int = _TREND_LOOKBACK,
) -> dict:
    """
    Compute last-period and trend profitability KPIs.

    Args:
        arrays:         Dict from to_numpy_financials() — all required and optional arrays.
        trend_lookback: Number of periods for OLS trend fits. Default 8 (2 years quarterly).

    Returns:
        Flat dict of scalars. NaN where insufficient history or missing data.
    """
    gwp              = arrays["gwp"]
    claims_incurred  = arrays["claims_incurred"]
    expenses         = arrays["expenses"]
    nwp              = arrays["nwp"]

    # ── Per-period ratio arrays ───────────────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        loss_ratios    = np.where(gwp != 0, claims_incurred / gwp,    np.nan)
        expense_ratios = np.where(gwp != 0, expenses / gwp,           np.nan)
        combined       = loss_ratios + expense_ratios

    # ── Current-period scalars ────────────────────────────────────────────────
    gwp_last      = last_valid(gwp)
    claims_last   = last_valid(claims_incurred)
    expenses_last = last_valid(expenses)
    nwp_last      = last_valid(nwp)

    lr  = last_valid(loss_ratios)
    er  = last_valid(expense_ratios)
    cr  = last_valid(combined)

    um_pct = (1.0 - cr) * 100.0 if not np.isnan(cr) else float("nan")
    uw_profit = (
        gwp_last - claims_last - expenses_last
        if not any(np.isnan(x) for x in (gwp_last, claims_last, expenses_last))
        else float("nan")
    )

    # Reinsurance cession: how much of GWP was ceded to reinsurers
    reins_cession_pct = (
        nan_div(gwp_last - nwp_last, gwp_last) * 100.0
        if not (np.isnan(gwp_last) or np.isnan(nwp_last))
        else float("nan")
    )

    # Net claims ratio (on NWP basis)
    net_claims_ratio = nan_div(claims_last, nwp_last)

    # ── Trends ───────────────────────────────────────────────────────────────
    lr_slope,  lr_r2  = _series_trend(loss_ratios,    trend_lookback)
    er_slope,  er_r2  = _series_trend(expense_ratios, trend_lookback)
    cr_slope,  cr_r2  = _series_trend(combined,       trend_lookback)

    return {
        # Current period
        "loss_ratio":              lr,
        "expense_ratio":           er,
        "combined_ratio":          cr,
        "underwriting_margin_pct": um_pct,
        "underwriting_profit":     uw_profit,
        "reinsurance_cession_pct": reins_cession_pct,
        "net_claims_ratio":        net_claims_ratio,
        # Trends
        "loss_ratio_trend":           lr_slope,
        "loss_ratio_trend_r2":        lr_r2,
        "expense_ratio_trend":        er_slope,
        "expense_ratio_trend_r2":     er_r2,
        "combined_ratio_trend":       cr_slope,
        "combined_ratio_trend_r2":    cr_r2,
    }
