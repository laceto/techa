"""
techa/actuarial/inforce/analysis.py — In-force portfolio health KPIs.

Persistency
-----------
lapse_rate_t        = lapses_t / pif_t   (policies in force at start of period)
persistency_rate    = 1 − avg_lapse_rate
mortality_rate_ppm  = deaths_t / pif_t × 1_000_000   (mortality per million exposed)

Portfolio growth
----------------
pif_growth_pct      = (pif_latest / pif_first − 1) × 100  (total period)
gwp_growth_pct      = (gwp_latest / gwp_first − 1) × 100
new_business_ratio  = new_policies_t / pif_t  (latest period)
CAGR annualised using periods_per_year.

Reserves and solvency (latest period)
--------------------------------------
reserve_adequacy_ratio  = own_funds / SCR    (Solvency II coverage ratio)
bel_to_gwp_ratio        = best_estimate_liability / annualised GWP
risk_margin_ratio       = risk_margin / best_estimate_liability
bel_trend_slope, bel_trend_r2  — OLS trend of BEL over time

Solvency coverage: ≥ 150% = adequate | 100–149% = watch | < 100% = breach
"""

from __future__ import annotations

import math

import numpy as np

from techa.utils import ols_slope_r2

__all__ = ["compute_inforce_analysis"]

nan = float("nan")


def _cagr(start: float, end: float, n_periods: int, periods_per_year: int) -> float:
    if start <= 0 or end <= 0 or n_periods <= 0:
        return nan
    years = n_periods / periods_per_year
    return (end / start) ** (1.0 / years) - 1.0


def _solvency_status(ratio: float) -> str:
    if math.isnan(ratio):
        return "unknown"
    if ratio >= 1.50:
        return "adequate"
    if ratio >= 1.00:
        return "watch"
    return "breach"


def _safe(val: float) -> float:
    return val if not math.isnan(val) else nan


def compute_inforce_analysis(data: dict) -> dict:
    """
    Compute in-force portfolio health KPIs.

    Args:
        data: Validated dict from validate_inforce_data().

    Returns:
        Flat dict of in-force metrics.
    """
    periods      = data["_periods"]
    ppy          = data["_periods_per_year"]
    n            = len(periods)

    # ── Per-period lapse and mortality rates ──────────────────────────────────
    lapse_rates: list[float] = []
    mort_rates:  list[float] = []

    for p in periods:
        pif = p["pif"]
        if not math.isnan(pif) and pif > 0:
            if not math.isnan(p["lapses"]):
                lapse_rates.append(p["lapses"] / pif)
            if not math.isnan(p["deaths"]):
                mort_rates.append(p["deaths"] / pif * 1_000_000)

    avg_lapse = float(np.mean(lapse_rates)) if lapse_rates else nan
    avg_mort  = float(np.mean(mort_rates))  if mort_rates  else nan

    # Lapse trend
    if len(lapse_rates) >= 2:
        lapse_slope, lapse_r2 = ols_slope_r2(np.array(lapse_rates, dtype=float))
    else:
        lapse_slope, lapse_r2 = nan, nan

    def _lapse_trend(slope, r2):
        if math.isnan(slope) or r2 < 0.20:
            return "stable"
        if slope > 0.002:
            return "deteriorating"
        if slope < -0.002:
            return "improving"
        return "stable"

    # ── Portfolio growth ──────────────────────────────────────────────────────
    pif_series = [p["pif"]  for p in periods if not math.isnan(p["pif"])]
    gwp_series = [p["gwp"]  for p in periods if not math.isnan(p["gwp"])]

    pif_latest  = pif_series[-1]  if pif_series else nan
    pif_first   = pif_series[0]   if pif_series else nan
    gwp_latest  = gwp_series[-1]  if gwp_series else nan
    gwp_first   = gwp_series[0]   if gwp_series else nan

    n_pif = len(pif_series)
    pif_cagr = _cagr(pif_first, pif_latest, n_pif - 1, ppy) if n_pif >= 2 else nan
    gwp_cagr = _cagr(gwp_first, gwp_latest, len(gwp_series) - 1, ppy) if len(gwp_series) >= 2 else nan

    # New business ratio (latest period)
    latest = periods[-1]
    nbr = (latest["new_pol"] / latest["pif"]
           if not math.isnan(latest["new_pol"]) and not math.isnan(latest["pif"]) and latest["pif"] > 0
           else nan)

    # ── Reserves and solvency (latest period with data) ──────────────────────
    bel_series   = [p["bel"]       for p in periods if not math.isnan(p["bel"])]
    rm_series    = [p["risk_margin"]for p in periods if not math.isnan(p["risk_margin"])]
    scr_series   = [p["scr"]       for p in periods if not math.isnan(p["scr"])]
    of_series    = [p["own_funds"] for p in periods if not math.isnan(p["own_funds"])]

    bel_latest   = bel_series[-1]  if bel_series  else nan
    rm_latest    = rm_series[-1]   if rm_series   else nan
    scr_latest   = scr_series[-1]  if scr_series  else nan
    of_latest    = of_series[-1]   if of_series   else nan

    sol_ratio    = of_latest / scr_latest if (not math.isnan(of_latest) and
                                               not math.isnan(scr_latest) and
                                               scr_latest > 0) else nan
    bel_to_gwp   = (bel_latest / (gwp_latest * ppy)
                    if not math.isnan(bel_latest) and not math.isnan(gwp_latest) and gwp_latest > 0
                    else nan)
    rm_ratio     = (rm_latest / bel_latest
                    if not math.isnan(rm_latest) and not math.isnan(bel_latest) and bel_latest > 0
                    else nan)

    # BEL trend
    if len(bel_series) >= 2:
        bel_slope, bel_r2 = ols_slope_r2(np.array(bel_series, dtype=float))
    else:
        bel_slope, bel_r2 = nan, nan

    def _r(v, dp=4):
        return round(v, dp) if not math.isnan(v) else nan

    return {
        "period_count":          n,
        "pif_latest":            _r(pif_latest, 0),
        "gwp_latest":            _r(gwp_latest, 2),
        "pif_cagr":              _r(pif_cagr,   4),
        "gwp_cagr":              _r(gwp_cagr,   4),
        "avg_lapse_rate":        _r(avg_lapse,  4),
        "persistency_rate":      _r(1.0 - avg_lapse, 4) if not math.isnan(avg_lapse) else nan,
        "lapse_trend_slope":     _r(lapse_slope, 5),
        "lapse_trend_r2":        _r(lapse_r2,   4),
        "lapse_trend_direction": _lapse_trend(lapse_slope, lapse_r2),
        "avg_mortality_rate_ppm":_r(avg_mort,   2),
        "new_business_ratio":    _r(nbr,        4),
        "bel_latest":            _r(bel_latest, 2),
        "risk_margin_latest":    _r(rm_latest,  2),
        "scr_latest":            _r(scr_latest, 2),
        "solvency_coverage_ratio": _r(sol_ratio, 4),
        "solvency_status":       _solvency_status(sol_ratio),
        "bel_to_annualised_gwp": _r(bel_to_gwp, 4),
        "risk_margin_ratio":     _r(rm_ratio,   4),
        "bel_trend_slope":       _r(bel_slope,  2),
        "bel_trend_r2":          _r(bel_r2,     4),
    }
