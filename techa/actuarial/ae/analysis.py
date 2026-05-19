"""
techa/actuarial/ae/analysis.py — Actual vs Expected experience analysis.

A/E Ratio
---------
aggregate_ae_ratio = sum(actual) / sum(expected)
period_ae_ratios   = [actual_i / expected_i for each period]

Alert bands
-----------
green:  A/E 90–110%
amber:  A/E 80–90% or 110–130%
red:    A/E < 80% or > 130%

Credibility weighting (Bühlmann)
---------------------------------
Z = min(1.0, sqrt(sum_expected / 1083))
credibility_weighted_ae = Z × aggregate_ae + (1 − Z) × 1.0

Poisson Z-score (statistical significance)
-------------------------------------------
z_score = (actual_total − expected_total) / sqrt(expected_total)
|z_score| > 1.96 → statistically significant at 95% confidence.

A/E trend
---------
OLS slope of period A/E ratios over time.
slope > +0.02/period: deteriorating | slope < −0.02/period: improving | else: stable
"""

from __future__ import annotations

import math

import numpy as np

from techa.utils import ols_slope_r2

__all__ = ["compute_ae_analysis"]

nan = float("nan")

_FULL_CREDIBILITY_N = 1083.0   # expected events for 90% full credibility, CV=0.33


def _ae_alert(ratio: float) -> str:
    if math.isnan(ratio):
        return "unknown"
    if 0.90 <= ratio <= 1.10:
        return "green"
    if 0.80 <= ratio <= 1.30:
        return "amber"
    return "red"


def _trend_direction(slope: float, r2: float) -> str:
    if r2 < 0.20:
        return "stable"
    if slope > 0.02:
        return "deteriorating"
    if slope < -0.02:
        return "improving"
    return "stable"


def compute_ae_analysis(data: dict) -> dict:
    """
    Compute A/E experience monitoring KPIs.

    Args:
        data: Validated dict from validate_ae_data().

    Returns:
        Flat dict of A/E metrics.
    """
    periods = data["_periods"]

    actuals   = [p["actual"]   for p in periods if not math.isnan(p["actual"])]
    expecteds = [p["expected"] for p in periods if not math.isnan(p["expected"])]

    # Pair only periods where both are present
    paired = [
        (p["actual"], p["expected"])
        for p in periods
        if not math.isnan(p["actual"]) and not math.isnan(p["expected"])
    ]

    if not paired:
        return {
            "period_count":             len(periods),
            "actual_total":             nan,
            "expected_total":           nan,
            "aggregate_ae_ratio":       nan,
            "aggregate_ae_pct":         nan,
            "ae_alert_level":           "unknown",
            "credibility_weight":       nan,
            "credibility_weighted_ae":  nan,
            "z_score":                  nan,
            "z_score_significant":      False,
            "cumulative_deviation":     nan,
            "max_ae_ratio":             nan,
            "min_ae_ratio":             nan,
            "periods_above_expected":   0,
            "ae_trend_slope":           nan,
            "ae_trend_r2":              nan,
            "ae_trend_direction":       "unknown",
            "ae_volatility":            nan,
        }

    act_total = sum(a for a, _ in paired)
    exp_total = sum(e for _, e in paired)

    agg_ae = act_total / exp_total if exp_total > 0 else nan

    # Period A/E ratios
    period_ae = [a / e for a, e in paired if e > 0]

    # Credibility (Bühlmann)
    cred_z  = min(1.0, math.sqrt(exp_total / _FULL_CREDIBILITY_N))
    cred_ae = cred_z * agg_ae + (1.0 - cred_z) * 1.0 if not math.isnan(agg_ae) else nan

    # Z-score (Poisson)
    z_score = (act_total - exp_total) / math.sqrt(exp_total) if exp_total > 0 else nan

    # Volatility and trend (need ≥2 periods)
    if len(period_ae) >= 2:
        arr = np.array(period_ae, dtype=float)
        ae_slope, ae_r2 = ols_slope_r2(arr)
        ae_vol = float(np.std(arr, ddof=1))
    else:
        ae_slope, ae_r2, ae_vol = nan, nan, nan

    return {
        "period_count":             len(periods),
        "actual_total":             round(act_total, 2),
        "expected_total":           round(exp_total, 4),
        "aggregate_ae_ratio":       round(agg_ae, 4) if not math.isnan(agg_ae) else nan,
        "aggregate_ae_pct":         round(agg_ae * 100, 1) if not math.isnan(agg_ae) else nan,
        "ae_alert_level":           _ae_alert(agg_ae),
        "credibility_weight":       round(cred_z, 4),
        "credibility_weighted_ae":  round(cred_ae, 4) if not math.isnan(cred_ae) else nan,
        "z_score":                  round(z_score, 3) if not math.isnan(z_score) else nan,
        "z_score_significant":      (not math.isnan(z_score)) and abs(z_score) > 1.96,
        "cumulative_deviation":     round(act_total - exp_total, 2),
        "max_ae_ratio":             round(max(period_ae), 4) if period_ae else nan,
        "min_ae_ratio":             round(min(period_ae), 4) if period_ae else nan,
        "periods_above_expected":   sum(1 for r in period_ae if r > 1.0),
        "ae_trend_slope":           round(ae_slope, 5) if not math.isnan(ae_slope) else nan,
        "ae_trend_r2":              round(ae_r2, 4)    if not math.isnan(ae_r2)    else nan,
        "ae_trend_direction":       _trend_direction(ae_slope, ae_r2)
                                    if not math.isnan(ae_slope) else "unknown",
        "ae_volatility":            round(ae_vol, 4)   if not math.isnan(ae_vol)   else nan,
    }
