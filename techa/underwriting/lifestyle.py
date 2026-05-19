"""
techa/underwriting/lifestyle.py — Smoking, alcohol, and exercise assessment.

Public API
----------
compute_lifestyle(data) -> dict
    Input is the validated dict from _adapter.validate_questionnaire().
    Returns smoking status/pack-years/loading, alcohol risk/loading.

Smoking loading schedule
------------------------
never:                         0%
ex, years_quit ≥ 5:            0%  (standard rates — treated as non-smoker)
ex, years_quit 3–4:            25%
ex, years_quit 1–2:            50%
ex, years_quit < 1:            100%
current, ≤ 10 cigarettes/day: 100%
current, 11–20/day:            125%
current, > 20/day:             150%

High pack-years surcharge (even in ex-smokers, elevated COPD/cancer risk):
pack_years ≥ 30: +25% surcharge on top of base loading.
pack_years ≥ 50: +50% surcharge.

Alcohol loading schedule (units per week, UK guidelines)
----------------------------------------------------------
≤ 14 units/week (low risk):    0%
14–21:                         +10%
22–28:                         +25%
29–35:                         +50%
> 35:                          +100%  (may require GP report; consider postpone > 50 units)

Alcohol risk categories
-----------------------
≤ 14: low | 14–21: moderate | 22–35: high | > 35: very_high
"""

from __future__ import annotations

import math

__all__ = ["compute_lifestyle"]

nan = float("nan")


def _smoking_loading(status: str, cpd: float, pack_years: float, years_quit: float) -> float:
    """Base loading from smoking status + high pack-years surcharge."""
    if status == "never":
        return 0.0

    if status == "ex":
        yq = years_quit if not math.isnan(years_quit) else 0.0
        if yq >= 5:
            base = 0.0
        elif yq >= 3:
            base = 25.0
        elif yq >= 1:
            base = 50.0
        else:
            base = 100.0
    elif status == "current":
        daily = cpd if not math.isnan(cpd) else 20.0
        if daily <= 10:
            base = 100.0
        elif daily <= 20:
            base = 125.0
        else:
            base = 150.0
    else:
        return nan

    # Pack-years surcharge
    py = pack_years if not math.isnan(pack_years) else 0.0
    surcharge = 50.0 if py >= 50 else (25.0 if py >= 30 else 0.0)

    return base + surcharge


def _alcohol_risk(units: float) -> str:
    if units <= 14: return "low"
    if units <= 21: return "moderate"
    if units <= 35: return "high"
    return "very_high"


def _alcohol_loading(units: float) -> float:
    if units <= 14: return 0.0
    if units <= 21: return 10.0
    if units <= 28: return 25.0
    if units <= 35: return 50.0
    return 100.0


def compute_lifestyle(data: dict) -> dict:
    """
    Assess smoking and alcohol risk.

    Args:
        data: Validated questionnaire dict from validate_questionnaire().

    Returns:
        Flat dict with smoking fields, alcohol fields, and individual loadings.
    """
    status    = data.get("smoking_status", "unknown")
    pack_yrs  = data.get("pack_years", nan)
    cpd       = data.get("cigarettes_per_day", nan)
    yrs_quit  = data.get("years_quit", nan)
    alcohol   = data.get("alcohol_units_per_week", nan)

    if pack_yrs is None: pack_yrs = nan
    if cpd      is None: cpd      = nan
    if yrs_quit is None: yrs_quit = nan
    if alcohol  is None: alcohol  = nan

    pack_yrs = float(pack_yrs)
    cpd      = float(cpd)
    yrs_quit = float(yrs_quit)
    alcohol  = float(alcohol)

    smoke_load = _smoking_loading(status, cpd, pack_yrs, yrs_quit)

    alc_risk  = _alcohol_risk(alcohol)  if not math.isnan(alcohol) else "unknown"
    alc_load  = _alcohol_loading(alcohol) if not math.isnan(alcohol) else nan

    return {
        "smoking_status":       status,
        "pack_years":           pack_yrs,
        "cigarettes_per_day":   cpd,
        "years_quit":           yrs_quit,
        "smoking_loading_pct":  smoke_load,
        "alcohol_units_per_week": alcohol,
        "alcohol_risk":         alc_risk,
        "alcohol_loading_pct":  alc_load,
    }
