"""
techa/actuarial/inforce/snapshot.py — In-force portfolio health snapshot orchestrator.

Public API
----------
build_inforce_snapshot(inforce_data, *, nan_to_none) -> dict
    Compute in-force portfolio health KPIs from a time-series of portfolio metrics.

Input inforce_data (dict)
--------------------------
Required:
    periods   — list of period records. Each must contain:
                  policies_in_force      — policies active at period end.
                  gross_premium_income   — GWP for the period (£).

Optional per period:
    new_policies, lapses, deaths, maturities  — movement counts.
    best_estimate_liability (or bel)          — Solvency II BEL (£).
    risk_margin                               — Solvency II risk margin (£).
    solvency_capital_requirement (or scr)     — SCR (£).
    own_funds                                 — available capital (£).
    period                                    — string label, e.g. "2024-Q1".

Optional top-level:
    periods_per_year  — 4 for quarterly, 12 for monthly. Default 4.
    product_type      — string label.

Output snapshot keys
--------------------
period_count, pif_latest, gwp_latest,
pif_cagr, gwp_cagr,
avg_lapse_rate, persistency_rate,
lapse_trend_slope, lapse_trend_r2, lapse_trend_direction,
avg_mortality_rate_ppm,
new_business_ratio,
bel_latest, risk_margin_latest, scr_latest,
solvency_coverage_ratio, solvency_status (adequate/watch/breach),
bel_to_annualised_gwp, risk_margin_ratio,
bel_trend_slope, bel_trend_r2,
product_type.
"""

from __future__ import annotations

import math

from techa.actuarial.inforce._adapter import validate_inforce_data
from techa.actuarial.inforce.analysis import compute_inforce_analysis

__all__ = ["build_inforce_snapshot"]


def build_inforce_snapshot(inforce_data: dict, *, nan_to_none: bool = False) -> dict:
    """
    Compute an in-force portfolio health snapshot.

    Args:
        inforce_data: Dict — see module docstring for schema.
        nan_to_none:  Replace float NaN with None for JSON-serialisable output.

    Returns:
        Flat dict of in-force KPIs (~22 keys).

    Raises:
        ValueError: If required fields are missing or periods list is empty.

    Example:
        from techa.actuarial.inforce import build_inforce_snapshot

        snap = build_inforce_snapshot({
            "periods_per_year": 4,
            "periods": [
                {"period": "2023-Q1", "policies_in_force": 10_000,
                 "gross_premium_income": 2_500_000, "lapses": 80, "deaths": 25,
                 "best_estimate_liability": 45_000_000, "scr": 5_000_000,
                 "own_funds": 8_500_000},
                {"period": "2023-Q2", "policies_in_force": 10_400,
                 "gross_premium_income": 2_600_000, "lapses": 75, "deaths": 22,
                 "best_estimate_liability": 46_500_000, "scr": 5_100_000,
                 "own_funds": 9_000_000},
            ],
        }, nan_to_none=True)

        print(snap["solvency_status"])         # "adequate"
        print(snap["solvency_coverage_ratio"]) # 1.76
        print(snap["persistency_rate"])        # 0.9925
    """
    data   = validate_inforce_data(inforce_data)
    result = compute_inforce_analysis(data)
    result["product_type"] = data.get("product_type", "unknown")

    if nan_to_none:
        result = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in result.items()
        }
    return result
