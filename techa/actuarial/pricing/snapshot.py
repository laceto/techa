"""
techa/actuarial/pricing/snapshot.py — Reinsurance deal pricing snapshot orchestrator.

Public API
----------
build_pricing_snapshot(pricing_data, *, nan_to_none) -> dict
    Compute reinsurance deal pricing KPIs from projected cash flows.

Input pricing_data (dict)
--------------------------
Required:
    cash_flows   — list of yearly dicts. Each must contain:
                     ceded_premium   — premium ceded to reinsurer (£).
                     ceded_claims    — claims expected to be ceded (£).

Optional per year:
    expenses         — treaty administration expenses (£). Default 0.
    commission       — reinsurance commission paid back to cedant (£). Default 0.
    profit_commission — profit commission paid back (£). Default 0.

Optional top-level:
    treaty_type      — "quota_share" | "surplus" | "excess_of_loss" | "stop_loss".
    discount_rate    — risk-free or hurdle rate. Default 0.05.
    allocated_capital — capital held against treaty for return-on-capital calc.

Output snapshot keys
--------------------
term_years, discount_rate,
total_ceded_premium, total_ceded_claims, total_expenses, total_commission, total_profit,
npv, payback_period_years,
loss_ratio, expense_ratio, commission_ratio, profit_margin,
break_even_loss_ratio,
stress_ae25_loss_ratio, stress_ae50_loss_ratio,
stress_ae25_profit_margin, stress_ae50_profit_margin,
pricing_adequacy (adequate/marginal/inadequate),
cumulative_cash_flows (list),
treaty_type.
"""

from __future__ import annotations

import math

from techa.actuarial.pricing._adapter import validate_pricing_data
from techa.actuarial.pricing.analysis import compute_pricing_analysis

__all__ = ["build_pricing_snapshot"]


def build_pricing_snapshot(pricing_data: dict, *, nan_to_none: bool = False) -> dict:
    """
    Compute a reinsurance deal pricing snapshot from projected cash flows.

    Args:
        pricing_data: Dict — see module docstring for schema.
        nan_to_none:  Replace float NaN with None for JSON-serialisable output.

    Returns:
        Flat dict of pricing KPIs (~21 keys).

    Raises:
        ValueError: If required fields are missing or cash_flows list is empty.

    Example:
        from techa.actuarial.pricing import build_pricing_snapshot

        snap = build_pricing_snapshot({
            "treaty_type": "quota_share",
            "discount_rate": 0.05,
            "cash_flows": [
                {"ceded_premium": 500_000, "ceded_claims": 200_000,
                 "expenses": 25_000, "commission": 100_000},
                {"ceded_premium": 520_000, "ceded_claims": 215_000,
                 "expenses": 26_000, "commission": 104_000},
                {"ceded_premium": 540_000, "ceded_claims": 225_000,
                 "expenses": 27_000, "commission": 108_000},
            ],
        }, nan_to_none=True)

        print(snap["loss_ratio"])       # ~0.416
        print(snap["profit_margin"])    # ~0.385
        print(snap["pricing_adequacy"]) # "adequate"
    """
    data   = validate_pricing_data(pricing_data)
    result = compute_pricing_analysis(data)
    result["treaty_type"] = data.get("treaty_type", "unknown")

    if nan_to_none:
        result = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in result.items()
        }
    return result
