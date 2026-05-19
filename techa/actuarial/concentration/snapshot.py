"""
techa/actuarial/concentration/snapshot.py — Portfolio concentration snapshot orchestrator.

Public API
----------
build_concentration_snapshot(portfolio_context, *, nan_to_none) -> dict
    Assess the concentration risk of a single policy within the in-force book.

Input portfolio_context (dict)
-------------------------------
Required:
    portfolio_total_sa      — total sum assured of the in-force book (£), must be > 0.
    portfolio_policy_count  — number of in-force policies, must be > 0.
    policy_sum_assured      — this policy's sum assured (£), must be > 0.

Optional:
    portfolio_gwp           — gross written premium of the book (£). Defaults to 0.0.
    policy_premium_annual   — this policy's annual premium (£). Defaults to 0.0.

Output snapshot keys
--------------------
policy_sum_assured          — float, pass-through
portfolio_total_sa          — float, pass-through
portfolio_policy_count      — int, pass-through
sa_concentration_pct        — float: policy_sum_assured / portfolio_total_sa × 100
avg_policy_sa               — float: portfolio_total_sa / portfolio_policy_count
sa_multiple_of_average      — float: policy_sum_assured / avg_policy_sa
concentration_flag          — bool: True if sa_concentration_pct > 0.5 OR
                                     sa_multiple_of_average > 5.0
concentration_loading_pct   — float: 0 / 5 / 10 / 20 based on multiple band
net_retention_recommendation — float: max(avg × 5, total × 0.001) rounded to nearest 50 000
reinsurance_trigger         — bool: True if policy_sum_assured > net_retention_recommendation
concentration_risk_level    — str: "low" | "elevated" | "high"
"""

from __future__ import annotations

import math

from techa.actuarial.concentration._adapter import validate_concentration_data
from techa.actuarial.concentration.analysis import compute_concentration_analysis

__all__ = ["build_concentration_snapshot"]


def build_concentration_snapshot(
    portfolio_context: dict,
    *,
    nan_to_none: bool = False,
) -> dict:
    """
    Compute a portfolio concentration risk snapshot for a single policy.

    Args:
        portfolio_context: Dict — see module docstring for schema.
        nan_to_none:       Replace float NaN with None for JSON-serialisable output.

    Returns:
        Flat dict of concentration KPIs (11 keys).

    Raises:
        ValueError: If required fields are missing or violate constraints.

    Example:
        from techa.actuarial.concentration import build_concentration_snapshot

        snap = build_concentration_snapshot({
            "portfolio_total_sa":     850_000_000,
            "portfolio_gwp":          12_500_000,
            "portfolio_policy_count": 10_500,
            "policy_sum_assured":     500_000,
            "policy_premium_annual":  3_200,
        }, nan_to_none=True)

        print(snap["sa_concentration_pct"])      # 0.0588
        print(snap["concentration_flag"])        # True
        print(snap["concentration_risk_level"])  # "elevated"
    """
    data   = validate_concentration_data(portfolio_context)
    result = compute_concentration_analysis(data)

    if nan_to_none:
        result = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in result.items()
        }
    return result
