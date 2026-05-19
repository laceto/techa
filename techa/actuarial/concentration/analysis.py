"""
techa/actuarial/concentration/analysis.py — Portfolio concentration risk KPIs.

Concentration percentage
------------------------
sa_concentration_pct = policy_sum_assured / portfolio_total_sa × 100

Average policy size
-------------------
avg_policy_sa = portfolio_total_sa / portfolio_policy_count

Multiple of average
--------------------
sa_multiple_of_average = policy_sum_assured / avg_policy_sa

Concentration flag
------------------
True  if sa_concentration_pct > 0.5  OR  sa_multiple_of_average > 5.0

Concentration loading
---------------------
multiple ≤ 5   → 0.0 %
5  < m ≤ 10   → 5.0 %
10 < m ≤ 25   → 10.0 %
m > 25         → 20.0 %

Net retention recommendation
-----------------------------
max(avg_policy_sa × 5, portfolio_total_sa × 0.001)
rounded to the nearest 50 000.

Reinsurance trigger
-------------------
True  if policy_sum_assured > net_retention_recommendation

Concentration risk level
------------------------
"low"      — flag is False
"elevated" — flag is True and sa_multiple_of_average ≤ 10
"high"     — flag is True and sa_multiple_of_average > 10
"""

from __future__ import annotations

__all__ = ["compute_concentration_analysis"]


def _round_to_nearest(value: float, nearest: float) -> float:
    """Round *value* to the nearest multiple of *nearest*."""
    return round(value / nearest) * nearest


def _concentration_loading(multiple: float) -> float:
    if multiple <= 5.0:
        return 0.0
    if multiple <= 10.0:
        return 5.0
    if multiple <= 25.0:
        return 10.0
    return 20.0


def _risk_level(flag: bool, multiple: float) -> str:
    if not flag:
        return "low"
    if multiple <= 10.0:
        return "elevated"
    return "high"


def compute_concentration_analysis(data: dict) -> dict:
    """
    Compute portfolio concentration risk KPIs.

    Args:
        data: Validated dict from validate_concentration_data().

    Returns:
        Flat dict of concentration metrics.
    """
    portfolio_total_sa     = data["portfolio_total_sa"]
    portfolio_policy_count = data["portfolio_policy_count"]
    policy_sum_assured     = data["policy_sum_assured"]

    # Core concentration metrics
    sa_concentration_pct = round(
        policy_sum_assured / portfolio_total_sa * 100, 4
    )
    avg_policy_sa = portfolio_total_sa / portfolio_policy_count
    sa_multiple_of_average = policy_sum_assured / avg_policy_sa

    # Flags and loadings
    concentration_flag = (
        sa_concentration_pct > 0.5 or sa_multiple_of_average > 5.0
    )
    concentration_loading_pct = _concentration_loading(sa_multiple_of_average)

    # Net retention recommendation
    raw_retention = max(avg_policy_sa * 5.0, portfolio_total_sa * 0.001)
    net_retention_recommendation = _round_to_nearest(raw_retention, 50_000.0)

    reinsurance_trigger = policy_sum_assured > net_retention_recommendation

    return {
        "policy_sum_assured":          policy_sum_assured,
        "portfolio_total_sa":          portfolio_total_sa,
        "portfolio_policy_count":      portfolio_policy_count,
        "sa_concentration_pct":        sa_concentration_pct,
        "avg_policy_sa":               round(avg_policy_sa, 4),
        "sa_multiple_of_average":      round(sa_multiple_of_average, 4),
        "concentration_flag":          concentration_flag,
        "concentration_loading_pct":   concentration_loading_pct,
        "net_retention_recommendation": net_retention_recommendation,
        "reinsurance_trigger":         reinsurance_trigger,
        "concentration_risk_level":    _risk_level(concentration_flag, sa_multiple_of_average),
    }
