"""
techa/actuarial/pricing/analysis.py — Reinsurance deal pricing KPIs.

Net cash flow per year
----------------------
NCF_t = ceded_premium_t − ceded_claims_t − expenses_t − commission_t + profit_commission_t

NPV (Net Present Value)
-----------------------
NPV = Σ NCF_t / (1 + r)^t   for t = 1 … N

Payback period
--------------
First year t where Σ NCF_1..t > 0.
None if cumulative cash flow never turns positive.

Loss / expense / commission ratios
-----------------------------------
loss_ratio       = Σ ceded_claims      / Σ ceded_premium
expense_ratio    = Σ expenses          / Σ ceded_premium
commission_ratio = Σ commission        / Σ ceded_premium
profit_margin    = 1 − loss_ratio − expense_ratio − commission_ratio

Break-even loss ratio
---------------------
break_even_lr = 1 − expense_ratio − commission_ratio
(maximum loss ratio consistent with a profitable treaty)

Stress scenarios (A/E shock)
-----------------------------
stress_ae25_loss_ratio: claims × 1.25
stress_ae50_loss_ratio: claims × 1.50

Pricing adequacy
----------------
profit_margin > 0.10  → adequate
0 < margin ≤ 0.10     → marginal
margin ≤ 0            → inadequate
"""

from __future__ import annotations

import math

__all__ = ["compute_pricing_analysis"]

nan = float("nan")


def _npv(net_flows: list[float], rate: float) -> float:
    if rate <= -1.0:
        return nan
    return sum(cf / (1.0 + rate) ** t for t, cf in enumerate(net_flows, start=1))


def _payback_period(cumulative: list[float]) -> int | None:
    for i, v in enumerate(cumulative, start=1):
        if v > 0:
            return i
    return None


def _pricing_adequacy(margin: float) -> str:
    if math.isnan(margin):
        return "unknown"
    if margin > 0.10:
        return "adequate"
    if margin > 0.0:
        return "marginal"
    return "inadequate"


def compute_pricing_analysis(data: dict) -> dict:
    """
    Compute reinsurance pricing KPIs from validated cash flow data.

    Args:
        data: Validated dict from validate_pricing_data().

    Returns:
        Flat dict of pricing metrics.
    """
    flows    = data["_flows"]
    rate     = data["_discount_rate"]

    total_premium    = sum(f["ceded_premium"]     for f in flows)
    total_claims     = sum(f["ceded_claims"]      for f in flows)
    total_expenses   = sum(f["expenses"]          for f in flows)
    total_commission = sum(f["commission"]        for f in flows)
    total_pc         = sum(f["profit_commission"] for f in flows)

    net_flows = [
        f["ceded_premium"] - f["ceded_claims"] - f["expenses"]
        - f["commission"] + f["profit_commission"]
        for f in flows
    ]

    cumulative = []
    running = 0.0
    for v in net_flows:
        running += v
        cumulative.append(round(running, 2))

    npv_val      = _npv(net_flows, rate)
    payback      = _payback_period(cumulative)
    total_profit = total_premium - total_claims - total_expenses - total_commission + total_pc

    def _safe_ratio(num, den):
        return num / den if den > 0 else nan

    loss_ratio       = _safe_ratio(total_claims,     total_premium)
    expense_ratio    = _safe_ratio(total_expenses,   total_premium)
    commission_ratio = _safe_ratio(total_commission, total_premium)
    profit_margin    = _safe_ratio(total_profit,     total_premium)

    break_even_lr = (1.0 - expense_ratio - commission_ratio
                    if not math.isnan(expense_ratio) and not math.isnan(commission_ratio)
                    else nan)

    # Stress scenarios
    stress_25_claims     = total_claims * 1.25
    stress_50_claims     = total_claims * 1.50
    stress_25_lr         = _safe_ratio(stress_25_claims, total_premium)
    stress_50_lr         = _safe_ratio(stress_50_claims, total_premium)
    stress_25_profit     = total_premium - stress_25_claims - total_expenses - total_commission + total_pc
    stress_50_profit     = total_premium - stress_50_claims - total_expenses - total_commission + total_pc
    stress_25_margin     = _safe_ratio(stress_25_profit, total_premium)
    stress_50_margin     = _safe_ratio(stress_50_profit, total_premium)

    return {
        "term_years":                  len(flows),
        "discount_rate":               rate,
        "total_ceded_premium":         round(total_premium,    2),
        "total_ceded_claims":          round(total_claims,     2),
        "total_expenses":              round(total_expenses,   2),
        "total_commission":            round(total_commission, 2),
        "total_profit":                round(total_profit,     2),
        "npv":                         round(npv_val, 2) if not math.isnan(npv_val) else nan,
        "payback_period_years":        payback,
        "loss_ratio":                  round(loss_ratio,       4) if not math.isnan(loss_ratio)       else nan,
        "expense_ratio":               round(expense_ratio,    4) if not math.isnan(expense_ratio)    else nan,
        "commission_ratio":            round(commission_ratio, 4) if not math.isnan(commission_ratio) else nan,
        "profit_margin":               round(profit_margin,    4) if not math.isnan(profit_margin)    else nan,
        "break_even_loss_ratio":       round(break_even_lr,   4) if not math.isnan(break_even_lr)    else nan,
        "stress_ae25_loss_ratio":      round(stress_25_lr,    4) if not math.isnan(stress_25_lr)     else nan,
        "stress_ae50_loss_ratio":      round(stress_50_lr,    4) if not math.isnan(stress_50_lr)     else nan,
        "stress_ae25_profit_margin":   round(stress_25_margin, 4) if not math.isnan(stress_25_margin) else nan,
        "stress_ae50_profit_margin":   round(stress_50_margin, 4) if not math.isnan(stress_50_margin) else nan,
        "pricing_adequacy":            _pricing_adequacy(profit_margin),
        "cumulative_cash_flows":       cumulative,
    }
