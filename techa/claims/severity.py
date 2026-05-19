"""
techa/claims/severity.py — Financial severity assessment of the claim.

Severity categories (by claim-to-sum-assured ratio)
----------------------------------------------------
< 10%:          low
10–49%:         moderate
50–100%:        high
> 100%:         catastrophic  (claim exceeds sum assured — over-indemnification risk)

Severity loading
----------------
claim_to_premium > 100×:  +25%
claim_to_premium 50–100×: +15%
claim_to_premium 20–50×:  +5%
claim > sum_assured:      +15% additional (over-indemnification flag)
"""

from __future__ import annotations

__all__ = ["compute_severity"]

nan = float("nan")


def _severity_category(ratio: float | None) -> str:
    if ratio is None:
        return "unknown"
    if ratio < 0.10:
        return "low"
    if ratio < 0.50:
        return "moderate"
    if ratio <= 1.00:
        return "high"
    return "catastrophic"


def _severity_loading(claim_to_sa: float | None, claim_to_premium: float | None) -> float:
    load = 0.0
    if claim_to_premium is not None:
        if claim_to_premium > 100:
            load += 25.0
        elif claim_to_premium > 50:
            load += 15.0
        elif claim_to_premium > 20:
            load += 5.0
    if claim_to_sa is not None and claim_to_sa > 1.0:
        load += 15.0
    return load


def compute_severity(data: dict) -> dict:
    """
    Assess financial severity of the claim relative to policy limits.

    Args:
        data: Validated claim form dict from validate_claim_form().

    Returns:
        Flat dict: claim_amount_requested, sum_assured, claim_to_sa_ratio,
        claim_to_premium_ratio, severity_category, severity_loading_pct.
    """
    claim_amount  = data.get("claim_amount_requested")
    sum_assured   = data.get("sum_assured")
    claim_to_sa   = data.get("_claim_to_sa_ratio")
    claim_to_prem = data.get("_claim_to_premium_ratio")

    return {
        "claim_amount_requested": float(claim_amount) if claim_amount is not None else nan,
        "sum_assured":            float(sum_assured)  if sum_assured  is not None else nan,
        "claim_to_sa_ratio":      claim_to_sa         if claim_to_sa  is not None else nan,
        "claim_to_premium_ratio": claim_to_prem       if claim_to_prem is not None else nan,
        "severity_category":      _severity_category(claim_to_sa),
        "severity_loading_pct":   _severity_loading(claim_to_sa, claim_to_prem),
    }
