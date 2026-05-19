"""
techa/claims/timeline.py — Policy seasoning and claim submission timeline.

Loading schedule
----------------
policy_age < 30 days:   +50%  (very early — high suspicion)
policy_age 30–179 days: +25%  (early claim period)
policy_age 180–364 days: +10% (first year — watchful)
policy_age ≥ 365 days:    0%  (seasoned policy)

Submission delay loading
------------------------
> 180 days from event to submission: +10% (unusual latency).

Submission delay risk bands
---------------------------
0–29 days:   low
30–89 days:  normal
90–179 days: elevated
≥ 180 days:  high
"""

from __future__ import annotations

__all__ = ["compute_timeline"]

nan = float("nan")

_VERY_EARLY_DAYS  = 30
_EARLY_CLAIM_DAYS = 180


def _submission_delay_risk(days: int | None) -> str:
    if days is None:
        return "unknown"
    if days < 0:
        return "invalid"
    if days < 30:
        return "low"
    if days < 90:
        return "normal"
    if days < 180:
        return "elevated"
    return "high"


def _timeline_loading(policy_age_days: int | None, submission_delay_days: int | None) -> float:
    load = 0.0
    if policy_age_days is not None:
        if policy_age_days < _VERY_EARLY_DAYS:
            load += 50.0
        elif policy_age_days < _EARLY_CLAIM_DAYS:
            load += 25.0
        elif policy_age_days < 365:
            load += 10.0
    if submission_delay_days is not None and submission_delay_days > 180:
        load += 10.0
    return load


def compute_timeline(data: dict) -> dict:
    """
    Assess policy seasoning and submission timeline risk.

    Args:
        data: Validated claim form dict from validate_claim_form().

    Returns:
        Flat dict: policy_age_days, policy_age_months, early_claim_flag,
        very_early_claim_flag, submission_delay_days, submission_delay_risk,
        inpatient_duration_days, timeline_loading_pct.
    """
    policy_age    = data.get("_policy_age_days")
    sub_delay     = data.get("_submission_delay_days")
    inpatient     = data.get("_inpatient_duration_days")

    early_claim      = policy_age is not None and policy_age < _EARLY_CLAIM_DAYS
    very_early_claim = policy_age is not None and policy_age < _VERY_EARLY_DAYS

    age_months = round(policy_age / 30.44, 1) if policy_age is not None else nan

    return {
        "policy_age_days":         policy_age    if policy_age is not None else nan,
        "policy_age_months":       age_months,
        "early_claim_flag":        early_claim,
        "very_early_claim_flag":   very_early_claim,
        "submission_delay_days":   sub_delay     if sub_delay  is not None else nan,
        "submission_delay_risk":   _submission_delay_risk(sub_delay),
        "inpatient_duration_days": inpatient     if inpatient  is not None else nan,
        "timeline_loading_pct":    _timeline_loading(policy_age, sub_delay),
    }
