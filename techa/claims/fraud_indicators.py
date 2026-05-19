"""
techa/claims/fraud_indicators.py — Rule-based fraud risk indicator scoring.

Each indicator is a named flag. The count of active flags determines the
fraud risk level and associated loading.

Indicators
----------
very_early_claim:
    Policy age < 30 days at time of event.  High suspicion of pre-inception knowledge.

early_claim:
    Policy age 30–179 days.  Known elevated risk period for moral hazard.

nondisclosure:
    Caller-supplied nondisclosure_flag = True (investigator or system flagged
    that a condition material to the risk was not declared at application).

round_amount:
    Claim amount is an exact multiple of £10,000 with no supporting schedule of
    loss — typical of inflated or fabricated claims.

medical_history_inconsistency:
    medical_history_consistent = False, meaning the treating physician's records
    reference conditions not mentioned in the original application.

over_indemnification:
    Claim amount > sum assured — policy cannot legally pay more than the sum
    assured; flags a data error or fraudulent inflation.

incomplete_documentation:
    documentation_status = "insufficient" — critical docs missing despite
    time to submit; consistent with reluctance to provide verifiable evidence.

Fraud risk levels
-----------------
0 flags:  low       → 0% loading
1 flag:   low       → 5% loading  (mild; single indicator is common)
2 flags:  medium    → 15% loading
3 flags:  high      → 30% loading
≥4 flags: very_high → 50% loading; refer to Special Investigations Unit
"""

from __future__ import annotations

__all__ = ["compute_fraud_indicators"]

_FLAG_DESCRIPTIONS: dict[str, str] = {
    "very_early_claim":             "Claim event within 30 days of policy inception",
    "early_claim":                  "Claim event within 6 months of policy inception",
    "nondisclosure":                "Non-disclosure of material pre-existing condition flagged",
    "round_amount":                 "Claim amount is a suspiciously round number (multiple of £10,000)",
    "medical_history_inconsistency":"Medical records reference conditions not declared at application",
    "over_indemnification":         "Claim amount exceeds the policy sum assured",
    "incomplete_documentation":     "Required documents missing (documentation_status = insufficient)",
}


def _is_round_amount(amount) -> bool:
    try:
        val = float(amount)
        return val > 0 and (val % 10_000 == 0)
    except (TypeError, ValueError):
        return False


def _fraud_level(count: int) -> str:
    if count == 0:
        return "low"
    if count == 1:
        return "low"
    if count == 2:
        return "medium"
    if count == 3:
        return "high"
    return "very_high"


def _fraud_loading(count: int) -> float:
    if count == 0:
        return 0.0
    if count == 1:
        return 5.0
    if count == 2:
        return 15.0
    if count == 3:
        return 30.0
    return 50.0


def compute_fraud_indicators(data: dict) -> dict:
    """
    Score rule-based fraud risk indicators.

    Args:
        data: Validated claim form dict from validate_claim_form().
              Expects pre-computed keys from timeline and documentation modules
              to already be merged in, or reads the private adapter keys directly.

    Returns:
        Flat dict: fraud_flags (list of active flag names), fraud_flag_descriptions
        (human-readable), fraud_indicator_count, fraud_risk_level, fraud_loading_pct.
    """
    policy_age   = data.get("_policy_age_days")
    nondisclosure = bool(data.get("nondisclosure_flag", False))
    med_consistent = data.get("medical_history_consistent")
    claim_amount  = data.get("claim_amount_requested")
    sum_assured   = data.get("sum_assured")
    claim_to_sa   = data.get("_claim_to_sa_ratio")
    doc_status    = data.get("documentation_status", "")

    flags: list[str] = []

    if policy_age is not None and policy_age < 30:
        flags.append("very_early_claim")
    elif policy_age is not None and policy_age < 180:
        flags.append("early_claim")

    if nondisclosure:
        flags.append("nondisclosure")

    if _is_round_amount(claim_amount):
        flags.append("round_amount")

    if med_consistent is not None and not med_consistent:
        flags.append("medical_history_inconsistency")

    if claim_to_sa is not None and claim_to_sa > 1.0:
        flags.append("over_indemnification")

    if doc_status == "insufficient":
        flags.append("incomplete_documentation")

    descriptions = [_FLAG_DESCRIPTIONS[f] for f in flags]

    return {
        "fraud_flags":             flags,
        "fraud_flag_descriptions": descriptions,
        "fraud_indicator_count":   len(flags),
        "fraud_risk_level":        _fraud_level(len(flags)),
        "fraud_loading_pct":       _fraud_loading(len(flags)),
    }
