"""
agents/insurance/_tools/prepare_tools.py — Risk profile validation and enrichment.

Accepts a caller-supplied risk profile dict and enriches it with derived fields
(BMI category, age band, computed ratios) so all four specialist workers receive
a consistent, fully-populated payload. If no profile is provided, a built-in demo
profile is used so the graph can run end-to-end without real data.

Public API:
    build_payload(policy_id, risk_profile) -> dict
"""

from __future__ import annotations

import logging
from datetime import date

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demo profile — used when the caller passes risk_profile=None
# ---------------------------------------------------------------------------

_DEMO_PROFILE: dict = {
    "product_type":    "term_life",
    "assessment_date": str(date.today()),
    "applicant": {
        "age":                    45,
        "gender":                 "male",
        "smoker":                 False,
        "occupation":             "office_worker",
        "occupation_class":       1,       # 1=low, 2=medium, 3=high, 4=very_high
        "bmi":                    27.5,
        "systolic_bp":            130,
        "medical_history":        ["hypertension"],
        "family_history":         ["cardiovascular_disease"],
        "alcohol_units_per_week": 10,
    },
    "coverage": {
        "sum_assured":    500_000,
        "premium_annual": 3_200,
        "term_years":     20,
        "coverage_type":  "level_term",
    },
    "claims_history": {
        "total_claims_count":        2,
        "total_claims_paid":         4_500,
        "largest_single_claim":      3_200,
        "years_since_last_claim":    3,
        "claim_types":               ["medical_expenses"],
    },
    "financial_metrics": {
        "gross_premium_income":   3_200,
        "net_claims_incurred":    1_800,
        "loss_ratio":             0.5625,
        "expense_ratio":          0.28,
        "combined_ratio":         0.8425,
        "reserve_held":           15_000,
        "reserve_required":       14_200,
        "reserve_adequacy_pct":   105.6,
        "premium_growth_yoy_pct": 8.5,
    },
}


# ---------------------------------------------------------------------------
# Derived-field helpers
# ---------------------------------------------------------------------------

def _bmi_category(bmi: float | None) -> str:
    if bmi is None:
        return "unknown"
    if bmi < 18.5:
        return "underweight"
    if bmi < 25.0:
        return "normal"
    if bmi < 30.0:
        return "overweight"
    if bmi < 35.0:
        return "obese"
    if bmi < 40.0:
        return "severely_obese"
    return "morbidly_obese"


def _age_band(age: int | None) -> str:
    if age is None:
        return "unknown"
    if age < 30:
        return "18-29"
    if age < 40:
        return "30-39"
    if age < 50:
        return "40-49"
    if age < 60:
        return "50-59"
    if age < 70:
        return "60-69"
    return "70+"


def _bp_category(systolic: int | None) -> str:
    if systolic is None:
        return "unknown"
    if systolic < 120:
        return "normal"
    if systolic < 130:
        return "elevated"
    if systolic < 140:
        return "stage1_hypertension"
    if systolic < 160:
        return "stage2_hypertension"
    return "hypertensive_crisis"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_payload(policy_id: str, risk_profile: dict | None) -> dict:
    """
    Validate and enrich the risk profile into the canonical payload dict.

    If risk_profile is None, the built-in demo profile is used so the graph
    can run without real data. Caller-supplied profiles are merged over the
    demo defaults so partial profiles work correctly.

    Args:
        policy_id:    Unique application reference.
        risk_profile: Caller-supplied dict (full or partial). None → demo profile.

    Returns:
        Fully-populated payload dict ready for all four specialist workers.
    """
    import copy

    profile = copy.deepcopy(_DEMO_PROFILE)
    if risk_profile:
        for section, value in risk_profile.items():
            if isinstance(value, dict) and isinstance(profile.get(section), dict):
                profile[section].update(value)
            else:
                profile[section] = value

    applicant = profile.get("applicant", {})
    coverage  = profile.get("coverage",  {})
    claims    = profile.get("claims_history", {})
    fin       = profile.get("financial_metrics", {})

    # ── Derived enrichments ──────────────────────────────────────────────────
    applicant["bmi_category"] = _bmi_category(applicant.get("bmi"))
    applicant["age_band"]     = _age_band(applicant.get("age"))
    applicant["bp_category"]  = _bp_category(applicant.get("systolic_bp"))

    # Claims severity relative to annual premium
    premium = coverage.get("premium_annual") or 1
    largest = claims.get("largest_single_claim", 0)
    claims["largest_claim_x_premium"] = round(largest / premium, 2)

    # Loss ratio from claims if not already supplied
    if "loss_ratio" not in fin and "gross_premium_income" in fin and fin["gross_premium_income"]:
        incurred = fin.get("net_claims_incurred", 0)
        fin["loss_ratio"] = round(incurred / fin["gross_premium_income"], 4)

    payload = {
        "policy_id":         policy_id,
        "product_type":      profile.get("product_type", "unknown"),
        "assessment_date":   profile.get("assessment_date", str(date.today())),
        "applicant":         applicant,
        "coverage":          coverage,
        "claims_history":    claims,
        "financial_metrics": fin,
    }

    log.info(
        "[prepare] payload built: policy=%s product=%s age=%s bmi_cat=%s",
        policy_id,
        payload["product_type"],
        applicant.get("age"),
        applicant.get("bmi_category"),
    )
    return payload
