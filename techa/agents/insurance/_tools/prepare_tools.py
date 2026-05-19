"""
agents/insurance/_tools/prepare_tools.py — Risk profile validation and enrichment.

Accepts a caller-supplied risk profile dict and enriches it with derived fields
(BMI category, age band, computed ratios) so all four specialist workers receive
a consistent, fully-populated payload. If no profile is provided, a built-in demo
profile is used so the graph can run end-to-end without real data.

When risk_profile contains a "financial_history" key (list of period records with
gwp / claims_incurred / expenses columns), build_payload calls build_kpi_snapshot()
and embeds the computed accountant KPI snapshot in payload["kpi_snapshot"]. This
gives ask_accountant access to derived KPIs (trends, CAGR, adequacy ratios) rather
than just the raw scalar financial_metrics.

Public API:
    build_payload(policy_id, risk_profile) -> dict
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from techa.insurance import build_kpi_snapshot
from techa.underwriting import build_medical_snapshot
from techa.claims import build_claims_snapshot

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
    "claim_form": {
        "claim_type":                    "critical_illness",
        "date_of_event":                 "2025-10-01",
        "date_of_submission":            "2025-10-20",
        "policy_inception_date":         "2020-03-01",
        "claim_amount_requested":        500_000,
        "sum_assured":                   500_000,
        "premium_annual":                3_200,
        "diagnosis":                     ["acute_myocardial_infarction", "coronary_artery_disease"],
        "icd_codes":                     ["I21.9", "I25.1"],
        "treating_physician":            "Dr. A. Patel, Cardiologist",
        "hospital_name":                 "St. Thomas Hospital, London",
        "admission_date":                "2025-10-01",
        "discharge_date":                "2025-10-08",
        "prognosis":                     "partial_recovery",
        "treatment_summary":             "Emergency PTCA; stent inserted LAD. Cardiac rehabilitation initiated.",
        "pre_existing_conditions_declared": ["hypertension"],
        "medical_history_consistent":    True,
        "nondisclosure_flag":            False,
        "documents_submitted":           [
            "specialist_report",
            "hospital_records",
            "medical_report",
            "discharge_summary",
        ],
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
        "kpi_snapshot":      None,
        "medical_snapshot":  None,
        "claims_snapshot":   None,
    }

    # ── Accountant KPI snapshot from financial history time series ────────────
    financial_history = profile.get("financial_history")
    if financial_history:
        try:
            hist_df = pd.DataFrame(financial_history)
            if "period" in hist_df.columns:
                hist_df["period"] = pd.to_datetime(hist_df["period"])
                hist_df = hist_df.set_index("period").sort_index()
            kpi = build_kpi_snapshot(hist_df, nan_to_none=True)
            payload["kpi_snapshot"] = kpi
            log.info(
                "[prepare] kpi_snapshot built from %d periods: %d KPI keys",
                len(hist_df),
                len(kpi),
            )
        except Exception as exc:
            log.warning("[prepare] kpi_snapshot failed (falling back to financial_metrics): %s", exc)

    # ── Medical underwriting snapshot from applicant questionnaire ───────────
    try:
        med = build_medical_snapshot(applicant, nan_to_none=True)
        payload["medical_snapshot"] = med
        log.info(
            "[prepare] medical_snapshot built: age=%s gender=%s risk_score=%.1f total_loading=%.1f",
            applicant.get("age"),
            applicant.get("gender"),
            med.get("risk_score", 0),
            med.get("total_medical_loading_pct", 0),
        )
    except Exception as exc:
        log.warning("[prepare] medical_snapshot failed (falling back to raw applicant): %s", exc)

    # ── Claims snapshot from claim form + medical documentation ─────────────
    claim_form = profile.get("claim_form")
    if claim_form:
        try:
            cls = build_claims_snapshot(claim_form, nan_to_none=True)
            payload["claims_snapshot"] = cls
            log.info(
                "[prepare] claims_snapshot built: type=%s risk=%s loading=%.1f fraud_flags=%s",
                cls.get("claim_type"),
                cls.get("claims_risk_level"),
                cls.get("claims_loading_pct", 0),
                cls.get("fraud_flags"),
            )
        except Exception as exc:
            log.warning("[prepare] claims_snapshot failed (falling back to claims_history): %s", exc)

    log.info(
        "[prepare] payload built: policy=%s product=%s age=%s bmi_cat=%s kpi=%s med=%s claims=%s",
        policy_id,
        payload["product_type"],
        applicant.get("age"),
        applicant.get("bmi_category"),
        "yes" if payload["kpi_snapshot"] else "no",
        "yes" if payload["medical_snapshot"] else "no",
        "yes" if payload["claims_snapshot"] else "no",
    )
    return payload
