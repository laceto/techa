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
from techa.actuarial import (
    build_ae_snapshot,
    build_pricing_snapshot,
    build_inforce_snapshot,
    build_geo_snapshot,
    build_concentration_snapshot,
)

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
    # Actuarial tool 1: A/E monitoring (portfolio mortality experience)
    "ae_data": {
        "risk_type":    "mortality",
        "product_type": "term_life",
        "periods": [
            {"period": "2022-Q3", "actual_claims": 38, "expected_claims": 40.0, "exposed_lives": 9_500},
            {"period": "2022-Q4", "actual_claims": 41, "expected_claims": 40.5, "exposed_lives": 9_700},
            {"period": "2023-Q1", "actual_claims": 43, "expected_claims": 41.0, "exposed_lives": 9_900},
            {"period": "2023-Q2", "actual_claims": 46, "expected_claims": 41.5, "exposed_lives": 10_100},
            {"period": "2023-Q3", "actual_claims": 49, "expected_claims": 42.0, "exposed_lives": 10_300},
            {"period": "2023-Q4", "actual_claims": 53, "expected_claims": 42.5, "exposed_lives": 10_500},
        ],
    },
    # Actuarial tool 2: Reinsurance deal pricing (5-year quota share)
    "pricing_data": {
        "treaty_type":   "quota_share",
        "discount_rate": 0.05,
        "cash_flows": [
            {"ceded_premium": 480_000, "ceded_claims": 200_000, "expenses": 24_000, "commission": 96_000},
            {"ceded_premium": 500_000, "ceded_claims": 215_000, "expenses": 25_000, "commission": 100_000},
            {"ceded_premium": 520_000, "ceded_claims": 228_000, "expenses": 26_000, "commission": 104_000},
            {"ceded_premium": 540_000, "ceded_claims": 240_000, "expenses": 27_000, "commission": 108_000},
            {"ceded_premium": 560_000, "ceded_claims": 252_000, "expenses": 28_000, "commission": 112_000},
        ],
    },
    # Actuarial tool 3: In-force portfolio assessment (6 quarters)
    "inforce_data": {
        "periods_per_year": 4,
        "product_type":     "term_life",
        "periods": [
            {"period": "2022-Q3", "policies_in_force": 9_500,  "gross_premium_income": 2_375_000,
             "new_policies": 380, "lapses": 76, "deaths": 38,
             "best_estimate_liability": 42_000_000, "risk_margin": 2_100_000,
             "solvency_capital_requirement": 4_800_000, "own_funds": 8_200_000},
            {"period": "2022-Q4", "policies_in_force": 9_750,  "gross_premium_income": 2_437_500,
             "new_policies": 400, "lapses": 72, "deaths": 35,
             "best_estimate_liability": 43_200_000, "risk_margin": 2_160_000,
             "solvency_capital_requirement": 4_900_000, "own_funds": 8_600_000},
            {"period": "2023-Q1", "policies_in_force": 9_900,  "gross_premium_income": 2_475_000,
             "new_policies": 410, "lapses": 75, "deaths": 33,
             "best_estimate_liability": 44_500_000, "risk_margin": 2_225_000,
             "solvency_capital_requirement": 5_000_000, "own_funds": 8_900_000},
            {"period": "2023-Q2", "policies_in_force": 10_100, "gross_premium_income": 2_525_000,
             "new_policies": 430, "lapses": 70, "deaths": 30,
             "best_estimate_liability": 45_500_000, "risk_margin": 2_275_000,
             "solvency_capital_requirement": 5_100_000, "own_funds": 9_300_000},
            {"period": "2023-Q3", "policies_in_force": 10_300, "gross_premium_income": 2_575_000,
             "new_policies": 445, "lapses": 68, "deaths": 28,
             "best_estimate_liability": 46_800_000, "risk_margin": 2_340_000,
             "solvency_capital_requirement": 5_200_000, "own_funds": 9_700_000},
            {"period": "2023-Q4", "policies_in_force": 10_500, "gross_premium_income": 2_625_000,
             "new_policies": 460, "lapses": 65, "deaths": 26,
             "best_estimate_liability": 47_800_000, "risk_margin": 2_390_000,
             "solvency_capital_requirement": 5_300_000, "own_funds": 10_100_000},
        ],
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
        "kpi_snapshot":             None,
        "medical_snapshot":         None,
        "claims_snapshot":          None,
        "ae_snapshot":              None,
        "pricing_snapshot":         None,
        "inforce_snapshot":         None,
        "geo_snapshot":             None,
        "concentration_snapshot":   None,
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

    # ── Actuarial A/E monitoring snapshot ────────────────────────────────────
    ae_data = profile.get("ae_data")
    if ae_data:
        try:
            ae = build_ae_snapshot(ae_data, nan_to_none=True)
            payload["ae_snapshot"] = ae
            log.info(
                "[prepare] ae_snapshot built: risk=%s periods=%d ae_pct=%s alert=%s trend=%s",
                ae.get("risk_type"), ae.get("period_count"),
                ae.get("aggregate_ae_pct"), ae.get("ae_alert_level"),
                ae.get("ae_trend_direction"),
            )
        except Exception as exc:
            log.warning("[prepare] ae_snapshot failed: %s", exc)

    # ── Actuarial reinsurance pricing snapshot ────────────────────────────────
    pricing_data = profile.get("pricing_data")
    if pricing_data:
        try:
            pr = build_pricing_snapshot(pricing_data, nan_to_none=True)
            payload["pricing_snapshot"] = pr
            log.info(
                "[prepare] pricing_snapshot built: treaty=%s loss_ratio=%s adequacy=%s",
                pr.get("treaty_type"), pr.get("loss_ratio"), pr.get("pricing_adequacy"),
            )
        except Exception as exc:
            log.warning("[prepare] pricing_snapshot failed: %s", exc)

    # ── Actuarial in-force portfolio snapshot ─────────────────────────────────
    inforce_data = profile.get("inforce_data")
    if inforce_data:
        try:
            inf = build_inforce_snapshot(inforce_data, nan_to_none=True)
            payload["inforce_snapshot"] = inf
            log.info(
                "[prepare] inforce_snapshot built: pif=%s solvency=%s lapse_trend=%s",
                inf.get("pif_latest"), inf.get("solvency_status"),
                inf.get("lapse_trend_direction"),
            )
        except Exception as exc:
            log.warning("[prepare] inforce_snapshot failed: %s", exc)

    # ── Geospatial / epidemiological enrichment snapshot ─────────────────────
    geo_data = profile.get("geo_data")
    if geo_data:
        try:
            geo = build_geo_snapshot(geo_data, nan_to_none=True)
            payload["geo_snapshot"] = geo
            log.info(
                "[prepare] geo_snapshot built: area=%s imd=%s regional_ae=%s geo_risk=%s",
                geo.get("postcode_area"), geo.get("imd_decile"),
                geo.get("regional_ae_index"), geo.get("geo_risk_level"),
            )
        except Exception as exc:
            log.warning("[prepare] geo_snapshot failed: %s", exc)

    # ── Portfolio concentration snapshot ──────────────────────────────────────
    portfolio_context = profile.get("portfolio_context")
    if portfolio_context:
        try:
            con = build_concentration_snapshot(portfolio_context, nan_to_none=True)
            payload["concentration_snapshot"] = con
            log.info(
                "[prepare] concentration_snapshot built: sa_conc=%.4f%% flag=%s reins_trigger=%s",
                con.get("sa_concentration_pct", 0),
                con.get("concentration_flag"),
                con.get("reinsurance_trigger"),
            )
        except Exception as exc:
            log.warning("[prepare] concentration_snapshot failed: %s", exc)

    log.info(
        "[prepare] payload built: policy=%s kpi=%s med=%s claims=%s "
        "ae=%s pricing=%s inforce=%s geo=%s conc=%s",
        policy_id,
        "yes" if payload["kpi_snapshot"] else "no",
        "yes" if payload["medical_snapshot"] else "no",
        "yes" if payload["claims_snapshot"] else "no",
        "yes" if payload["ae_snapshot"] else "no",
        "yes" if payload["pricing_snapshot"] else "no",
        "yes" if payload["inforce_snapshot"] else "no",
        "yes" if payload["geo_snapshot"] else "no",
        "yes" if payload["concentration_snapshot"] else "no",
    )
    return payload
