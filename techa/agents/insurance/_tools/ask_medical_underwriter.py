"""
agents/insurance/_tools/ask_medical_underwriter.py — Medical underwriting AI analyst.

Evaluates individual risk factors: BMI, blood pressure, smoking status, medical
history, occupation class, and family history. Returns a structured MedicalUnderwritingAnalysis.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Literal

from pydantic import BaseModel, Field

from techa.agents._llm import invoke_structured, MODEL

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a senior medical underwriter with 15+ years of experience in life and health insurance.
You assess individual applicant medical risk to determine underwriting terms and exclusion clauses.

When a "medical_snapshot" is provided in the payload, use it as the primary data source —
it contains pre-computed, clinically-derived KPIs from the applicant questionnaire:

  Biometrics:
    bmi, bmi_category, bmi_loading_pct
  Cardiovascular:
    bp_systolic, bp_diastolic, pulse_pressure, bp_category, bp_loading_pct,
    cholesterol_ratio, cholesterol_risk, cholesterol_loading_pct, cv_risk_score
  Metabolic:
    diabetes_status, hba1c, hba1c_category, fasting_glucose, glucose_category,
    metabolic_loading_pct
  Lifestyle:
    smoking_status, pack_years, cigarettes_per_day, years_quit,
    smoking_loading_pct, alcohol_units_per_week, alcohol_risk, alcohol_loading_pct
  Medical conditions:
    condition_count, conditions_loading_pct, critical_condition_flag
  Family history:
    family_risk_factor_count, family_history_loading_pct, hereditary_cancer_risk
  Aggregate:
    occupation_class, occupation_loading_pct,
    total_medical_loading_pct (sum of all components, capped at 250%),
    risk_score (0–100, = total_loading × 0.4)

When medical_snapshot is absent, fall back to the raw "applicant" dict using these guidelines:
- bmi_category: normal=standard, overweight=10–25%, obese=25–75%, severely_obese=75–150%,
  morbidly_obese=decline for life cover, underweight=15–25%.
- bp_category: normal=standard, elevated=10%, stage1_hypertension=25–50%,
  stage2_hypertension=50–100% + GP report, hypertensive_crisis=postpone.
- smoker=True: 100–150% loading; non-smoker=standard.
- occupation_class: 1=standard, 2=10–25%, 3=25–75%, 4=75–200% or decline.
- medical_history: hypertension controlled=25–75%; type2 diabetes=75–150%; type1=150–300%;
  coronary artery disease/stroke/active cancer=postpone or decline.
- family_history: cardiovascular ≥2 relatives before 60=+25%; hereditary cancer=+25–50%.

Exclusion clause guideline:
    Apply an exclusion clause when a condition cannot be commercially rated but the
    applicant is otherwise insurable. Document each exclusion explicitly.

Combined loading rule:
    Sum loadings from independent risk factors. Cap at 250% before switching to postpone/decline.
    If any single condition warrants postpone/decline, that classification takes precedence.
    critical_condition_flag = True in the snapshot always triggers postpone or decline.

Output:
- underwriting_decision: standard / rated / postpone / decline.
- medical_loading_pct: total additional premium % from all medical factors combined.
  When medical_snapshot is present use total_medical_loading_pct directly.
- smoker_loading_pct: loading from smoking (snapshot: smoking_loading_pct).
- bmi_loading_pct: loading from BMI (snapshot: bmi_loading_pct).
- occupation_loading_pct: loading from occupation (snapshot: occupation_loading_pct).
- exclusions: list of exclusion clauses (empty if none).
- additional_requirements: evidence required before issuance (GP report, ECG, blood test, etc.).
- conviction: high / medium / low.\
"""


class MedicalUnderwritingAnalysis(BaseModel):
    description:             str = Field(
        description=(
            "2–3 sentence summary of the medical underwriting picture. "
            "Cover: BMI category, BP category, smoking status, key medical conditions, "
            "and the overall medical loading. Quote exact values."
        )
    )
    underwriting_decision:   Literal["standard", "rated", "postpone", "decline"] = Field(
        description=(
            "standard: no medical loading. rated: loading applied. "
            "postpone: reassess in 12–24 months. decline: uninsurable."
        )
    )
    medical_loading_pct:     float = Field(
        description="Total additional premium % from all medical risk factors combined."
    )
    smoker_loading_pct:      float = Field(
        description="Loading % attributable to smoking status (0.0 if non-smoker)."
    )
    bmi_loading_pct:         float = Field(
        description="Loading % attributable to BMI category (0.0 if normal BMI)."
    )
    occupation_loading_pct:  float = Field(
        description="Loading % attributable to occupation class (0.0 if class 1)."
    )
    bmi_assessment:          Literal["underweight", "normal", "overweight", "obese",
                                     "severely_obese", "morbidly_obese"] = Field(
        description="BMI category derived from bmi_category field."
    )
    bp_assessment:           Literal["normal", "elevated", "stage1_hypertension",
                                     "stage2_hypertension", "hypertensive_crisis"] = Field(
        description="Blood pressure category from bp_category field."
    )
    occupation_risk:         Literal["low", "medium", "high", "very_high"] = Field(
        description="low: class 1. medium: class 2. high: class 3. very_high: class 4."
    )
    exclusions:              list[str] = Field(
        description=(
            "List of exclusion clauses to apply to the policy. "
            "Empty list if no exclusions required."
        )
    )
    additional_requirements: list[str] = Field(
        description=(
            "Evidence required before policy issuance, e.g. 'GP report', "
            "'fasting blood glucose test', 'ECG'. Empty list if none required."
        )
    )
    conviction:              Literal["high", "medium", "low"] = Field(
        description=(
            "high: complete medical data, clear-cut case. "
            "medium: borderline findings or missing some history. "
            "low: significant data gaps or ambiguous presentation."
        )
    )
    verdict:                 str = Field(
        description=(
            "One actionable sentence for a senior underwriter. "
            "Include total medical loading %, decision, and primary medical driver."
        )
    )


def ask_medical_underwriter(
    payload: dict,
    policy_id: str,
    question: str | None = None,
) -> MedicalUnderwritingAnalysis:
    """
    Send the applicant medical data to the model for underwriting assessment.

    Args:
        payload:   Dict from prepare_node — contains applicant and coverage sections.
        policy_id: Application reference string.
        question:  Optional follow-up question.

    Returns:
        MedicalUnderwritingAnalysis Pydantic model.
    """
    medical_snapshot = payload.get("medical_snapshot")
    if medical_snapshot:
        medical_data = {
            "medical_snapshot": medical_snapshot,
            "coverage": payload.get("coverage"),
        }
        data_label = "Medical underwriting snapshot (pre-computed KPIs)"
    else:
        medical_data = {k: payload[k] for k in ("applicant", "coverage") if k in payload}
        data_label = "Medical underwriting data (raw applicant fields)"

    user_content = (
        f"Policy ID: {policy_id}  "
        f"Product: {payload.get('product_type', 'unknown')}  "
        f"Date: {payload.get('assessment_date', 'unknown')}\n\n"
        f"{data_label}:\n{json.dumps(medical_data, indent=2)}"
    )
    if question:
        user_content += f"\n\nQuestion: {question}"

    log.info("Sending medical data for %s to %s", policy_id, MODEL)

    return invoke_structured(
        MedicalUnderwritingAnalysis,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=1024,
    )
