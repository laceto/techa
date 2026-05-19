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

Relevant payload fields:
- applicant.bmi / bmi_category:
    normal (18.5–24.9)    = standard rates.
    overweight (25–29.9)  = mild loading 10–25%.
    obese (30–34.9)       = moderate loading 25–75%.
    severely_obese (35–39.9) = loading 75–150%; decline for sum_assured > £750k.
    morbidly_obese (40+)  = decline for life cover. Refer for income protection only.
    underweight (<18.5)   = loading 15–25% (possible underlying illness).

- applicant.systolic_bp / bp_category:
    normal (<120)              = standard.
    elevated (120–129)         = mild loading 10%.
    stage1_hypertension (130–139) = loading 25–50%.
    stage2_hypertension (140–159) = loading 50–100%; require GP report.
    hypertensive_crisis (160+)    = postpone 3 months; require evidence of treatment.

- applicant.smoker:
    True (current smoker)  = loading 100–150%; assess for COPD and cardiovascular risk.
    False + <5 yr ex-smoker = loading 50% if medical_history indicates prior smoking.
    False + non-smoker      = standard rates.

- applicant.occupation_class:
    1 (office / professional)  = standard.
    2 (light manual / clerical) = loading 10–25%.
    3 (manual / outdoor labour) = loading 25–75%; consider accident benefit exclusions.
    4 (hazardous: mining, diving, roofing) = loading 75–200% or decline; special terms required.

- applicant.medical_history: List of diagnosed conditions. Apply loadings cumulatively.
    Hypertension (controlled)  = 25–75%. Hypertension (uncontrolled) = 75–150%.
    Type 2 diabetes (diet/oral) = 75–150%. Type 1 diabetes = 150–300%.
    Coronary artery disease     = postpone or decline.
    Stroke / TIA                = postpone 12–24 months; loading 100–200% on return.
    Cancer (in remission > 5yr) = 50–100% depending on type.
    Cancer (active / < 5yr remission) = postpone or decline.
    Anxiety / depression (mild) = loading 25–50%.
    Mental health (severe)      = postpone; refer to Chief Underwriter.

- applicant.family_history:
    Cardiovascular disease (≥2 first-degree relatives before age 60) = +25% loading.
    Cancer (hereditary type in first-degree relative) = +25–50% loading.

Exclusion clause guideline:
    Apply an exclusion clause when a condition cannot be commercially rated but the
    applicant is otherwise insurable. E.g., "excluding claims arising from pre-existing
    lumbar disc disease". Document each exclusion explicitly.

Combined loading rule:
    Sum loadings from independent risk factors. Cap at 250% before switching to postpone/decline.
    Exception: if any single condition warrants postpone/decline, that classification takes precedence.

Output:
- underwriting_decision: standard / rated / postpone / decline.
- medical_loading_pct: total additional premium % from medical factors (0 = standard).
- smoker_loading_pct: specific loading attributable to smoking status.
- bmi_loading_pct: specific loading attributable to BMI.
- occupation_loading_pct: specific loading attributable to occupation class.
- exclusions: list of exclusion clauses to apply (empty list if none).
- additional_requirements: list of required evidence before policy can be issued
  (e.g., GP report, blood test, ECG).
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
    medical_keys = ["applicant", "coverage"]
    medical_data = {k: payload[k] for k in medical_keys if k in payload}

    user_content = (
        f"Policy ID: {policy_id}  "
        f"Product: {payload.get('product_type', 'unknown')}  "
        f"Date: {payload.get('assessment_date', 'unknown')}\n\n"
        f"Medical underwriting data:\n{json.dumps(medical_data, indent=2)}"
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
