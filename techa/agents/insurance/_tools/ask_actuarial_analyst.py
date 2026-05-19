"""
agents/insurance/_tools/ask_actuarial_analyst.py — Actuarial risk AI analyst.

Assesses mortality and morbidity risk from demographic and medical data.
Returns a structured ActuarialAnalysis with loading recommendations.
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
You are a qualified actuary (FIA/FSA) specialising in life and health insurance risk pricing.
You assess individual policyholder risk profiles to determine mortality and morbidity
loadings, and recommend a risk classification.

Relevant payload fields:
- applicant.age:             Age at entry. Mortality rises exponentially after 50.
                             Standard: 18–55. Substandard loading: 55–70. Postpone: 70+.
- applicant.gender:          Male mortality is ~20–30% higher than female at ages 40–60.
- applicant.smoker:          Current smokers carry 2–3× the mortality risk of non-smokers.
                             Loading: 100–150% for current smokers; 50% for ex-smokers < 5 yr.
- applicant.bmi / bmi_category:
                             normal (18.5–24.9) = standard.
                             overweight (25–29.9) = mild loading 10–25%.
                             obese (30–34.9) = moderate loading 25–75%.
                             severely_obese (35–39.9) = 75–150% or decline for large sums.
                             morbidly_obese (40+) = decline for life cover > £500k.
- applicant.systolic_bp / bp_category:
                             normal (<120) = standard. elevated (120–129) = mild loading 10%.
                             stage1 (130–139) = 25–50%. stage2 (140–159) = 50–100%.
                             hypertensive_crisis (160+) = postpone or decline.
- applicant.medical_history: Diagnosed conditions.
                             Hypertension: 25–100% depending on control.
                             Type 2 diabetes: 50–200%. Type 1 diabetes: 150–300%.
                             Heart disease / stroke / cancer: postpone or decline.
- applicant.family_history:  Genetic loading. ≥2 first-degree relatives with cardiovascular
                             disease before age 60 = +25% additional loading.
- financial_metrics.loss_ratio: Portfolio loss experience. > 0.85 = adverse underwriting result.

Risk classifications:
- standard:    No additional loading (mortality within 130% of standard table).
- substandard: Loading required (130–250% of standard table). Policy issued with extra premium.
- postpone:    Reassess in 12–24 months (recent diagnosis, pending investigation).
- decline:     Uninsurable at any commercially viable premium (> 250% standard table).

Combined loading rule: loadings from independent risk factors are additive, not multiplicative.
Cap total loading at 250% before switching to decline recommendation.

Output:
- mortality_percentile: 0–100 where 100 = worst insurable risk relative to standard table.
  Standard population = 50. Use this to calibrate conviction.
- expected_loss_ratio: Projected claims / premium given the recommended loading.
  Target 0.60–0.75 for a commercially viable life product.
- mortality_loading_pct: Total extra premium % recommended (0 = standard rates).
- risk_classification: standard / substandard / postpone / decline.
- key_risk_factors: Up to 5 factors driving the loading, in descending order of materiality.
- conviction: high (clear-cut case with strong data), medium (some uncertainty),
  low (incomplete data or borderline decision).
- verdict: One sentence for a senior underwriter. Quote the total loading and classification.\
"""


class ActuarialAnalysis(BaseModel):
    description:          str = Field(
        description=(
            "2–3 sentence summary of the actuarial risk picture. "
            "Cover: age/gender baseline, key loading drivers, and the net mortality percentile. "
            "Quote exact values from the payload."
        )
    )
    mortality_percentile: int = Field(
        description=(
            "Mortality risk percentile relative to the standard insured population (0–100). "
            "50 = average standard risk. 80 = 80th percentile (significantly above average). "
            "100 = maximum insurable risk."
        )
    )
    expected_loss_ratio:  float = Field(
        description=(
            "Projected loss ratio (claims / premium) at the recommended loading. "
            "Target range 0.60–0.75 for commercial viability."
        )
    )
    mortality_loading_pct: float = Field(
        description=(
            "Total additional premium percentage recommended to cover excess mortality/morbidity risk. "
            "0.0 = standard rates. 100.0 = double the standard premium."
        )
    )
    risk_classification:  Literal["standard", "substandard", "postpone", "decline"] = Field(
        description=(
            "standard: loading 0%. substandard: loading 1–250%. "
            "postpone: loading would exceed 250%, reassess in 12–24 months. "
            "decline: uninsurable."
        )
    )
    key_risk_factors:     list[str] = Field(
        description="Up to 5 risk factors ranked by materiality. Each as a short phrase."
    )
    conviction:           Literal["high", "medium", "low"] = Field(
        description=(
            "high: clear-cut case, complete data. medium: borderline or some data gaps. "
            "low: incomplete history, ambiguous presentation."
        )
    )
    verdict:              str = Field(
        description=(
            "One actionable sentence for a senior underwriter. "
            "Include total loading %, risk classification, and the primary driver."
        )
    )


def ask_actuarial_analyst(
    payload: dict,
    policy_id: str,
    question: str | None = None,
) -> ActuarialAnalysis:
    """
    Send the insurance risk payload to the model for actuarial analysis.

    Args:
        payload:   Dict from prepare_node — contains applicant, coverage,
                   claims_history, financial_metrics.
        policy_id: Application reference string (for context in the user message).
        question:  Optional follow-up question.

    Returns:
        ActuarialAnalysis Pydantic model.
    """
    actuarial_keys = ["applicant", "coverage", "financial_metrics"]
    actuarial_data = {k: payload[k] for k in actuarial_keys if k in payload}

    user_content = (
        f"Policy ID: {policy_id}  "
        f"Product: {payload.get('product_type', 'unknown')}  "
        f"Date: {payload.get('assessment_date', 'unknown')}\n\n"
        f"Actuarial data:\n{json.dumps(actuarial_data, indent=2)}"
    )
    if question:
        user_content += f"\n\nQuestion: {question}"

    log.info("Sending actuarial data for %s to %s", policy_id, MODEL)

    return invoke_structured(
        ActuarialAnalysis,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=1024,
    )
