"""
agents/insurance/_tools/ask_claims_assessor.py — Claims assessment AI analyst.

Reviews claims history: frequency, severity, loss development, and suspicious
pattern indicators. Returns a structured ClaimsAssessment.
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
You are a senior insurance claims assessor and fraud analytics specialist with 12+ years of experience.
You evaluate current claim submissions and historical claims data to assess validity, risk, and fraud exposure.

When a "claims_snapshot" is provided, it is your primary data source — a pre-computed KPI
snapshot derived from the claim form and medical documentation supplied by the insured:

  Timeline:
    policy_age_days, policy_age_months
    early_claim_flag (< 6 months),  very_early_claim_flag (< 30 days)
    submission_delay_days, submission_delay_risk (low/normal/elevated/high)
    inpatient_duration_days, timeline_loading_pct

  Severity:
    claim_amount_requested (£), sum_assured (£)
    claim_to_sa_ratio,  claim_to_premium_ratio
    severity_category (low/moderate/high/catastrophic), severity_loading_pct

  Medical coherence:
    diagnosis_categories (list),  primary_diagnosis_category
    icd_chapters (list),  claim_type_match (bool)
    prognosis_risk (low/moderate/high/very_high), coherence_loading_pct

  Documentation:
    documents_submitted_count, required_documents (list), missing_documents (list)
    documentation_completeness_pct, documentation_status (complete/partial/insufficient)
    documentation_loading_pct

  Fraud indicators:
    fraud_flags (list of active flag names), fraud_flag_descriptions (list)
    fraud_indicator_count, fraud_risk_level (low/medium/high/very_high)
    fraud_loading_pct

  Aggregate:
    claim_type, claims_loading_pct (capped at 100%), claims_risk_level

Fraud flag names and their meaning:
  very_early_claim          — event within 30 days of inception; pre-inception knowledge suspected.
  early_claim               — event within 6 months; elevated moral hazard period.
  nondisclosure             — investigator flagged undeclared material condition.
  round_amount              — claim is an exact £10,000 multiple; possible inflation.
  medical_history_inconsistency — treating physician records reference undisclosed conditions.
  over_indemnification      — claim exceeds sum assured; data error or fraud.
  incomplete_documentation  — critical documents missing despite adequate time to obtain.

When claims_snapshot is absent, fall back to the raw "claims_history" and "financial_metrics":
- claims_history.total_claims_count: 0–1 = low; 2–3 = normal; 4–5 = elevated; 6+ = high.
- claims_history.largest_claim_x_premium: < 3× = low; 3–10× = normal; > 10× = elevated; > 50× = investigate.
- financial_metrics.loss_ratio: < 0.60 = favourable; 0.60–0.85 = acceptable; > 0.85 = adverse.

Claims risk loading guideline:
- Low risk (fraud_count=0, claims_loading_pct < 10%):   0% additional loading.
- Medium risk (fraud_count=1–2, loading 10–25%):        10–25% loading.
- High risk (fraud_count≥3 or loading 25–50%):          25–50% loading; refer to SIU.
- Very high (loading > 50%):                            refer to Special Investigations Unit; withhold payment.

Output:
- claims_risk_level: low / medium / high / very_high. Use claims_snapshot value if present.
- frequency_risk: low / normal / elevated / high. Assess from claim history or snapshot.
- severity_risk: low / normal / elevated / high. Use severity_category from snapshot if present.
- loss_ratio_assessment: favourable / acceptable / adverse.
- suspicious_patterns: True if any fraud flag is active or fraud_indicator_count ≥ 1.
- suspicious_pattern_notes: describe active fraud flags; empty string if none.
- claims_loading_pct: use claims_loading_pct from snapshot when available; otherwise derive.
- conviction: high (complete multi-year + claims_snapshot) / medium / low.\
"""


class ClaimsAssessment(BaseModel):
    description:              str = Field(
        description=(
            "2–3 sentence summary of the claims history. "
            "Cover: total claims count, largest claim severity, recency, and loss ratio. "
            "Quote exact figures."
        )
    )
    claims_risk_level:        Literal["low", "medium", "high", "very_high"] = Field(
        description=(
            "low: 0–1 claims, loss ratio < 0.60. medium: 2–3 claims, loss ratio 0.60–0.85. "
            "high: 4–5 claims or loss ratio 0.85–1.00. very_high: 6+ claims or loss ratio > 1.00."
        )
    )
    frequency_risk:           Literal["low", "normal", "elevated", "high"] = Field(
        description="low: 0–1 claims. normal: 2–3. elevated: 4–5. high: 6+."
    )
    severity_risk:            Literal["low", "normal", "elevated", "high"] = Field(
        description=(
            "Based on largest_claim_x_premium. "
            "low: < 3×. normal: 3–10×. elevated: 10–50×. high: > 50×."
        )
    )
    total_claims_count:       int   = Field(description="Total claims count from payload.")
    total_claims_paid:        float = Field(description="Total claims paid (£) from payload.")
    largest_single_claim:     float = Field(description="Largest individual claim (£) from payload.")
    loss_ratio_assessment:    Literal["favourable", "acceptable", "adverse"] = Field(
        description="favourable: < 0.60. acceptable: 0.60–0.85. adverse: > 0.85."
    )
    suspicious_patterns:      bool  = Field(
        description="True if any fraud or unusual pattern indicator is triggered."
    )
    suspicious_pattern_notes: str   = Field(
        description=(
            "Description of flagged patterns. Empty string if suspicious_patterns is False."
        )
    )
    claims_loading_pct:       float = Field(
        description=(
            "Additional premium % recommended based on claims experience. "
            "0.0 for low risk. Up to 50.0 for very_high risk."
        )
    )
    conviction:               Literal["high", "medium", "low"] = Field(
        description=(
            "high: complete multi-year claims data. medium: limited history (< 3 years). "
            "low: first policy, no claims history available."
        )
    )
    verdict:                  str   = Field(
        description=(
            "One actionable sentence for a senior underwriter. "
            "Include claims risk level, loading %, and any fraud flag."
        )
    )


def ask_claims_assessor(
    payload: dict,
    policy_id: str,
    question: str | None = None,
) -> ClaimsAssessment:
    """
    Send claims history data to the model for claims risk assessment.

    Args:
        payload:   Dict from prepare_node — contains claims_history, financial_metrics,
                   and coverage sections.
        policy_id: Application reference string.
        question:  Optional follow-up question.

    Returns:
        ClaimsAssessment Pydantic model.
    """
    claims_snapshot = payload.get("claims_snapshot")
    if claims_snapshot:
        claims_data = {
            "claims_snapshot": claims_snapshot,
            "claims_history":  payload.get("claims_history"),
            "coverage":        payload.get("coverage"),
        }
        data_label = "Claims snapshot (pre-computed KPIs from claim form + medical docs)"
    else:
        claims_data = {k: payload[k] for k in ("claims_history", "financial_metrics", "coverage")
                       if k in payload}
        data_label = "Claims data (historical summary)"

    user_content = (
        f"Policy ID: {policy_id}  "
        f"Product: {payload.get('product_type', 'unknown')}  "
        f"Date: {payload.get('assessment_date', 'unknown')}\n\n"
        f"{data_label}:\n{json.dumps(claims_data, indent=2)}"
    )
    if question:
        user_content += f"\n\nQuestion: {question}"

    log.info("Sending claims data for %s to %s", policy_id, MODEL)

    return invoke_structured(
        ClaimsAssessment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=1024,
    )
