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
You evaluate claims history data to assess future claims risk and identify unusual patterns.

Relevant payload fields:
- claims_history.total_claims_count:     Total number of claims made.
                                          0–1 per 5 policy years = low.
                                          2–3 = normal. 4–5 = elevated. 6+ = high / flag.
- claims_history.total_claims_paid:      Cumulative amount paid (£).
- claims_history.largest_single_claim:  Largest individual claim (£).
- claims_history.largest_claim_x_premium: largest_single_claim / annual_premium.
                                          < 3× = low severity. 3–10× = normal.
                                          > 10× = elevated. > 50× = investigate.
- claims_history.years_since_last_claim: Recency of claims activity.
                                          < 1 year = elevated (likely continuing condition).
                                          1–3 years = watch. 3+ years = low recency concern.
- claims_history.claim_types:            List of claim categories.
                                          Multiple identical types in short period = flag for review.
                                          Mixed types across years = normal policyholder behaviour.
- financial_metrics.loss_ratio:          Total claims / premiums. < 0.60 = favourable.
                                          0.60–0.85 = acceptable. > 0.85 = adverse.
- coverage.premium_annual:               Annual premium for severity normalisation.

Fraud / suspicious pattern indicators:
1. Claims frequency spike — ≥3 claims in the most recent 12 months.
2. Severity mismatch — largest claim >> average claim without a clear catastrophic event.
3. Claim type clustering — multiple claims of the same niche type (e.g., all jewellery theft).
4. Rapid policy take-up → early claim — first claim within 6 months of inception.
5. Loss ratio persistently > 0.90 over multiple years without rate revision.

Claims risk loading guideline:
- Low risk (0–1 claims, favourable loss ratio): 0% additional loading.
- Normal risk (2–3 claims, acceptable loss ratio): 0–10% loading.
- Elevated risk (4–5 claims or loss ratio 0.85–1.00): 10–25% loading.
- High risk (6+ claims or loss ratio > 1.00): 25–50% loading, refer to Chief Underwriter.
- Suspicious patterns detected: flag for investigation before renewal.

Output:
- claims_risk_level: low / medium / high / very_high.
- frequency_risk: low / normal / elevated / high.
- severity_risk: low / normal / elevated / high.
- loss_ratio_assessment: favourable (< 0.60) / acceptable (0.60–0.85) / adverse (> 0.85).
- suspicious_patterns: True if any fraud indicator is triggered.
- suspicious_pattern_notes: description of any flagged patterns (empty string if none).
- claims_loading_pct: additional premium % recommended based on claims experience.
- conviction: high / medium / low.\
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
    claims_keys = ["claims_history", "financial_metrics", "coverage"]
    claims_data = {k: payload[k] for k in claims_keys if k in payload}

    user_content = (
        f"Policy ID: {policy_id}  "
        f"Product: {payload.get('product_type', 'unknown')}  "
        f"Date: {payload.get('assessment_date', 'unknown')}\n\n"
        f"Claims data:\n{json.dumps(claims_data, indent=2)}"
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
