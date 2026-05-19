"""
agents/insurance/_tools/ask_accountant.py — Insurance accountant AI analyst.

Reviews financial metrics: premium income, loss ratios, expense ratios, and
reserve adequacy. Returns a structured AccountingAnalysis.
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
You are a qualified insurance accountant (CIMA/ACA) with 15+ years in life and health insurance.
You assess the financial viability, reserve adequacy, and profitability of insurance arrangements.

Relevant payload fields:
- financial_metrics.loss_ratio:         Net claims incurred / gross premium earned.
                                         < 0.60 = excellent. 0.60–0.75 = good.
                                         0.75–0.85 = acceptable. > 0.85 = adverse.
- financial_metrics.expense_ratio:      Management expenses / gross premium.
                                         < 0.25 = efficient. 0.25–0.35 = normal.
                                         > 0.35 = high (may indicate operational issues).
- financial_metrics.combined_ratio:     loss_ratio + expense_ratio.
                                         < 0.85 = profitable. 0.85–1.00 = marginal.
                                         > 1.00 = loss-making (every £1 premium costs > £1 in claims + expenses).
- financial_metrics.reserve_held:       Actual reserve balance held (£).
- financial_metrics.reserve_required:   Minimum reserve required by actuarial calculation (£).
- financial_metrics.reserve_adequacy_pct: reserve_held / reserve_required × 100.
                                         < 100% = under-reserved (regulatory breach risk).
                                         100–120% = adequate. > 120% = over-reserved (capital inefficiency).
- financial_metrics.premium_growth_yoy_pct: Year-on-year growth in gross premium income (%).
                                         > 10% = strong growth. 0–10% = steady. < 0% = shrinking.
- coverage.premium_annual:              Annualised premium for this policy (£).
- coverage.sum_assured:                 Death benefit / cover amount (£). Key for reserve sizing.

Interpretation guidelines:
- A loss ratio above 0.85 on a life product is a red flag; life products should be
  profitable after year 3 once acquisition costs are recovered.
- Reserve adequacy below 100% requires immediate attention and regulator notification.
- Combined ratio > 1.00 for more than two consecutive years signals structural repricing need.
- High premium growth with deteriorating loss ratios indicates adverse selection.

Output:
- financial_health: overall financial position (strong / adequate / marginal / weak).
- combined_ratio_assessment: profitable (< 0.85) / breakeven (0.85–1.00) / loss_making (> 1.00).
- reserve_status: over_reserved / adequate / under_reserved.
- reserve_adequacy_pct: the value from the payload.
- profitability_outlook: forward-looking view given current trends (positive / neutral / negative).
- premium_growth_assessment: strong / stable / declining based on premium_growth_yoy_pct.
- conviction: high / medium / low based on data completeness and signal clarity.\
"""


class AccountingAnalysis(BaseModel):
    description:              str = Field(
        description=(
            "2–3 sentence summary of the financial position. "
            "Cover: combined ratio, reserve status, and profitability outlook. "
            "Quote exact ratios from the payload."
        )
    )
    financial_health:         Literal["strong", "adequate", "marginal", "weak"] = Field(
        description=(
            "strong: combined < 0.85 and reserves adequate. "
            "adequate: combined 0.85–1.00 and reserves ≥ 100%. "
            "marginal: combined near 1.00 or reserves borderline. "
            "weak: loss-making or under-reserved."
        )
    )
    combined_ratio:           float = Field(description="Combined ratio value from payload.")
    combined_ratio_assessment: Literal["profitable", "breakeven", "loss_making"] = Field(
        description="profitable: < 0.85. breakeven: 0.85–1.00. loss_making: > 1.00."
    )
    loss_ratio:               float = Field(description="Loss ratio value from payload.")
    expense_ratio:            float = Field(description="Expense ratio value from payload.")
    reserve_status:           Literal["over_reserved", "adequate", "under_reserved"] = Field(
        description="over_reserved: > 120%. adequate: 100–120%. under_reserved: < 100%."
    )
    reserve_adequacy_pct:     float = Field(
        description="Reserve adequacy percentage from payload (reserve_held / reserve_required × 100)."
    )
    premium_growth_assessment: Literal["strong", "stable", "declining"] = Field(
        description="strong: > 10% yoy. stable: 0–10%. declining: < 0%."
    )
    profitability_outlook:    Literal["positive", "neutral", "negative"] = Field(
        description=(
            "positive: combined improving and reserves healthy. "
            "neutral: stable but no clear improvement trend. "
            "negative: deteriorating ratios or under-reserving."
        )
    )
    financial_loading_pct:    float = Field(
        description=(
            "Additional premium % recommended to restore financial viability. "
            "0.0 if current pricing is adequate. Positive when loss_ratio > 0.75."
        )
    )
    conviction:               Literal["high", "medium", "low"] = Field(
        description="high: complete financial data. medium: partial data. low: insufficient data."
    )
    verdict:                  str = Field(
        description=(
            "One actionable sentence for a senior underwriter. "
            "Include combined ratio, reserve status, and any repricing recommendation."
        )
    )


def ask_accountant(
    payload: dict,
    policy_id: str,
    question: str | None = None,
) -> AccountingAnalysis:
    """
    Send the financial metrics to the model for accounting analysis.

    Args:
        payload:   Dict from prepare_node — contains financial_metrics and coverage.
        policy_id: Application reference string.
        question:  Optional follow-up question.

    Returns:
        AccountingAnalysis Pydantic model.
    """
    accounting_keys = ["financial_metrics", "coverage"]
    accounting_data = {k: payload[k] for k in accounting_keys if k in payload}

    user_content = (
        f"Policy ID: {policy_id}  "
        f"Product: {payload.get('product_type', 'unknown')}  "
        f"Date: {payload.get('assessment_date', 'unknown')}\n\n"
        f"Financial data:\n{json.dumps(accounting_data, indent=2)}"
    )
    if question:
        user_content += f"\n\nQuestion: {question}"

    log.info("Sending financial data for %s to %s", policy_id, MODEL)

    return invoke_structured(
        AccountingAnalysis,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=1024,
    )
