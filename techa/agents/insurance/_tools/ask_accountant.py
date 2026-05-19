"""
agents/insurance/_tools/ask_accountant.py — Insurance accountant AI analyst.

Reviews financial KPIs: loss ratios, expense ratios, combined ratio, reserve adequacy,
premium growth trends, and portfolio quality. Returns a structured AccountingAnalysis.

When payload["kpi_snapshot"] is present (computed by techa.insurance.build_kpi_snapshot
from a financial history time series), the richer computed KPIs — including OLS trend
slopes, CAGR, reserve adequacy ratios, and YoY growth — are sent to the model instead
of the raw scalar financial_metrics. The fallback is the scalar financial_metrics dict.
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

You will receive either a kpi_snapshot (preferred — computed from multi-period financial history)
or a raw financial_metrics dict (single-period scalars). Use whichever is present; prefer kpi_snapshot.

─── KPI SNAPSHOT FIELDS (when kpi_snapshot is present) ───────────────────────────────────────

Profitability (current period):
- loss_ratio             — net claims incurred / GWP. < 0.60 = excellent. 0.75–0.85 = acceptable. > 0.85 = adverse.
- expense_ratio          — expenses / GWP. < 0.25 = efficient. > 0.35 = high.
- combined_ratio         — loss_ratio + expense_ratio. < 0.85 = profitable. > 1.00 = loss-making.
- underwriting_margin_pct — (1 − combined_ratio) × 100. Positive = underwriting profit.
- underwriting_profit    — GWP − claims_incurred − expenses (£ absolute P&L).
- reinsurance_cession_pct — % of GWP ceded to reinsurers. High = heavy reinsurance dependency.
- net_claims_ratio       — claims / NWP (net-of-reinsurance loss ratio; higher than loss_ratio if cession % is high).

Profitability trends (OLS slope over last N periods):
- loss_ratio_trend       — slope of loss_ratio. Positive = deteriorating. Negative = improving.
- loss_ratio_trend_r2    — R² of the slope. < 0.3 = noisy (trend unreliable). ≥ 0.7 = directional.
- combined_ratio_trend   — slope of combined_ratio. Same interpretation.
- combined_ratio_trend_r2
- expense_ratio_trend    — slope of expense_ratio (positive = rising costs).
- expense_ratio_trend_r2

Reserve adequacy:
- reserve_adequacy_ratio  — reserve_held / reserve_required. < 1.0 = regulatory breach risk.
- reserve_adequacy_pct    — × 100. 100–120% = adequate. > 120% = over-reserved (capital tied up).
- reserve_surplus         — reserve_held − reserve_required (£). Negative = deficit.
- reserve_to_gwp_pct      — reserve_held / GWP × 100. Contextual depth indicator.
- claims_settlement_ratio — claims_paid / claims_incurred (0–1). 1 = all claims settled in period.
- claims_outstanding      — claims_incurred − claims_paid (£). Approximate open liability.
- claims_outstanding_ratio — claims_outstanding / GWP × 100.
- reserve_adequacy_trend  — slope of adequacy_ratio. Positive = reserves improving vs. requirement.
- reserve_adequacy_trend_r2

Growth:
- gwp_latest             — GWP this period (£).
- nwp_latest             — NWP this period (£).
- premium_growth_pp      — period-over-period GWP growth (%). Negative = premium income falling.
- premium_growth_yoy     — year-over-year GWP growth (%).
- claims_growth_pp       — period-over-period claims growth (%). If > premium_growth_pp = loss ratio worsening.
- claims_growth_yoy      — year-over-year claims growth (%).
- gwp_cagr               — CAGR of GWP over full history (%). Core volume growth metric.
- gwp_trend              — OLS slope of GWP (£/period). Positive = growing book.
- gwp_trend_r2           — R² of GWP trend.
- avg_premium            — GWP / policies_in_force (£/policy). Rising = premium rate hardening.
- lapse_rate             — % of policies that lapsed in the period. > 15% = retention problem.
- new_business_ratio     — new_policies / policies_in_force × 100. Growth composition indicator.

─── FALLBACK FINANCIAL_METRICS FIELDS (when kpi_snapshot is absent) ──────────────────────────

- financial_metrics.loss_ratio, expense_ratio, combined_ratio: as above.
- financial_metrics.reserve_held, reserve_required, reserve_adequacy_pct: reserve position.
- financial_metrics.premium_growth_yoy_pct: single YoY growth figure.
- coverage.premium_annual, sum_assured: policy-level context.

─── INTERPRETATION THRESHOLDS ────────────────────────────────────────────────────────────────

- Loss ratio > 0.85 on life: structural repricing required.
- Combined ratio > 1.00 for 2+ periods: business is loss-making at current rates.
- Reserve adequacy < 100%: immediate regulatory notification required (Solvency II breach risk).
- loss_ratio_trend > +0.01/period with R² ≥ 0.5: credible deterioration — flag for Chief Underwriter.
- GWP growth > 15% with rising loss_ratio: adverse selection risk (growing into unprofitable segments).
- claims_growth_pp > premium_growth_pp consistently: book is repricing below loss cost inflation.

Output:
- financial_health: overall financial position (strong / adequate / marginal / weak).
- combined_ratio_assessment: profitable (< 0.85) / breakeven (0.85–1.00) / loss_making (> 1.00).
- reserve_status: over_reserved / adequate / under_reserved.
- reserve_adequacy_pct: value from kpi_snapshot or financial_metrics.
- profitability_outlook: forward-looking (positive / neutral / negative).
- premium_growth_assessment: strong (> 10%) / stable (0–10%) / declining (< 0%).
- conviction: high (multi-period KPI snapshot) / medium (single-period) / low (minimal data).\
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
    Send financial KPIs to the model for accounting analysis.

    Prefers payload["kpi_snapshot"] (computed by techa.insurance.build_kpi_snapshot
    from multi-period history) over the raw scalar payload["financial_metrics"].
    Both paths are supported so the function degrades gracefully when only scalar
    point-in-time data is available.

    Args:
        payload:   Dict from prepare_node — contains kpi_snapshot (preferred),
                   financial_metrics (fallback), and coverage.
        policy_id: Application reference string.
        question:  Optional follow-up question.

    Returns:
        AccountingAnalysis Pydantic model.
    """
    kpi_snapshot = payload.get("kpi_snapshot")

    if kpi_snapshot:
        financial_section = {"kpi_snapshot": kpi_snapshot, "coverage": payload.get("coverage", {})}
        data_label = "KPI snapshot (multi-period)"
        log.info("Sending kpi_snapshot for %s to %s", policy_id, MODEL)
    else:
        financial_section = {
            "financial_metrics": payload.get("financial_metrics", {}),
            "coverage": payload.get("coverage", {}),
        }
        data_label = "financial_metrics (single-period)"
        log.info("Sending financial_metrics for %s to %s (no kpi_snapshot)", policy_id, MODEL)

    user_content = (
        f"Policy ID: {policy_id}  "
        f"Product: {payload.get('product_type', 'unknown')}  "
        f"Date: {payload.get('assessment_date', 'unknown')}  "
        f"Data: {data_label}\n\n"
        f"Financial data:\n{json.dumps(financial_section, indent=2)}"
    )
    if question:
        user_content += f"\n\nQuestion: {question}"

    return invoke_structured(
        AccountingAnalysis,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=1024,
    )
