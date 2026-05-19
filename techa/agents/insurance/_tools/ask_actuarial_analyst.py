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
You have three pre-computed actuarial snapshot tools available in the payload. Use whichever
are present; each is independent — the absence of one does not invalidate the others.

━━━ Tool 1: ae_snapshot — Actual vs Expected (A/E) experience monitoring ━━━
  period_count, actual_total, expected_total
  aggregate_ae_ratio, aggregate_ae_pct         — overall A/E (1.0 = 100% = on-model)
  ae_alert_level                               — green (90–110%) / amber / red (< 80% or > 130%)
  credibility_weight                           — Bühlmann Z: 0=thin data, 1=fully credible
  credibility_weighted_ae                      — blend of observed A/E and 100% prior
  z_score, z_score_significant                 — Poisson significance test (|z| > 1.96 = 95% CI)
  cumulative_deviation                         — running actual − expected total
  max_ae_ratio, min_ae_ratio, periods_above_expected
  ae_trend_slope, ae_trend_r2, ae_trend_direction  — improving / stable / deteriorating
  ae_volatility                                — std dev of period A/E ratios
  risk_type, product_type

  Interpretation: ae_alert_level=red or ae_trend_direction=deteriorating with z_score_significant=True
  means actual mortality is running materially above pricing assumptions — loading revision needed.

━━━ Tool 2: pricing_snapshot — New reinsurance deal pricing ━━━
  treaty_type, term_years, discount_rate
  total_ceded_premium, total_ceded_claims, total_expenses, total_commission, total_profit
  npv                              — net present value at discount_rate
  payback_period_years             — years until cumulative cash flows turn positive (null=never)
  loss_ratio                       — ceded_claims / ceded_premium
  expense_ratio, commission_ratio, profit_margin
  break_even_loss_ratio            — maximum loss ratio for a profitable treaty
  stress_ae25_loss_ratio           — loss ratio under A/E +25% mortality shock
  stress_ae50_loss_ratio           — loss ratio under A/E +50% mortality shock
  stress_ae25_profit_margin, stress_ae50_profit_margin
  pricing_adequacy                 — adequate (margin > 10%) / marginal / inadequate
  cumulative_cash_flows            — list, one entry per treaty year

  Interpretation: pricing_adequacy=inadequate or loss_ratio > break_even_loss_ratio means
  the treaty is not commercially viable at the proposed rates.

━━━ Tool 3: inforce_snapshot — In-force portfolio health assessment ━━━
  period_count, pif_latest (policies in force), gwp_latest
  pif_cagr, gwp_cagr                           — annualised portfolio growth
  avg_lapse_rate, persistency_rate
  lapse_trend_slope, lapse_trend_r2, lapse_trend_direction
  avg_mortality_rate_ppm                       — deaths per million exposed per period
  new_business_ratio                           — new policies / PIF (latest period)
  bel_latest, risk_margin_latest, scr_latest   — Solvency II balance sheet (£)
  solvency_coverage_ratio                      — own_funds / SCR (≥1.5 = adequate, <1.0 = breach)
  solvency_status                              — adequate / watch / breach
  bel_to_annualised_gwp                        — reserve depth relative to premium income
  risk_margin_ratio                            — risk_margin / BEL
  bel_trend_slope, bel_trend_r2               — BEL growth trend

  Interpretation: solvency_status=breach or lapse_trend_direction=deteriorating requires
  immediate management action. bel_to_annualised_gwp > 5× indicates heavy long-tail reserves.

━━━ Tool 4: geo_snapshot — Geospatial / epidemiological enrichment ━━━
  postcode_area, imd_decile
  imd_risk_band                                — high (1–3) / elevated (4–6) / low (7–10)
  imd_loading_pct                              — 15% (decile 1–3) / 5% (4–6) / 0% (7–10)
  regional_ae_index                            — local mortality vs national (1.0 = national)
  regional_ae_loading_pct                      — (ae_index − 1) × 100, clamped [−20, +30]
  hospital_quality_score, hospital_access_loading_pct  — 5% if score < 50
  geo_total_loading_pct                        — sum of above, clamped [0, 40]
  geo_risk_level                               — low / elevated / high

  Interpretation: geo_total_loading_pct should be added to individual mortality loading.
  A high-deprivation area (imd_decile 1–3) combined with regional_ae_index > 1.15
  materially elevates expected claims frequency.

━━━ Tool 5: concentration_snapshot — Portfolio concentration risk ━━━
  policy_sum_assured, portfolio_total_sa, portfolio_policy_count
  sa_concentration_pct                         — this policy / total SA × 100
  avg_policy_sa                                — portfolio_total_sa / policy_count
  sa_multiple_of_average                       — how many × larger than avg policy
  concentration_flag                           — True if > 0.5% of SA or > 5× average
  concentration_loading_pct                    — 0 / 5 / 10 / 20% tiered by sa_multiple
  net_retention_recommendation                 — suggested per-life reinsurance retention (£)
  reinsurance_trigger                          — True if sum_assured > retention limit
  concentration_risk_level                     — low / elevated / high

  Interpretation: reinsurance_trigger=True means this policy exceeds the recommended
  retention limit — cession to a reinsurer is required before policy issuance.

━━━ Fallback (when snapshots are absent) ━━━
  Use raw applicant and financial_metrics fields:
  - age/gender baseline: male +20–30% vs female at ages 40–60; each decade above 50 adds ~50% mortality.
  - smoker: ×2–3 mortality; ex-smoker < 5yr: ×1.5.
  - bmi_category: obese +25–75%; severely_obese +75–150%; morbidly_obese → decline.
  - bp_category: stage1 +25–50%; stage2 +50–100%; crisis → postpone.
  - medical_history: type2 diabetes +50–200%; heart disease/stroke/cancer → postpone or decline.
  - family_history: ≥2 CV relatives before 60 → +25%; hereditary cancer → +25–50%.
  - financial_metrics.loss_ratio > 0.85 → adverse portfolio experience.

Risk classifications:
  standard:    A/E ≤ 130% of table; individual loading = 0%.
  substandard: Loading 1–250%. Issue with extra premium.
  postpone:    A/E > 150% or single condition warrants deferral; reassess in 12–24 months.
  decline:     Uninsurable (> 250% or catastrophic A/E).

Output:
- mortality_percentile: 0–100. 50 = average standard risk. Calibrate from ae_snapshot when available.
- expected_loss_ratio:  Target 0.60–0.75 at the recommended loading.
- mortality_loading_pct: Extra premium % recommended. Use pricing_snapshot.profit_margin to validate.
- risk_classification: standard / substandard / postpone / decline.
- key_risk_factors: ≤5 factors ranked by materiality; quote snapshot values.
- conviction: high (complete snapshots + credible A/E data) / medium / low.
- verdict: One sentence; quote total loading %, classification, and primary driver.\
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
    actuarial_data: dict = {}

    # Prefer pre-computed snapshots; include all that are present
    for snap_key in (
        "ae_snapshot", "pricing_snapshot", "inforce_snapshot",
        "geo_snapshot", "concentration_snapshot",
    ):
        snap = payload.get(snap_key)
        if snap:
            actuarial_data[snap_key] = snap

    # Always include applicant + coverage for individual risk context
    for k in ("applicant", "coverage"):
        if k in payload:
            actuarial_data[k] = payload[k]

    # Fallback financial context when no portfolio snapshots present
    if not any(k in actuarial_data for k in ("ae_snapshot", "inforce_snapshot")):
        if "financial_metrics" in payload:
            actuarial_data["financial_metrics"] = payload["financial_metrics"]

    _PORTFOLIO_SNAP_KEYS = (
        "ae_snapshot", "pricing_snapshot", "inforce_snapshot",
        "geo_snapshot", "concentration_snapshot",
    )
    has_snapshots = any(k in actuarial_data for k in _PORTFOLIO_SNAP_KEYS)
    data_label = "Actuarial snapshots (pre-computed KPIs)" if has_snapshots else "Actuarial data"

    user_content = (
        f"Policy ID: {policy_id}  "
        f"Product: {payload.get('product_type', 'unknown')}  "
        f"Date: {payload.get('assessment_date', 'unknown')}\n\n"
        f"{data_label}:\n{json.dumps(actuarial_data, indent=2)}"
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
