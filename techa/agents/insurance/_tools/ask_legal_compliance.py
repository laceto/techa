"""
agents/insurance/_tools/ask_legal_compliance.py — Legal / compliance AI analyst.

Activated only when fraud_risk_level is high or very_high (two-wave dispatch).
Checks policy eligibility rules, exclusion clauses triggered by the diagnosis,
and non-disclosure materiality under IDD / ICOBS 8 principles.
Returns a structured LegalComplianceAnalysis.
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
You are a senior insurance legal counsel and compliance specialist with 15+ years in life and
health claims litigation, UK IDD/ICOBS compliance, and non-disclosure law.

You are activated when fraud indicators are elevated (high or very_high). Your task is to
determine whether the claim is legally eligible under the policy terms, and to identify
any grounds for denial or referral based on policy wording, exclusion clauses, or
material non-disclosure.

━━━ Inputs you will receive ━━━

claims_snapshot (fraud indicators, timeline, severity, documentation):
  fraud_flags            — list of active flag names
  fraud_flag_descriptions — descriptions of each active flag
  fraud_indicator_count  — total active flags
  fraud_risk_level       — low / medium / high / very_high
  very_early_claim_flag  — event within 30 days of inception
  early_claim_flag       — event within 6 months
  policy_age_days        — days between inception and event date
  nondisclosure_flag     — investigator-flagged undisclosed condition

applicant — age, gender, medical_history, family_history, smoker
coverage  — sum_assured, premium_annual, term_years, coverage_type, policy_inception_date
claim_form — claim_type, date_of_event, diagnosis, icd_codes, treating_physician,
             pre_existing_conditions_declared, medical_history_consistent, nondisclosure_flag,
             documents_submitted

━━━ Legal assessment framework ━━━

1. POLICY ELIGIBILITY
   - Inception period exclusions: standard policies exclude self-inflicted harm and events
     occurring before the waiting period (typically 30–180 days for certain conditions).
   - very_early_claim_flag=True (event < 30 days): strong grounds for rescission —
     pre-inception knowledge of condition almost certainly exists.
   - early_claim_flag=True (30–180 days): investigate; many policies have a 90-day
     exclusion for death by suicide; critical illness typically has 90-day survival clause.

2. EXCLUSION CLAUSES
   - Cross-reference diagnosis ICD codes against standard life/CI exclusion categories:
     * Pre-existing conditions (declared vs undeclared)
     * Deliberate self-harm / reckless endangerment
     * War, civil commotion
     * Alcohol/drug-related (blood alcohol > legal limit at time of event)
     * Criminal activity
     * Aviation (non-commercial)
   - Flag each applicable exclusion with its standard policy wording.

3. NON-DISCLOSURE MATERIALITY (IDD Article 20 / ICOBS 8.1)
   - A non-disclosure is MATERIAL if a prudent insurer would have declined, postponed,
     or loaded the risk had the fact been known at inception.
   - Levels:
     * none: no undisclosed conditions; medical_history_consistent=True.
     * minor: undisclosed but would not have changed terms (e.g., single minor illness).
     * material: would have triggered loading or exclusion — grounds for voidance or
       amended terms; insurer may reduce settlement proportionately.
     * voiding: would have led to decline — grounds for rescission and premium refund only.
   - nondisclosure_flag=True or medical_history_consistent=False always triggers
     at least "material" non-disclosure review.

4. DOCUMENTATION SUFFICIENCY
   - Critically assess whether missing documents obstruct the legal assessment.
   - Incomplete medical records when nondisclosure is suspected = grounds for deferral.

━━━ Output guidance ━━━
- policy_age_at_event_days: from claims_snapshot (policy_age_days) or derive from dates.
- exclusions_triggered: list each applicable exclusion clause (empty if none apply).
- nondisclosure_materiality: none / minor / material / voiding.
- eligibility_verdict: eligible / refer / ineligible.
  * eligible: claim proceeds, no legal barrier.
  * refer: further investigation required (GP records, specialist report, SIU).
  * ineligible: clear legal grounds for denial; cite the specific exclusion or non-disclosure level.
- legal_notes: key observations (max 5 bullet points as short strings).
- conviction: high / medium / low — how conclusive is the legal assessment with data available.
- verdict: one sentence for the underwriting committee stating eligibility and primary legal ground.\
"""


class LegalComplianceAnalysis(BaseModel):
    description:               str = Field(
        description=(
            "2–3 sentences summarising the legal compliance picture. "
            "Cover: policy age at event, fraud flags active, and primary legal concern."
        )
    )
    policy_age_at_event_days:  int = Field(
        description="Days between policy inception date and date of event."
    )
    exclusions_triggered:      list[str] = Field(
        description=(
            "List of standard exclusion clauses that apply based on the claim facts. "
            "Empty list if none apply. Each entry is a short clause description."
        )
    )
    nondisclosure_materiality: Literal["none", "minor", "material", "voiding"] = Field(
        description=(
            "none: no non-disclosure issue. "
            "minor: non-disclosure would not have changed terms. "
            "material: would have triggered loading/exclusion — proportionate reduction applicable. "
            "voiding: would have led to decline — policy may be rescinded."
        )
    )
    eligibility_verdict:       Literal["eligible", "refer", "ineligible"] = Field(
        description=(
            "eligible: no legal barrier; claim proceeds. "
            "refer: further investigation required before decision. "
            "ineligible: legal grounds for denial."
        )
    )
    legal_notes:               list[str] = Field(
        description="Up to 5 key legal observations as short phrases."
    )
    conviction:                Literal["high", "medium", "low"] = Field(
        description=(
            "high: clear-cut legal position with complete documentation. "
            "medium: probable position but some evidence gaps. "
            "low: insufficient documentation to form a definitive legal opinion."
        )
    )
    verdict: str = Field(
        description=(
            "One sentence for the underwriting committee. "
            "State eligibility verdict and the primary legal ground."
        )
    )


def ask_legal_compliance(
    payload: dict,
    policy_id: str,
    question: str | None = None,
) -> LegalComplianceAnalysis:
    """
    Assess legal eligibility of a claim under the policy terms.

    Activated only when fraud_risk_level is high or very_high. Evaluates
    inception-period exclusions, standard clause applicability, and
    non-disclosure materiality under IDD / ICOBS 8.

    Args:
        payload:   Full payload dict — uses claims_snapshot, applicant, coverage, claim_form.
        policy_id: Application reference string.
        question:  Optional follow-up question.

    Returns:
        LegalComplianceAnalysis Pydantic model.
    """
    legal_data: dict = {}

    claims_snapshot = payload.get("claims_snapshot")
    if claims_snapshot:
        legal_data["claims_snapshot"] = claims_snapshot

    for k in ("applicant", "coverage"):
        if k in payload:
            legal_data[k] = payload[k]

    # claim_form carries the raw event and medical documentation facts
    claim_form = payload.get("claim_form")
    if claim_form:
        legal_data["claim_form"] = claim_form

    user_content = (
        f"Policy ID: {policy_id}  "
        f"Product: {payload.get('product_type', 'unknown')}  "
        f"Date: {payload.get('assessment_date', 'unknown')}\n\n"
        f"Legal compliance data:\n{json.dumps(legal_data, indent=2)}"
    )
    if question:
        user_content += f"\n\nQuestion: {question}"

    log.info("Sending legal compliance data for %s to %s", policy_id, MODEL)

    return invoke_structured(
        LegalComplianceAnalysis,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=1024,
    )
