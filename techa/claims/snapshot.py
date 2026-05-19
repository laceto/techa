"""
techa/claims/snapshot.py — Claims KPI snapshot orchestrator.

Responsibility: thin orchestrator.
    - Input validation via validate_claim_form() from _adapter.
    - Delegates to timeline, severity, medical_coherence, documentation,
      fraud_indicators.
    - Sums component loadings, caps total at 100%, derives claims_risk_level.
    - nan_to_none flag for JSON-serialisable output.

Public API
----------
build_claims_snapshot(claim_form, *, nan_to_none) -> dict
    Compute a claims assessment snapshot from a claim form dict.

Input claim_form (dict)
-----------------------
Required:
    claim_type  — str: "death" | "critical_illness" | "total_permanent_disability"
                       | "income_protection" | "medical_expense" | "hospital_cash"
                       | "accident"

Optional — timeline:
    date_of_event           — ISO date of the insured event.
    date_of_submission      — ISO date claim was submitted.
    policy_inception_date   — ISO date policy started.

Optional — financial:
    claim_amount_requested  — float, £ amount the insured is claiming.
    sum_assured             — float, £ maximum policy benefit.
    premium_annual          — float, £ annual premium.

Optional — medical documentation:
    diagnosis               — list[str], e.g. ["acute_myocardial_infarction"].
    icd_codes               — list[str], e.g. ["I21.9"].
    admission_date          — ISO date of hospital admission.
    discharge_date          — ISO date of discharge.
    prognosis               — str: "full_recovery" | "partial_recovery" | "permanent"
                              | "progressive" | "terminal" | "fatal".
    treatment_summary       — str, free-text description of treatment.
    treating_physician      — str.
    hospital_name           — str.

Optional — declarations:
    pre_existing_conditions_declared — list[str].
    medical_history_consistent       — bool; False triggers fraud indicator.
    nondisclosure_flag               — bool; True triggers fraud indicator.

Optional — documents:
    documents_submitted     — list[str], e.g. ["specialist_report", "hospital_records"].

Output snapshot keys
--------------------
Timeline (8):
    policy_age_days, policy_age_months, early_claim_flag, very_early_claim_flag,
    submission_delay_days, submission_delay_risk, inpatient_duration_days,
    timeline_loading_pct.

Severity (6):
    claim_amount_requested, sum_assured, claim_to_sa_ratio, claim_to_premium_ratio,
    severity_category, severity_loading_pct.

Medical coherence (6):
    diagnosis_categories, icd_chapters, primary_diagnosis_category,
    claim_type_match, prognosis_risk, coherence_loading_pct.

Documentation (6):
    documents_submitted_count, required_documents, missing_documents,
    documentation_completeness_pct, documentation_status, documentation_loading_pct.

Fraud indicators (5):
    fraud_flags, fraud_flag_descriptions, fraud_indicator_count,
    fraud_risk_level, fraud_loading_pct.

Aggregate (3):
    claim_type, claims_loading_pct (sum of components, capped at 100%),
    claims_risk_level.
"""

from __future__ import annotations

import logging
import math

from techa.claims._adapter import validate_claim_form
from techa.claims.timeline import compute_timeline
from techa.claims.severity import compute_severity
from techa.claims.medical_coherence import compute_medical_coherence
from techa.claims.documentation import compute_documentation
from techa.claims.fraud_indicators import compute_fraud_indicators

__all__ = ["build_claims_snapshot"]

log = logging.getLogger(__name__)

_LOADING_CAP = 100.0


def _claims_risk_level(total_loading: float, fraud_count: int) -> str:
    if fraud_count >= 3 or total_loading >= 50.0:
        return "very_high"
    if fraud_count >= 2 or total_loading >= 25.0:
        return "high"
    if fraud_count >= 1 or total_loading >= 10.0:
        return "medium"
    return "low"


def build_claims_snapshot(
    claim_form: dict,
    *,
    nan_to_none: bool = False,
) -> dict:
    """
    Compute a claims assessment snapshot from a claim form and medical documentation.

    Args:
        claim_form:  Dict with claim and medical data — see module docstring for schema.
                     Minimum: {"claim_type": str}.
        nan_to_none: Replace float NaN with None for JSON-serialisable output.
                     Default False.

    Returns:
        Flat dict of ~34 keys. See module docstring for key listing.
        NaN (or None when nan_to_none=True) for any metric whose input is absent.

    Raises:
        ValueError: If required fields (claim_type) are missing.

    Example:
        from techa.claims import build_claims_snapshot

        snap = build_claims_snapshot({
            "claim_type":              "critical_illness",
            "date_of_event":           "2025-10-01",
            "date_of_submission":      "2025-10-20",
            "policy_inception_date":   "2020-03-01",
            "claim_amount_requested":  500_000,
            "sum_assured":             500_000,
            "premium_annual":          3_200,
            "diagnosis":               ["acute_myocardial_infarction"],
            "icd_codes":               ["I21.9"],
            "admission_date":          "2025-10-01",
            "discharge_date":          "2025-10-08",
            "prognosis":               "partial_recovery",
            "nondisclosure_flag":      False,
            "documents_submitted":     ["specialist_report", "hospital_records",
                                        "medical_report"],
        }, nan_to_none=True)

        print(snap["claims_risk_level"])    # "low"
        print(snap["claims_loading_pct"])   # 0.0
        print(snap["claim_type_match"])     # True
    """
    data = validate_claim_form(claim_form)

    result: dict = {}
    result.update(compute_timeline(data))
    result.update(compute_severity(data))
    result.update(compute_medical_coherence(data))

    # documentation module needs claim_type already resolved — use data dict
    result.update(compute_documentation(data))

    # fraud_indicators uses documentation_status derived above
    data["documentation_status"] = result.get("documentation_status", "")
    result.update(compute_fraud_indicators(data))

    result["claim_type"] = data["claim_type"]

    # ── Aggregate loading (capped at 100%) ────────────────────────────────────
    components = [
        result.get("timeline_loading_pct"),
        result.get("severity_loading_pct"),
        result.get("coherence_loading_pct"),
        result.get("documentation_loading_pct"),
        result.get("fraud_loading_pct"),
    ]
    total = sum(c for c in components if c is not None and not math.isnan(c))
    total = min(total, _LOADING_CAP)
    result["claims_loading_pct"] = round(total, 1)

    result["claims_risk_level"] = _claims_risk_level(
        total, result.get("fraud_indicator_count", 0)
    )

    log.debug(
        "build_claims_snapshot: claim_type=%s policy_age=%s days loading=%.1f risk=%s",
        data.get("claim_type"),
        data.get("_policy_age_days"),
        total,
        result["claims_risk_level"],
    )

    if nan_to_none:
        result = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in result.items()
        }

    return result
