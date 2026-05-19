"""
techa/claims — Insurance claims KPI snapshot builder.

Primary entry point
-------------------
build_claims_snapshot(claim_form, *, nan_to_none) -> dict
    Claims assessment snapshot from a structured claim form dict that includes
    both the claim form data and the medical documentation supplied by the insured.

    Required fields: claim_type.
    Optional fields: date_of_event, date_of_submission, policy_inception_date,
                     claim_amount_requested, sum_assured, premium_annual,
                     diagnosis (list), icd_codes (list),
                     admission_date, discharge_date, prognosis, treatment_summary,
                     treating_physician, hospital_name,
                     pre_existing_conditions_declared (list),
                     medical_history_consistent, nondisclosure_flag,
                     documents_submitted (list).

    Output groups:
        Timeline        — policy_age_days, policy_age_months, early_claim_flag,
                          very_early_claim_flag, submission_delay_days,
                          submission_delay_risk, inpatient_duration_days,
                          timeline_loading_pct.
        Severity        — claim_amount_requested, sum_assured, claim_to_sa_ratio,
                          claim_to_premium_ratio, severity_category,
                          severity_loading_pct.
        Medical         — diagnosis_categories, icd_chapters,
                          primary_diagnosis_category, claim_type_match,
                          prognosis_risk, coherence_loading_pct.
        Documentation   — documents_submitted_count, required_documents,
                          missing_documents, documentation_completeness_pct,
                          documentation_status, documentation_loading_pct.
        Fraud           — fraud_flags, fraud_flag_descriptions,
                          fraud_indicator_count, fraud_risk_level,
                          fraud_loading_pct.
        Aggregate       — claim_type, claims_loading_pct (capped at 100%),
                          claims_risk_level.
"""

from techa.claims.snapshot import build_claims_snapshot

__all__ = ["build_claims_snapshot"]
