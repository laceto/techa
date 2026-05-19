"""
techa/claims/documentation.py — Claim documentation completeness check.

Compares the documents the insured has submitted against the minimum required
set for their claim type. Produces a completeness percentage, a list of
missing documents, and a status classification.

Required documents by claim type
---------------------------------
death:
    death_certificate, attending_physician_report, proof_of_identity_beneficiary

critical_illness:
    specialist_report, hospital_records, medical_report

total_permanent_disability:
    specialist_report, medical_report, functional_assessment_report

income_protection:
    medical_report, employer_confirmation, proof_of_income

medical_expense:
    medical_report, receipts_invoices

hospital_cash:
    hospital_admission_records, discharge_summary

accident:
    accident_report, medical_report

Documentation status thresholds
--------------------------------
100%:     complete
75–99%:   partial
< 75%:    insufficient

Documentation loading
---------------------
complete:     0%
partial:      10%
insufficient: 25%
no documents: 50%
"""

from __future__ import annotations

__all__ = ["compute_documentation"]

_REQUIRED_DOCS: dict[str, frozenset[str]] = {
    "death": frozenset({
        "death_certificate",
        "attending_physician_report",
        "proof_of_identity_beneficiary",
    }),
    "critical_illness": frozenset({
        "specialist_report",
        "hospital_records",
        "medical_report",
    }),
    "total_permanent_disability": frozenset({
        "specialist_report",
        "medical_report",
        "functional_assessment_report",
    }),
    "income_protection": frozenset({
        "medical_report",
        "employer_confirmation",
        "proof_of_income",
    }),
    "medical_expense": frozenset({
        "medical_report",
        "receipts_invoices",
    }),
    "hospital_cash": frozenset({
        "hospital_admission_records",
        "discharge_summary",
    }),
    "accident": frozenset({
        "accident_report",
        "medical_report",
    }),
}

# Acceptable aliases for common document names
_DOC_ALIASES: dict[str, str] = {
    "hospital_discharge_summary":    "discharge_summary",
    "discharge_report":              "discharge_summary",
    "gp_report":                     "medical_report",
    "gp_letter":                     "medical_report",
    "doctors_report":                "medical_report",
    "consultants_report":            "specialist_report",
    "consultant_report":             "specialist_report",
    "specialist_letter":             "specialist_report",
    "hospital_notes":                "hospital_records",
    "inpatient_records":             "hospital_records",
    "hospital_admission_record":     "hospital_admission_records",
    "functional_assessment":         "functional_assessment_report",
    "receipts":                      "receipts_invoices",
    "invoices":                      "receipts_invoices",
    "proof_of_identity":             "proof_of_identity_beneficiary",
    "id_proof":                      "proof_of_identity_beneficiary",
    "salary_slip":                   "proof_of_income",
    "payslip":                       "proof_of_income",
    "employer_letter":               "employer_confirmation",
    "police_report":                 "accident_report",
    "coroner_certificate":           "death_certificate",
    "death_registration":            "death_certificate",
}


def _normalise_docs(docs: list[str]) -> set[str]:
    result: set[str] = set()
    for d in docs:
        key = d.lower().strip().replace(" ", "_").replace("-", "_")
        result.add(_DOC_ALIASES.get(key, key))
    return result


def _status(pct: float) -> str:
    if pct >= 100.0:
        return "complete"
    if pct >= 75.0:
        return "partial"
    return "insufficient"


def _loading(pct: float, submitted_count: int) -> float:
    if submitted_count == 0:
        return 50.0
    if pct >= 100.0:
        return 0.0
    if pct >= 75.0:
        return 10.0
    return 25.0


def compute_documentation(data: dict) -> dict:
    """
    Assess documentation completeness for the claim type.

    Args:
        data: Validated claim form dict from validate_claim_form().

    Returns:
        Flat dict: documents_submitted_count, required_documents, missing_documents,
        documentation_completeness_pct, documentation_status, documentation_loading_pct.
    """
    claim_type  = data.get("claim_type", "unknown")
    submitted   = _normalise_docs(data.get("documents_submitted", []))
    required    = _REQUIRED_DOCS.get(claim_type, frozenset())

    if required:
        present  = submitted & required
        missing  = sorted(required - submitted)
        pct      = round(100.0 * len(present) / len(required), 1)
    else:
        missing  = []
        pct      = 100.0 if submitted else 0.0

    return {
        "documents_submitted_count":       len(submitted),
        "required_documents":              sorted(required),
        "missing_documents":               missing,
        "documentation_completeness_pct":  pct,
        "documentation_status":            _status(pct),
        "documentation_loading_pct":       _loading(pct, len(submitted)),
    }
