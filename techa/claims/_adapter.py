"""
techa/claims/_adapter.py — Claim form and medical documentation validation.

Validates required fields, canonicalises claim_type, parses all dates,
and computes derived numeric fields consumed by domain modules.

Required fields: claim_type.
"""

from __future__ import annotations

from datetime import date, datetime

__all__ = ["validate_claim_form"]

REQUIRED_FIELDS = frozenset({"claim_type"})

_CLAIM_TYPE_ALIASES: dict[str, str] = {
    "ci":                          "critical_illness",
    "critical illness":            "critical_illness",
    "critical_illness_cover":      "critical_illness",
    "tpd":                         "total_permanent_disability",
    "total permanent disability":  "total_permanent_disability",
    "income protection":           "income_protection",
    "ip":                          "income_protection",
    "medical expense":             "medical_expense",
    "medical expenses":            "medical_expense",
    "hospital cash":               "hospital_cash",
    "death benefit":               "death",
    "life claim":                  "death",
    "life":                        "death",
}


def _parse_date(val) -> date | None:
    if val is None:
        return None
    if isinstance(val, date):
        return val
    try:
        return datetime.fromisoformat(str(val)).date()
    except (ValueError, TypeError):
        return None


def _canonical_claim_type(ct: str) -> str:
    raw = ct.lower().strip()
    return _CLAIM_TYPE_ALIASES.get(raw, raw.replace(" ", "_").replace("-", "_"))


def _to_list(val, lower: bool = True) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        items = [val]
    else:
        items = list(val)
    return [str(v).lower().strip() if lower else str(v).strip() for v in items]


def validate_claim_form(form: dict) -> dict:
    """
    Validate and enrich a raw claim form dict.

    Args:
        form: Raw claim form from the caller.

    Returns:
        Enriched dict with parsed dates and derived ratio fields (private keys
        prefixed with "_") consumed by domain modules.

    Raises:
        ValueError: If required fields are absent.
    """
    missing = REQUIRED_FIELDS - set(form)
    if missing:
        raise ValueError(f"Missing required claim form fields: {missing}")

    data = dict(form)

    # ── Canonical claim type ──────────────────────────────────────────────────
    data["claim_type"] = _canonical_claim_type(str(data["claim_type"]))

    # ── Parse dates ───────────────────────────────────────────────────────────
    event_date      = _parse_date(data.get("date_of_event"))
    submission_date = _parse_date(data.get("date_of_submission"))
    inception_date  = _parse_date(data.get("policy_inception_date"))
    admission_date  = _parse_date(data.get("admission_date"))
    discharge_date  = _parse_date(data.get("discharge_date"))

    data["_event_date"]      = event_date
    data["_submission_date"] = submission_date
    data["_inception_date"]  = inception_date
    data["_admission_date"]  = admission_date
    data["_discharge_date"]  = discharge_date

    # ── Date-derived deltas ───────────────────────────────────────────────────
    data["_policy_age_days"] = (
        (event_date - inception_date).days
        if inception_date and event_date else None
    )
    data["_submission_delay_days"] = (
        (submission_date - event_date).days
        if event_date and submission_date else None
    )
    data["_inpatient_duration_days"] = (
        (discharge_date - admission_date).days
        if admission_date and discharge_date else None
    )

    # ── Financial ratios ──────────────────────────────────────────────────────
    claim_amount = data.get("claim_amount_requested")
    sum_assured  = data.get("sum_assured")
    premium      = data.get("premium_annual")

    def _safe_ratio(num, den):
        try:
            n, d = float(num), float(den)
            return n / d if d > 0 else None
        except (TypeError, ValueError):
            return None

    data["_claim_to_sa_ratio"]      = _safe_ratio(claim_amount, sum_assured)
    data["_claim_to_premium_ratio"] = _safe_ratio(claim_amount, premium)

    # ── Normalise list fields ─────────────────────────────────────────────────
    for field in ("diagnosis", "icd_codes", "documents_submitted",
                  "pre_existing_conditions_declared"):
        data[field] = _to_list(data.get(field))

    return data
