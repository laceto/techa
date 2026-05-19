"""
techa/underwriting/_adapter.py — Medical questionnaire validation and biometric derivation.

All domain modules (biometrics, cardiovascular, metabolic, lifestyle, conditions,
family_risk) receive the dict returned by validate_questionnaire(). This is the
single point for arithmetic derivations that require no medical judgment — BMI,
pack-years, cholesterol ratio, pulse pressure. Actuarial loading rules stay in
the domain modules.

Public API
----------
validate_questionnaire(q)  — validate required fields and compute derived biometrics.
REQUIRED_FIELDS            — minimum set of questionnaire fields.
nan                        — float("nan") sentinel used throughout the module.
"""

from __future__ import annotations

import math

__all__ = ["validate_questionnaire", "REQUIRED_FIELDS", "nan"]

nan: float = float("nan")

REQUIRED_FIELDS: frozenset[str] = frozenset({"age", "gender"})


def _safe(value, *, default: float = nan) -> float:
    """Return float(value) or default when value is None / non-numeric."""
    if value is None:
        return default
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _smoking_status(q: dict) -> str:
    """
    Derive canonical smoking status from questionnaire fields.
    Priority: explicit smoking_status field → smoker bool → "unknown".
    """
    explicit = q.get("smoking_status")
    if explicit in ("never", "current", "ex"):
        return explicit
    smoker = q.get("smoker")
    if smoker is True:
        return "current"
    if smoker is False:
        yrs_quit = q.get("years_quit")
        return "ex" if yrs_quit is not None else "never"
    return "unknown"


def _pack_years(q: dict) -> float:
    """
    pack_years = (cigarettes_per_day / 20) × years_smoked.
    Returns nan when insufficient data.
    """
    cpd = _safe(q.get("cigarettes_per_day"))
    yrs = _safe(q.get("years_smoked"))
    if math.isnan(cpd) or math.isnan(yrs):
        return nan
    return (cpd / 20.0) * yrs


def validate_questionnaire(q: dict) -> dict:
    """
    Validate a medical questionnaire dict and compute derived biometric fields.

    Required fields: age, gender.
    All other fields are optional; derived values are float("nan") when inputs
    are absent or non-numeric.

    Derived fields added to the returned dict:
        bmi              — weight_kg / (height_cm / 100) ** 2
                           Falls back to q["bmi"] when height/weight are absent.
        smoking_status   — "never" | "current" | "ex" | "unknown"
        pack_years       — (cigarettes_per_day / 20) × years_smoked
        cholesterol_ratio — total_cholesterol / hdl_cholesterol  (TC/HDL)
        pulse_pressure   — systolic_bp − diastolic_bp

    Args:
        q: Raw questionnaire dict from the caller or risk_profile["applicant"].

    Returns:
        Enriched dict with all original fields plus derived biometrics.

    Raises:
        ValueError: If any required field is missing.
    """
    missing = REQUIRED_FIELDS - set(q.keys())
    if missing:
        raise ValueError(
            f"build_medical_snapshot: missing required questionnaire fields: "
            f"{sorted(missing)}."
        )

    data = dict(q)  # shallow copy; domain modules must not mutate

    # ── BMI ──────────────────────────────────────────────────────────────────
    h = _safe(data.get("height_cm"))
    w = _safe(data.get("weight_kg"))
    if not math.isnan(h) and not math.isnan(w) and h > 0:
        data["bmi"] = w / (h / 100.0) ** 2
    elif "bmi" not in data or data["bmi"] is None:
        data["bmi"] = nan

    # ── Smoking ───────────────────────────────────────────────────────────────
    data["smoking_status"] = _smoking_status(data)
    data["pack_years"]     = _pack_years(data)

    # ── Cholesterol ratio ─────────────────────────────────────────────────────
    tc  = _safe(data.get("total_cholesterol"))
    hdl = _safe(data.get("hdl_cholesterol"))
    data["cholesterol_ratio"] = (
        tc / hdl if not (math.isnan(tc) or math.isnan(hdl) or hdl == 0.0)
        else nan
    )

    # ── Blood pressure pulse pressure ─────────────────────────────────────────
    sys_ = _safe(data.get("systolic_bp"))
    dia  = _safe(data.get("diastolic_bp"))
    data["pulse_pressure"] = (
        sys_ - dia if not (math.isnan(sys_) or math.isnan(dia))
        else nan
    )

    # ── Normalise list fields to plain lists ──────────────────────────────────
    for key in ("medical_history", "medications", "family_history"):
        val = data.get(key)
        if val is None:
            data[key] = []
        elif not isinstance(val, list):
            data[key] = list(val)
        else:
            data[key] = [str(c).lower().strip() for c in val]

    return data
