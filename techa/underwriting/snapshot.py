"""
techa/underwriting/snapshot.py — Medical underwriting KPI snapshot builder.

Responsibility: thin orchestrator.
    - Input validation via validate_questionnaire() from _adapter.
    - Delegates all domain computations to biometrics, cardiovascular, metabolic,
      lifestyle, conditions, family_risk.
    - Computes occupation loading, sums all component loadings, caps total at 250%,
      derives composite risk_score (0–100).
    - nan_to_none flag for JSON-serialisable output.

Public API
----------
build_medical_snapshot(questionnaire, *, nan_to_none) -> dict
    Compute a medical underwriting snapshot from a questionnaire dict.

Input questionnaire (dict)
--------------------------
Required:
    age      — int, applicant age at entry.
    gender   — str, "male" | "female" | "non_binary".

Optional biometrics:
    height_cm, weight_kg       — used to derive bmi (or pass bmi directly).
    bmi                        — float; used when height/weight absent.
    systolic_bp, diastolic_bp  — mmHg integers.
    total_cholesterol          — mmol/L float.
    hdl_cholesterol            — mmol/L float.
    fasting_glucose            — mmol/L float.
    hba1c                      — % float (NGSP scale).

Optional lifestyle:
    smoker                     — bool.
    smoking_status             — "never" | "current" | "ex" (overrides smoker bool).
    cigarettes_per_day         — int (current/ex smokers).
    years_smoked               — int.
    years_quit                 — int (ex-smokers).
    alcohol_units_per_week     — float.

Optional history:
    medical_history            — list[str], e.g. ["hypertension", "type2_diabetes"].
    medications                — list[str], e.g. ["metformin", "lisinopril"].
    family_history             — list[str], e.g. ["cardiovascular_disease"].
    family_history_age_at_onset — dict[str, int], condition → age of affected relative.

Optional occupation:
    occupation_class           — int 1–4 (1=office, 4=hazardous).

Optional investigation results:
    ecg_normal, chest_xray_normal, gp_report_clear — bool; used as conviction modifiers.

Output snapshot keys
--------------------
Biometrics:
    bmi, bmi_category, bmi_loading_pct.

Cardiovascular:
    bp_systolic, bp_diastolic, pulse_pressure, bp_category, bp_loading_pct,
    cholesterol_ratio, cholesterol_risk, cholesterol_loading_pct, cv_risk_score.

Metabolic:
    diabetes_status, hba1c, hba1c_category, fasting_glucose, glucose_category,
    metabolic_loading_pct.

Lifestyle:
    smoking_status, pack_years, cigarettes_per_day, years_quit,
    smoking_loading_pct, alcohol_units_per_week, alcohol_risk, alcohol_loading_pct.

Medical conditions:
    condition_count, conditions_loading_pct, critical_condition_flag.

Family history:
    family_risk_factor_count, family_history_loading_pct, hereditary_cancer_risk.

Aggregate:
    occupation_class, occupation_loading_pct,
    total_medical_loading_pct  — sum of all component loadings, capped at 250%.
    risk_score                 — composite 0–100 (total_loading × 0.4, capped at 100).
"""

from __future__ import annotations

import logging
import math

from techa.underwriting._adapter import validate_questionnaire
from techa.underwriting.biometrics import compute_biometrics
from techa.underwriting.cardiovascular import compute_cardiovascular
from techa.underwriting.metabolic import compute_metabolic
from techa.underwriting.lifestyle import compute_lifestyle
from techa.underwriting.conditions import compute_conditions
from techa.underwriting.family_risk import compute_family_risk

__all__ = ["build_medical_snapshot"]

log = logging.getLogger(__name__)

_LOADING_CAP = 250.0

_OCCUPATION_LOADINGS: dict[int, float] = {
    1: 0.0,    # office / professional
    2: 15.0,   # light manual / clerical
    3: 50.0,   # manual / outdoor
    4: 125.0,  # hazardous (mining, diving, roofing, etc.)
}


def build_medical_snapshot(
    questionnaire: dict,
    *,
    nan_to_none: bool = False,
) -> dict:
    """
    Compute a medical underwriting snapshot from a structured questionnaire.

    Args:
        questionnaire: Dict with applicant data — see module docstring for full schema.
                       Minimum: {"age": int, "gender": str}.
        nan_to_none:   Replace float NaN with None for JSON-serialisable output.
                       Default False.

    Returns:
        Flat dict of scalars (~35 keys). See module docstring for key listing.
        NaN (or None when nan_to_none=True) for any metric whose required input
        field is absent from the questionnaire.

    Raises:
        ValueError: If required fields (age, gender) are missing.

    Example:
        from techa.underwriting import build_medical_snapshot

        snap = build_medical_snapshot({
            "age": 45, "gender": "male",
            "height_cm": 178, "weight_kg": 90,
            "systolic_bp": 138, "diastolic_bp": 88,
            "total_cholesterol": 5.8, "hdl_cholesterol": 1.1,
            "smoker": False, "years_quit": 6,
            "alcohol_units_per_week": 18,
            "medical_history": ["hypertension", "type2_diabetes"],
            "medications": ["lisinopril", "metformin"],
            "family_history": ["cardiovascular_disease"],
            "family_history_age_at_onset": {"cardiovascular_disease": 55},
            "occupation_class": 1,
        }, nan_to_none=True)

        print(snap["total_medical_loading_pct"])   # 215.0
        print(snap["risk_score"])                  # 86.0
        print(snap["critical_condition_flag"])     # False
    """
    data = validate_questionnaire(questionnaire)

    result: dict = {}
    result.update(compute_biometrics(data))
    result.update(compute_cardiovascular(data))
    result.update(compute_metabolic(data))
    result.update(compute_lifestyle(data))
    result.update(compute_conditions(data))
    result.update(compute_family_risk(data))

    # ── Occupation ────────────────────────────────────────────────────────────
    occ_class = data.get("occupation_class")
    try:
        occ_class = int(occ_class) if occ_class is not None else None
    except (TypeError, ValueError):
        occ_class = None
    occ_load = _OCCUPATION_LOADINGS.get(occ_class, float("nan")) if occ_class else float("nan")

    result["occupation_class"]      = occ_class
    result["occupation_loading_pct"] = occ_load

    # ── Total loading (sum of all components, cap at 250%) ────────────────────
    components = [
        result.get("bmi_loading_pct"),
        result.get("bp_loading_pct"),
        result.get("cholesterol_loading_pct"),
        result.get("metabolic_loading_pct"),
        result.get("smoking_loading_pct"),
        result.get("alcohol_loading_pct"),
        result.get("conditions_loading_pct"),
        result.get("family_history_loading_pct"),
        occ_load,
    ]

    total = sum(c for c in components if c is not None and not math.isnan(c))
    total = min(total, _LOADING_CAP)
    result["total_medical_loading_pct"] = round(total, 1)

    # ── Risk score 0–100 ──────────────────────────────────────────────────────
    result["risk_score"] = round(min(100.0, total * 0.4), 1)

    log.debug(
        "build_medical_snapshot: age=%s gender=%s total_loading=%.1f risk_score=%.1f",
        data.get("age"),
        data.get("gender"),
        total,
        result["risk_score"],
    )

    if nan_to_none:
        result = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in result.items()
        }

    return result
