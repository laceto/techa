"""
techa/underwriting/metabolic.py — Diabetes, glucose, and HbA1c assessment.

Public API
----------
compute_metabolic(data) -> dict
    Input is the validated dict from _adapter.validate_questionnaire().
    Returns diabetes status, HbA1c/glucose categories, and metabolic loading %.

Diabetes status derivation
--------------------------
Detected from medical_history keywords + medications + lab values:
  type1_diabetes:       "type1_diabetes" or "type 1 diabetes" in medical_history,
                        OR insulin in medications.
  type2_diabetes:       "type2_diabetes", "type 2 diabetes", "diabetes" in history.
  pre_diabetic:         "pre_diabetes", "impaired_fasting_glucose", "igt" in history,
                        OR fasting_glucose 6.1–6.9 mmol/L,
                        OR hba1c 5.7–6.4%.
  none:                 no evidence of diabetes.

HbA1c categories (%)
---------------------
< 5.7    normal
5.7–6.4  pre_diabetic
6.5–7.4  diabetic_controlled
7.5–9.0  diabetic_moderate
> 9.0    diabetic_uncontrolled

Fasting glucose categories (mmol/L)
------------------------------------
< 5.6    normal
5.6–6.9  impaired  (pre-diabetic range)
≥ 7.0    diabetic

Metabolic loading schedule
--------------------------
pre_diabetic:           +25%
type2, diet-only:       +75%
type2, oral medication: +100%  (detected via metformin/glipizide/sitagliptin in medications)
type2, insulin:         +150%  (any insulin in medications + type2 history)
type1, HbA1c < 7.5:     +200%
type1, HbA1c ≥ 7.5:     +250%  (critical; may warrant postpone)
"""

from __future__ import annotations

import math

__all__ = ["compute_metabolic"]

nan = float("nan")

_ORAL_ANTIDIABETICS = {
    "metformin", "glipizide", "gliclazide", "glibenclamide",
    "sitagliptin", "empagliflozin", "dapagliflozin", "canagliflozin",
    "pioglitazone", "acarbose", "liraglutide", "semaglutide",
    "dulaglutide", "exenatide",
}
_INSULINS = {
    "insulin", "lantus", "glargine", "detemir", "degludec",
    "humalog", "novorapid", "aspart", "lispro", "glulisine",
    "novomix", "mixtard",
}


def _hba1c_category(hba1c: float) -> str:
    if hba1c < 5.7:  return "normal"
    if hba1c < 6.5:  return "pre_diabetic"
    if hba1c < 7.5:  return "diabetic_controlled"
    if hba1c < 9.0:  return "diabetic_moderate"
    return "diabetic_uncontrolled"


def _glucose_category(glucose: float) -> str:
    if glucose < 5.6: return "normal"
    if glucose < 7.0: return "impaired"
    return "diabetic"


def _diabetes_status(history: list[str], meds: list[str], hba1c: float, glucose: float) -> str:
    hist_lower = {h.lower().replace(" ", "_").replace("-", "_") for h in history}
    meds_lower = {m.lower() for m in meds}

    has_insulin = bool(_INSULINS & meds_lower)
    has_oral    = bool(_ORAL_ANTIDIABETICS & meds_lower)

    # Type 1
    if "type1_diabetes" in hist_lower or "type_1_diabetes" in hist_lower:
        return "type1"
    if has_insulin and ("type2_diabetes" not in hist_lower and "type_2_diabetes" not in hist_lower):
        return "type1"

    # Type 2
    if any(k in hist_lower for k in ("type2_diabetes", "type_2_diabetes", "diabetes")):
        if has_insulin:
            return "type2_insulin"
        if has_oral:
            return "type2_oral"
        return "type2_diet"

    # Pre-diabetic (by history, HbA1c, or glucose)
    if any(k in hist_lower for k in ("pre_diabetes", "prediabetes", "impaired_fasting_glucose", "igt")):
        return "pre_diabetic"
    if not math.isnan(hba1c) and 5.7 <= hba1c < 6.5:
        return "pre_diabetic"
    if not math.isnan(glucose) and 5.6 <= glucose < 7.0:
        return "pre_diabetic"

    return "none"


def _metabolic_loading(status: str, hba1c: float) -> float:
    if status == "none":         return 0.0
    if status == "pre_diabetic": return 25.0
    if status == "type2_diet":   return 75.0
    if status == "type2_oral":   return 100.0
    if status == "type2_insulin": return 150.0
    if status == "type1":
        if not math.isnan(hba1c) and hba1c >= 7.5:
            return 250.0
        return 200.0
    return nan


def compute_metabolic(data: dict) -> dict:
    """
    Assess diabetes, glucose, and HbA1c status.

    Args:
        data: Validated questionnaire dict from validate_questionnaire().

    Returns:
        Flat dict with diabetes_status, hba1c, glucose, categories, and loading.
    """
    history = data.get("medical_history", [])
    meds    = data.get("medications", [])

    hba1c   = data.get("hba1c",           nan)
    glucose = data.get("fasting_glucose",  nan)
    if hba1c   is None: hba1c   = nan
    if glucose is None: glucose = nan
    hba1c   = float(hba1c)
    glucose = float(glucose)

    status = _diabetes_status(history, meds, hba1c, glucose)

    return {
        "diabetes_status":      status,
        "hba1c":                hba1c,
        "hba1c_category":       _hba1c_category(hba1c) if not math.isnan(hba1c) else "unknown",
        "fasting_glucose":      glucose,
        "glucose_category":     _glucose_category(glucose) if not math.isnan(glucose) else "unknown",
        "metabolic_loading_pct": _metabolic_loading(status, hba1c),
    }
