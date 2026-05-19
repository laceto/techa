"""
techa/underwriting/cardiovascular.py — Blood pressure, cholesterol, and CV risk.

Public API
----------
compute_cardiovascular(data) -> dict
    Input is the validated dict from _adapter.validate_questionnaire().
    Returns BP category/loading, cholesterol risk/loading, and a composite
    cardiovascular risk score (0–100).

BP loading schedule (ACC/AHA 2018 categories)
---------------------------------------------
Normal          systolic < 120 AND diastolic < 80           0%
Elevated        systolic 120–129 AND diastolic < 80         +10%
Stage 1 HTN     systolic 130–139 OR diastolic 80–89         +25%
Stage 2 HTN     systolic 140–159 OR diastolic 90–99         +75%
Hypertensive    systolic ≥ 160 OR diastolic ≥ 100           +150%   (postpone)

Treatment discount: if "hypertension" in medical_history AND BP is Stage 1 or below,
subtract 10% from loading (evidence of treated, controlled hypertension).

Cholesterol loading schedule (TC/HDL ratio)
-------------------------------------------
< 3.5   very_low        0%
3.5–4.0 low             0%
4.0–5.0 borderline      +15%
5.0–6.0 high            +40%
6.0–7.0 very_high       +75%
≥ 7.0   extreme         +100%

CV risk score (0–100): simplified composite of age/gender baseline,
BP severity, cholesterol, smoking contribution.
- Age/gender adds 0–30 points (risk rises with age; male > female at 40–60)
- BP adds 0–25 points
- Cholesterol adds 0–20 points
- Smoking adds 0–25 points (sourced from lifestyle snapshot if available; estimated here)
"""

from __future__ import annotations

import math

__all__ = ["compute_cardiovascular"]

nan = float("nan")


def _bp_category(sys_: float, dia: float) -> str:
    if math.isnan(sys_):
        return "unknown"
    if sys_ < 120 and (math.isnan(dia) or dia < 80):
        return "normal"
    if sys_ < 130 and (math.isnan(dia) or dia < 80):
        return "elevated"
    if sys_ < 140 or (not math.isnan(dia) and dia < 90):
        return "stage1_hypertension"
    if sys_ < 160 or (not math.isnan(dia) and dia < 100):
        return "stage2_hypertension"
    return "hypertensive_crisis"


def _bp_loading(category: str, treated: bool) -> float:
    base = {
        "normal":               0.0,
        "elevated":             10.0,
        "stage1_hypertension":  25.0,
        "stage2_hypertension":  75.0,
        "hypertensive_crisis":  150.0,
        "unknown":              nan,
    }.get(category, nan)
    if math.isnan(base):
        return nan
    # Controlled hypertension discount
    discount = 10.0 if treated and category in ("elevated", "stage1_hypertension") else 0.0
    return max(0.0, base - discount)


def _cholesterol_risk(ratio: float) -> str:
    if ratio < 3.5: return "very_low"
    if ratio < 4.0: return "low"
    if ratio < 5.0: return "borderline"
    if ratio < 6.0: return "high"
    if ratio < 7.0: return "very_high"
    return "extreme"


def _cholesterol_loading(ratio: float) -> float:
    if ratio < 4.0: return 0.0
    if ratio < 5.0: return 15.0
    if ratio < 6.0: return 40.0
    if ratio < 7.0: return 75.0
    return 100.0


def _cv_risk_score(
    age: float,
    gender: str,
    bp_category: str,
    cholesterol_ratio: float,
    smoking_status: str,
) -> float:
    """
    Simplified composite cardiovascular risk score (0–100).
    Not a formal Framingham score — for screening/loading guidance only.
    """
    score = 0.0

    # Age/gender component (0–30)
    if not math.isnan(age):
        male = str(gender).lower() in ("male", "m")
        if age < 35:
            score += 0.0
        elif age < 45:
            score += 10.0 if male else 5.0
        elif age < 55:
            score += 18.0 if male else 12.0
        elif age < 65:
            score += 25.0 if male else 20.0
        else:
            score += 30.0

    # BP component (0–25)
    score += {
        "normal":               0.0,
        "elevated":             5.0,
        "stage1_hypertension":  12.0,
        "stage2_hypertension":  20.0,
        "hypertensive_crisis":  25.0,
    }.get(bp_category, 0.0)

    # Cholesterol component (0–20)
    if not math.isnan(cholesterol_ratio):
        if cholesterol_ratio >= 7.0:
            score += 20.0
        elif cholesterol_ratio >= 6.0:
            score += 15.0
        elif cholesterol_ratio >= 5.0:
            score += 10.0
        elif cholesterol_ratio >= 4.0:
            score += 5.0

    # Smoking component (0–25)
    score += {"current": 25.0, "ex": 12.0, "never": 0.0}.get(
        smoking_status, 0.0
    )

    return round(min(100.0, score), 1)


def compute_cardiovascular(data: dict) -> dict:
    """
    Assess blood pressure, cholesterol, and composite CV risk.

    Args:
        data: Validated questionnaire dict from validate_questionnaire().

    Returns:
        Flat dict with BP fields, cholesterol fields, and cv_risk_score.
    """
    sys_ = data.get("systolic_bp",  nan)
    dia  = data.get("diastolic_bp", nan)
    age  = data.get("age",          nan)
    if sys_ is None: sys_ = nan
    if dia  is None: dia  = nan
    if age  is None: age  = nan
    sys_ = float(sys_) if sys_ is not None else nan
    dia  = float(dia)  if dia  is not None else nan
    age  = float(age)  if age  is not None else nan

    gender           = str(data.get("gender", "unknown"))
    smoking_status   = data.get("smoking_status", "unknown")
    cholesterol_ratio = data.get("cholesterol_ratio", nan)
    if cholesterol_ratio is None:
        cholesterol_ratio = nan
    cholesterol_ratio = float(cholesterol_ratio)

    treated_htn = "hypertension" in data.get("medical_history", [])

    bp_cat   = _bp_category(sys_, dia)
    bp_load  = _bp_loading(bp_cat, treated_htn)
    chol_risk = (
        _cholesterol_risk(cholesterol_ratio)
        if not math.isnan(cholesterol_ratio)
        else "unknown"
    )
    chol_load = (
        _cholesterol_loading(cholesterol_ratio)
        if not math.isnan(cholesterol_ratio)
        else nan
    )
    cv_score = _cv_risk_score(age, gender, bp_cat, cholesterol_ratio, smoking_status)

    return {
        "bp_systolic":            sys_,
        "bp_diastolic":           dia,
        "pulse_pressure":         data.get("pulse_pressure", nan),
        "bp_category":            bp_cat,
        "bp_loading_pct":         bp_load,
        "cholesterol_ratio":      cholesterol_ratio,
        "cholesterol_risk":       chol_risk,
        "cholesterol_loading_pct": chol_load,
        "cv_risk_score":          cv_score,
    }
