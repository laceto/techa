"""
techa/underwriting/biometrics.py — BMI assessment and loading.

Public API
----------
compute_biometrics(data) -> dict
    Input is the validated dict from _adapter.validate_questionnaire().
    Returns BMI category and standard actuarial loading %.

BMI loading schedule (life insurance, standard market rates)
-------------------------------------------------------------
< 18.5  underweight       +25%   (possible underlying illness)
18.5–24.9  normal          0%
25.0–27.4  overweight_mild +10%
27.5–29.9  overweight      +25%
30.0–32.4  obese_1         +50%
32.5–34.9  obese_1_high    +75%
35.0–37.4  obese_2         +100%
37.5–39.9  obese_2_high    +150%
≥ 40.0  morbidly_obese    +200%  (decline for life cover > £500k)
"""

from __future__ import annotations

import math

__all__ = ["compute_biometrics"]


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:  return "underweight"
    if bmi < 25.0:  return "normal"
    if bmi < 30.0:  return "overweight"
    if bmi < 35.0:  return "obese"
    if bmi < 40.0:  return "severely_obese"
    return "morbidly_obese"


def _bmi_loading(bmi: float) -> float:
    if bmi < 18.5:  return 25.0
    if bmi < 25.0:  return 0.0
    if bmi < 27.5:  return 10.0
    if bmi < 30.0:  return 25.0
    if bmi < 32.5:  return 50.0
    if bmi < 35.0:  return 75.0
    if bmi < 37.5:  return 100.0
    if bmi < 40.0:  return 150.0
    return 200.0


def compute_biometrics(data: dict) -> dict:
    """
    Assess BMI and return loading.

    Args:
        data: Validated questionnaire dict from validate_questionnaire().

    Returns:
        Flat dict with bmi, bmi_category, bmi_loading_pct.
        NaN for bmi and 0.0 loading when height/weight absent.
    """
    bmi = data.get("bmi", float("nan"))
    if not isinstance(bmi, float):
        try:
            bmi = float(bmi)
        except (TypeError, ValueError):
            bmi = float("nan")

    if math.isnan(bmi):
        return {
            "bmi":              float("nan"),
            "bmi_category":     "unknown",
            "bmi_loading_pct":  float("nan"),
        }

    return {
        "bmi":             round(bmi, 1),
        "bmi_category":    _bmi_category(bmi),
        "bmi_loading_pct": _bmi_loading(bmi),
    }
