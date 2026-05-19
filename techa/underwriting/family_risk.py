"""
techa/underwriting/family_risk.py — Family history hereditary risk assessment.

Public API
----------
compute_family_risk(data) -> dict
    Input is the validated dict from _adapter.validate_questionnaire().
    Returns family_risk_factor_count, family_history_loading_pct,
    and hereditary_cancer_risk flag.

Loading rules
-------------
Each family history entry contributes a base loading. If the optional field
family_history_age_at_onset is provided and the relative developed the condition
before age 60 (early onset), the loading for that entry is doubled.

Loading per condition type (per affected first-degree relative)
--------------------------------------------------------------
cardiovascular_disease / heart_disease / coronary_artery_disease:  +15% each
                (max 2 relatives counted; additional relatives: +5% each)
stroke:                                                             +10%
type2_diabetes:                                                     +10%
type1_diabetes:                                                     +15%
cancer (general):                                                   +15%
breast_cancer / ovarian_cancer:                                     +25% (BRCA risk)
bowel_cancer / colorectal_cancer:                                   +15%
prostate_cancer:                                                    +10%
hypertension:                                                       +5%
alzheimers / dementia:                                              +15%
multiple_sclerosis:                                                 +10%
huntingtons / huntington:                                           +50% (direct hereditary)
brca1 / brca2 (genetic diagnosis):                                  +50%
polycystic_kidney / pkd:                                            +25%

Hereditary cancer risk flag
---------------------------
True when any of the following is present:
  breast_cancer, ovarian_cancer (if female applicant or flagged in history),
  bowel_cancer, colorectal_cancer, brca1, brca2, lynch_syndrome,
  hereditary_breast_ovarian_cancer.
"""

from __future__ import annotations

__all__ = ["compute_family_risk"]

_FAMILY_LOADINGS: dict[str, float] = {
    "cardiovascular_disease":           15.0,
    "heart_disease":                    15.0,
    "coronary_artery_disease":          15.0,
    "ischemic_heart_disease":           15.0,
    "stroke":                           10.0,
    "type2_diabetes":                   10.0,
    "type_2_diabetes":                  10.0,
    "type1_diabetes":                   15.0,
    "diabetes":                         10.0,
    "cancer":                           15.0,
    "breast_cancer":                    25.0,
    "ovarian_cancer":                   25.0,
    "bowel_cancer":                     15.0,
    "colorectal_cancer":                15.0,
    "colon_cancer":                     15.0,
    "prostate_cancer":                  10.0,
    "lung_cancer":                      15.0,
    "pancreatic_cancer":                20.0,
    "hypertension":                      5.0,
    "high_blood_pressure":               5.0,
    "alzheimers":                       15.0,
    "dementia":                         15.0,
    "multiple_sclerosis":               10.0,
    "huntingtons":                      50.0,
    "huntington":                       50.0,
    "brca1":                            50.0,
    "brca2":                            50.0,
    "polycystic_kidney":                25.0,
    "pkd":                              25.0,
    "lynch_syndrome":                   25.0,
}

_HEREDITARY_CANCER_TRIGGERS: frozenset[str] = frozenset({
    "breast_cancer", "ovarian_cancer", "bowel_cancer", "colorectal_cancer",
    "colon_cancer", "pancreatic_cancer", "brca1", "brca2", "lynch_syndrome",
    "hereditary_breast_ovarian_cancer",
})

_CV_CONDITIONS: frozenset[str] = frozenset({
    "cardiovascular_disease", "heart_disease", "coronary_artery_disease",
    "ischemic_heart_disease",
})


def _canonical(condition: str) -> str:
    return condition.lower().strip().replace(" ", "_").replace("-", "_")


def compute_family_risk(data: dict) -> dict:
    """
    Assess family history hereditary risk and loading.

    Args:
        data: Validated questionnaire dict from validate_questionnaire().

    Returns:
        Flat dict: family_risk_factor_count, family_history_loading_pct,
        hereditary_cancer_risk.
    """
    raw_history      = data.get("family_history", [])
    age_at_onset_map = data.get("family_history_age_at_onset") or {}

    total_loading     = 0.0
    hereditary_cancer = False
    factor_count      = 0
    cv_count          = 0

    for raw in raw_history:
        key = _canonical(raw)
        base = _FAMILY_LOADINGS.get(key, 0.0)
        if base == 0.0:
            continue  # unknown condition — conservative: no loading rather than default

        factor_count += 1

        # CV condition cap: first 2 at full rate, additional at +5%
        if key in _CV_CONDITIONS:
            cv_count += 1
            if cv_count > 2:
                base = 5.0

        # Early onset multiplier (x2 if relative affected before age 60)
        onset_age = age_at_onset_map.get(raw) or age_at_onset_map.get(key)
        if onset_age is not None:
            try:
                if float(onset_age) < 60:
                    base *= 2.0
            except (TypeError, ValueError):
                pass

        if key in _HEREDITARY_CANCER_TRIGGERS:
            hereditary_cancer = True

        total_loading += base

    return {
        "family_risk_factor_count":   factor_count,
        "family_history_loading_pct": total_loading,
        "hereditary_cancer_risk":     hereditary_cancer,
    }
