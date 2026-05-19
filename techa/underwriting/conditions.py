"""
techa/underwriting/conditions.py — Medical history aggregate loading.

Public API
----------
compute_conditions(data) -> dict
    Input is the validated dict from _adapter.validate_questionnaire().
    Returns aggregate loading, condition count, and critical-condition flag.

Loading table
-------------
Loading values follow standard reinsurance market rates (Swiss Re / Munich Re
guidelines). Each entry is (base_loading_pct, is_critical).
is_critical = True means the condition alone may warrant postpone or decline.

Conditions not in the table receive a default loading of +25% (unknown, mild).
Duplicates in medical_history (e.g. "diabetes" and "type2_diabetes") are
deduplicated via canonical key normalisation before summing.

BMI and smoking are handled in their own modules; they do not appear here.
Hypertension loading is halved when systolic_bp is controlled (< 140) on
medication, as a treated hypertension discount.

Combined loading rule
---------------------
Loadings from independent conditions are summed. The sum is NOT capped here —
capping to 250% happens at the snapshot level to preserve per-condition detail.
"""

from __future__ import annotations

__all__ = ["compute_conditions"]

# (loading_pct, is_critical)
_CONDITION_TABLE: dict[str, tuple[float, bool]] = {
    # Cardiovascular
    "hypertension":                 (50.0,  False),
    "hypertension_uncontrolled":    (100.0, True),
    "coronary_artery_disease":      (200.0, True),
    "angina":                       (150.0, True),
    "myocardial_infarction":        (200.0, True),
    "heart_failure":                (250.0, True),
    "atrial_fibrillation":          (100.0, False),
    "cardiomyopathy":               (150.0, True),
    "peripheral_vascular_disease":  (100.0, False),
    # Neurological
    "stroke":                       (200.0, True),
    "tia":                          (150.0, True),
    "epilepsy_controlled":          (50.0,  False),
    "epilepsy_uncontrolled":        (150.0, True),
    "multiple_sclerosis":           (150.0, True),
    "parkinsons":                   (200.0, True),
    "alzheimers":                   (200.0, True),
    "motor_neurone_disease":        (250.0, True),
    # Respiratory
    "copd":                         (100.0, False),
    "asthma_mild":                  (25.0,  False),
    "asthma_moderate":              (50.0,  False),
    "asthma_severe":                (100.0, False),
    "pulmonary_fibrosis":           (200.0, True),
    # Metabolic / endocrine (base; metabolic.py computes detail)
    "type2_diabetes":               (100.0, False),
    "type1_diabetes":               (200.0, True),
    "pre_diabetes":                 (25.0,  False),
    "hypothyroidism":               (25.0,  False),
    "hyperthyroidism":              (50.0,  False),
    # Renal / hepatic
    "kidney_disease":               (100.0, True),
    "chronic_kidney_disease":       (150.0, True),
    "liver_disease":                (100.0, True),
    "cirrhosis":                    (250.0, True),
    "non_alcoholic_fatty_liver":    (50.0,  False),
    # Oncology
    "cancer_remission_5yr":         (75.0,  False),
    "cancer_remission_2to5yr":      (150.0, True),
    "cancer_active":                (250.0, True),
    "melanoma":                     (100.0, False),
    # Gastrointestinal
    "crohns_disease":               (75.0,  False),
    "ulcerative_colitis":           (75.0,  False),
    "ibd":                          (75.0,  False),
    # Musculoskeletal
    "rheumatoid_arthritis":         (50.0,  False),
    "osteoporosis":                 (25.0,  False),
    "ankylosing_spondylitis":       (50.0,  False),
    # Mental health
    "anxiety":                      (25.0,  False),
    "depression":                   (25.0,  False),
    "depression_severe":            (75.0,  True),
    "bipolar_disorder":             (100.0, True),
    "schizophrenia":                (150.0, True),
    "eating_disorder":              (100.0, True),
    "substance_abuse":              (100.0, True),
    # Infectious / immune
    "hiv_treated":                  (75.0,  False),
    "hepatitis_b":                  (75.0,  False),
    "hepatitis_c_cured":            (50.0,  False),
    "hepatitis_c_active":           (150.0, True),
    "systemic_lupus":               (150.0, True),
    # Other
    "sleep_apnea":                  (25.0,  False),
    "chronic_fatigue_syndrome":     (50.0,  False),
}

# Canonical aliases: normalise questionnaire strings to table keys
_ALIASES: dict[str, str] = {
    "type 2 diabetes":    "type2_diabetes",
    "type2diabetes":      "type2_diabetes",
    "t2dm":               "type2_diabetes",
    "type 1 diabetes":    "type1_diabetes",
    "type1diabetes":      "type1_diabetes",
    "t1dm":               "type1_diabetes",
    "heart attack":       "myocardial_infarction",
    "mi":                 "myocardial_infarction",
    "htn":                "hypertension",
    "high blood pressure":"hypertension",
    "asthma":             "asthma_moderate",
    "tia":                "tia",
    "mini stroke":        "tia",
    "cancer in remission":"cancer_remission_5yr",
    "cancer":             "cancer_remission_5yr",
    "depression and anxiety": "depression",
    "ms":                 "multiple_sclerosis",
    "ibd":                "crohns_disease",
    "ckd":                "chronic_kidney_disease",
    "nafld":              "non_alcoholic_fatty_liver",
    "copd":               "copd",
    "sleep apnoea":       "sleep_apnea",
    "sleep apnea":        "sleep_apnea",
}

_DEFAULT_LOADING = 25.0  # unknown condition


def _canonical(condition: str) -> str:
    c = condition.lower().strip().replace(" ", "_").replace("-", "_")
    return _ALIASES.get(c, _ALIASES.get(condition.lower().strip(), c))


def compute_conditions(data: dict) -> dict:
    """
    Compute aggregate medical history loading.

    Args:
        data: Validated questionnaire dict from validate_questionnaire().

    Returns:
        Flat dict: condition_count, conditions_loading_pct, critical_condition_flag.
    """
    history = data.get("medical_history", [])
    sys_    = data.get("systolic_bp")
    treated_htn = "hypertension" in [_canonical(c) for c in history]

    # Deduplicate via canonical keys
    seen: set[str] = set()
    canonical_conditions: list[str] = []
    for raw in history:
        key = _canonical(raw)
        if key not in seen:
            seen.add(key)
            canonical_conditions.append(key)

    total_loading  = 0.0
    critical_flag  = False
    counted        = 0

    for key in canonical_conditions:
        loading, critical = _CONDITION_TABLE.get(key, (_DEFAULT_LOADING, False))

        # Treated hypertension discount: halve loading if BP controlled
        if key == "hypertension" and treated_htn:
            try:
                if sys_ is not None and float(sys_) < 140:
                    loading = loading / 2.0
            except (TypeError, ValueError):
                pass

        # Skip metabolic conditions that metabolic.py already prices in detail
        # to avoid double-counting (metabolic.py loading takes precedence)
        if key in ("type2_diabetes", "type1_diabetes", "pre_diabetes"):
            critical_flag = critical_flag or critical
            counted += 1
            continue  # loading counted in metabolic module

        total_loading += loading
        critical_flag = critical_flag or critical
        counted += 1

    return {
        "condition_count":        counted,
        "conditions_loading_pct": total_loading,
        "critical_condition_flag": critical_flag,
    }
