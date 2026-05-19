"""
techa/underwriting — Medical underwriting KPI snapshot builder.

Primary entry point
-------------------
build_medical_snapshot(questionnaire, *, nan_to_none) -> dict
    Medical underwriting snapshot from a structured questionnaire dict.

    Required fields:  age, gender.
    Optional fields:  height_cm, weight_kg (or bmi), systolic_bp, diastolic_bp,
                      total_cholesterol, hdl_cholesterol, fasting_glucose, hba1c,
                      smoker, smoking_status, cigarettes_per_day, years_smoked,
                      years_quit, alcohol_units_per_week,
                      medical_history (list), medications (list),
                      family_history (list), family_history_age_at_onset (dict),
                      occupation_class (1–4).

    Output groups:
        Biometrics     — bmi, bmi_category, bmi_loading_pct.
        Cardiovascular — bp_systolic, bp_diastolic, pulse_pressure, bp_category,
                         bp_loading_pct, cholesterol_ratio, cholesterol_risk,
                         cholesterol_loading_pct, cv_risk_score.
        Metabolic      — diabetes_status, hba1c, hba1c_category, fasting_glucose,
                         glucose_category, metabolic_loading_pct.
        Lifestyle      — smoking_status, pack_years, cigarettes_per_day, years_quit,
                         smoking_loading_pct, alcohol_units_per_week, alcohol_risk,
                         alcohol_loading_pct.
        Conditions     — condition_count, conditions_loading_pct, critical_condition_flag.
        Family history — family_risk_factor_count, family_history_loading_pct,
                         hereditary_cancer_risk.
        Aggregate      — occupation_class, occupation_loading_pct,
                         total_medical_loading_pct, risk_score.
"""

from techa.underwriting.snapshot import build_medical_snapshot

__all__ = ["build_medical_snapshot"]
