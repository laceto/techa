"""
techa/claims/medical_coherence.py — Medical documentation coherence assessment.

Checks whether the supplied diagnosis and ICD codes are consistent with the
claim type, and classifies prognosis severity.

Diagnosis categorisation
------------------------
Diagnosis strings are mapped to broad clinical categories. ICD-10 codes are
classified by their leading chapter letter. Both sources are combined to
determine the primary diagnosis category.

Claim type → expected diagnosis categories
------------------------------------------
death:                    cardiovascular, cancer, trauma, respiratory, neurological, infectious
critical_illness:         cardiovascular, cancer, neurological, renal, respiratory, endocrine_metabolic
total_permanent_disability: trauma, musculoskeletal, neurological, cardiovascular, cancer
income_protection:        musculoskeletal, mental_health, cardiovascular, respiratory,
                          gastrointestinal, neurological, endocrine_metabolic
medical_expense:          all categories
hospital_cash:            all categories
accident:                 trauma

Coherence loading
-----------------
claim_type_match = True:  0%  (diagnosis consistent with claim type)
claim_type_match = False: 25% (mismatch — requires clinical review)
diagnosis not provided:   10% (incomplete medical documentation)

Prognosis risk
--------------
full_recovery / expected_recovery: low
partial_recovery / chronic:        moderate
permanent / progressive:           high
terminal / fatal:                  very_high
"""

from __future__ import annotations

__all__ = ["compute_medical_coherence"]

nan = float("nan")

_DIAGNOSIS_CATEGORY_MAP: dict[str, str] = {
    # Cardiovascular
    "myocardial_infarction":          "cardiovascular",
    "acute_myocardial_infarction":    "cardiovascular",
    "heart_attack":                   "cardiovascular",
    "coronary_artery_disease":        "cardiovascular",
    "cardiac_arrest":                 "cardiovascular",
    "heart_failure":                  "cardiovascular",
    "stroke":                         "cardiovascular",
    "cerebrovascular_accident":       "cardiovascular",
    "cva":                            "cardiovascular",
    "tia":                            "cardiovascular",
    "aortic_aneurysm":                "cardiovascular",
    "peripheral_vascular_disease":    "cardiovascular",
    "atrial_fibrillation":            "cardiovascular",
    "cardiomyopathy":                 "cardiovascular",
    # Cancer
    "cancer":                         "cancer",
    "malignant_neoplasm":             "cancer",
    "carcinoma":                      "cancer",
    "breast_cancer":                  "cancer",
    "lung_cancer":                    "cancer",
    "colorectal_cancer":              "cancer",
    "colon_cancer":                   "cancer",
    "prostate_cancer":                "cancer",
    "ovarian_cancer":                 "cancer",
    "leukaemia":                      "cancer",
    "lymphoma":                       "cancer",
    "melanoma":                       "cancer",
    "pancreatic_cancer":              "cancer",
    # Neurological
    "multiple_sclerosis":             "neurological",
    "parkinsons":                     "neurological",
    "alzheimers":                     "neurological",
    "motor_neurone_disease":          "neurological",
    "epilepsy":                       "neurological",
    "traumatic_brain_injury":         "neurological",
    "tbi":                            "neurological",
    "spinal_cord_injury":             "neurological",
    # Trauma / musculoskeletal
    "fracture":                       "trauma",
    "amputation":                     "trauma",
    "spinal_injury":                  "trauma",
    "polytrauma":                     "trauma",
    "road_traffic_accident":          "trauma",
    "fall_injury":                    "trauma",
    "rheumatoid_arthritis":           "musculoskeletal",
    "osteoarthritis":                 "musculoskeletal",
    "back_pain":                      "musculoskeletal",
    "disc_herniation":                "musculoskeletal",
    "ankylosing_spondylitis":         "musculoskeletal",
    # Respiratory
    "copd":                           "respiratory",
    "pulmonary_fibrosis":             "respiratory",
    "pneumonia":                      "respiratory",
    "respiratory_failure":            "respiratory",
    "asthma":                         "respiratory",
    # Renal
    "chronic_kidney_disease":         "renal",
    "kidney_failure":                 "renal",
    "renal_failure":                  "renal",
    "end_stage_renal_disease":        "renal",
    "polycystic_kidney":              "renal",
    # Mental health
    "depression":                     "mental_health",
    "anxiety":                        "mental_health",
    "bipolar_disorder":               "mental_health",
    "schizophrenia":                  "mental_health",
    "ptsd":                           "mental_health",
    "burnout":                        "mental_health",
    "eating_disorder":                "mental_health",
    # Endocrine / metabolic
    "type1_diabetes":                 "endocrine_metabolic",
    "type2_diabetes":                 "endocrine_metabolic",
    "diabetes":                       "endocrine_metabolic",
    "hypothyroidism":                 "endocrine_metabolic",
    "hyperthyroidism":                "endocrine_metabolic",
    # Gastrointestinal
    "liver_failure":                  "gastrointestinal",
    "cirrhosis":                      "gastrointestinal",
    "crohns_disease":                 "gastrointestinal",
    "ulcerative_colitis":             "gastrointestinal",
    "inflammatory_bowel_disease":     "gastrointestinal",
    # Infectious
    "hiv":                            "infectious",
    "sepsis":                         "infectious",
    "meningitis":                     "infectious",
    # Terminal / multi-system
    "terminal_illness":               "terminal",
    "multi_organ_failure":            "terminal",
}

# ICD-10 chapter → clinical category (by leading letter)
_ICD_CHAPTER_MAP: dict[str, str] = {
    "A": "infectious", "B": "infectious",
    "C": "cancer",
    "D": "blood",
    "E": "endocrine_metabolic",
    "F": "mental_health",
    "G": "neurological",
    "H": "sensory",
    "I": "cardiovascular",
    "J": "respiratory",
    "K": "gastrointestinal",
    "L": "skin",
    "M": "musculoskeletal",
    "N": "renal",
    "O": "obstetric",
    "P": "perinatal",
    "Q": "congenital",
    "R": "symptoms",
    "S": "trauma",
    "T": "trauma",
    "V": "external_causes",
    "W": "external_causes",
    "X": "external_causes",
    "Y": "external_causes",
    "Z": "factors",
}

_CLAIM_TYPE_EXPECTED_CATEGORIES: dict[str, frozenset[str]] = {
    "death": frozenset({
        "cardiovascular", "cancer", "trauma", "respiratory",
        "neurological", "infectious", "terminal", "renal", "gastrointestinal",
    }),
    "critical_illness": frozenset({
        "cardiovascular", "cancer", "neurological", "renal",
        "respiratory", "endocrine_metabolic", "terminal",
    }),
    "total_permanent_disability": frozenset({
        "trauma", "musculoskeletal", "neurological", "cardiovascular", "cancer",
    }),
    "income_protection": frozenset({
        "musculoskeletal", "mental_health", "cardiovascular", "respiratory",
        "gastrointestinal", "neurological", "endocrine_metabolic", "cancer", "renal",
    }),
    "medical_expense":          frozenset(),   # any diagnosis valid
    "hospital_cash":            frozenset(),   # any diagnosis valid
    "accident":                 frozenset({"trauma"}),
}

_PROGNOSIS_RISK_MAP: dict[str, str] = {
    "full_recovery":       "low",
    "expected_recovery":   "low",
    "partial_recovery":    "moderate",
    "chronic":             "moderate",
    "permanent":           "high",
    "progressive":         "high",
    "terminal":            "very_high",
    "fatal":               "very_high",
    "unknown":             "unknown",
}


def _categorise_diagnosis(diagnoses: list[str]) -> list[str]:
    cats: list[str] = []
    for d in diagnoses:
        key = d.lower().strip().replace(" ", "_").replace("-", "_")
        cat = _DIAGNOSIS_CATEGORY_MAP.get(key)
        if cat and cat not in cats:
            cats.append(cat)
    return cats


def _categorise_icd(icd_codes: list[str]) -> list[str]:
    cats: list[str] = []
    for code in icd_codes:
        ch = code.strip().upper()[:1]
        cat = _ICD_CHAPTER_MAP.get(ch)
        if cat and cat not in cats:
            cats.append(cat)
    return cats


def compute_medical_coherence(data: dict) -> dict:
    """
    Assess whether the medical documentation is coherent with the claim type.

    Args:
        data: Validated claim form dict from validate_claim_form().

    Returns:
        Flat dict: diagnosis_categories, icd_chapters, primary_diagnosis_category,
        claim_type_match, prognosis_risk, coherence_loading_pct.
    """
    claim_type = data.get("claim_type", "unknown")
    diagnoses  = data.get("diagnosis", [])
    icd_codes  = data.get("icd_codes", [])
    prognosis  = str(data.get("prognosis", "unknown")).lower().strip().replace(" ", "_")

    diag_cats = _categorise_diagnosis(diagnoses)
    icd_cats  = _categorise_icd(icd_codes)

    # Merge: diagnosis string categories first, then ICD
    all_cats: list[str] = list(diag_cats)
    for c in icd_cats:
        if c not in all_cats:
            all_cats.append(c)

    primary = all_cats[0] if all_cats else "unknown"

    # Claim type match
    expected = _CLAIM_TYPE_EXPECTED_CATEGORIES.get(claim_type, frozenset())
    if not expected:
        # medical_expense / hospital_cash accept any diagnosis
        claim_type_match = bool(diagnoses or icd_codes)
    elif all_cats:
        claim_type_match = bool(set(all_cats) & expected)
    else:
        claim_type_match = False

    # Loading
    if not diagnoses and not icd_codes:
        coherence_load = 10.0  # no medical documentation supplied
    elif not claim_type_match:
        coherence_load = 25.0  # diagnosis does not fit claim type
    else:
        coherence_load = 0.0

    prognosis_risk = _PROGNOSIS_RISK_MAP.get(prognosis, "unknown")

    return {
        "diagnosis_categories":       all_cats,
        "icd_chapters":               icd_cats,
        "primary_diagnosis_category": primary,
        "claim_type_match":           claim_type_match,
        "prognosis_risk":             prognosis_risk,
        "coherence_loading_pct":      coherence_load,
    }
