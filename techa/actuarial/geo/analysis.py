"""
techa/actuarial/geo/analysis.py — Geospatial/epidemiological risk KPI computation.

Loading schedule
----------------
IMD loading (deprivation)
    decile 1–3  → +15.0 %   (high deprivation)
    decile 4–6  → +5.0 %    (elevated deprivation)
    decile 7–10 → +0.0 %    (low deprivation)

Regional A/E loading
    loading = (regional_ae_index − 1.0) × 100
    clamped to [−20.0, +30.0]

Hospital access loading
    quality_score < 50 → +5.0 %
    quality_score ≥ 50 → +0.0 %

Geo total loading
    sum of the three loadings above, clamped to [0.0, 40.0]

Geo risk level
    total < 5 %         → "low"
    5 % ≤ total ≤ 15 %  → "elevated"
    total > 15 %        → "high"
"""

from __future__ import annotations

__all__ = ["compute_geo_analysis"]


def _imd_risk_band(decile: int) -> str:
    if decile <= 3:
        return "high"
    if decile <= 6:
        return "elevated"
    return "low"


def _imd_loading(decile: int) -> float:
    if decile <= 3:
        return 15.0
    if decile <= 6:
        return 5.0
    return 0.0


def _regional_ae_loading(index: float) -> float:
    raw = (index - 1.0) * 100.0
    return max(-20.0, min(30.0, raw))


def _hospital_access_loading(score: float) -> float:
    return 5.0 if score < 50.0 else 0.0


def _geo_risk_level(total: float) -> str:
    if total < 5.0:
        return "low"
    if total <= 15.0:
        return "elevated"
    return "high"


def compute_geo_analysis(data: dict) -> dict:
    """
    Compute geospatial/epidemiological risk KPIs from validated data.

    Args:
        data: Validated dict from validate_geo_data().

    Returns:
        Flat dict of geo risk metrics (10 keys).
    """
    postcode_area         = data["_postcode_area"]
    imd_decile            = data["_imd_decile"]
    regional_ae_index     = data["_regional_ae_index"]
    hospital_quality_score = data["_hospital_quality_score"]

    imd_risk_band            = _imd_risk_band(imd_decile)
    imd_loading_pct          = _imd_loading(imd_decile)
    regional_ae_loading_pct  = _regional_ae_loading(regional_ae_index)
    hospital_access_loading_pct = _hospital_access_loading(hospital_quality_score)

    geo_total_raw = imd_loading_pct + regional_ae_loading_pct + hospital_access_loading_pct
    geo_total_loading_pct = max(0.0, min(40.0, geo_total_raw))

    return {
        "postcode_area":               postcode_area,
        "imd_decile":                  imd_decile,
        "imd_risk_band":               imd_risk_band,
        "imd_loading_pct":             imd_loading_pct,
        "regional_ae_index":           round(regional_ae_index, 4),
        "regional_ae_loading_pct":     round(regional_ae_loading_pct, 4),
        "hospital_quality_score":      round(hospital_quality_score, 2),
        "hospital_access_loading_pct": hospital_access_loading_pct,
        "geo_total_loading_pct":       round(geo_total_loading_pct, 4),
        "geo_risk_level":              _geo_risk_level(geo_total_loading_pct),
    }
