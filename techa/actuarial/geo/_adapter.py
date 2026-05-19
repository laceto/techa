"""
techa/actuarial/geo/_adapter.py — Geospatial/epidemiological risk data validation.

All fields are optional; defaults are applied when absent.
Numeric fields are clamped to safe ranges before analysis.
"""

from __future__ import annotations

__all__ = ["validate_geo_data"]


def _to_float(val, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _to_int(val, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def validate_geo_data(data: dict) -> dict:
    """
    Validate and normalise a geospatial/epidemiological risk dataset.

    All fields are optional.  Missing values are replaced with safe defaults.
    Numeric fields are clamped to defined safe ranges.

    Args:
        data: Dict with optional keys: postcode_area, imd_decile,
              regional_ae_index, hospital_quality_score.

    Returns:
        Enriched dict with validated and clamped scalar fields.
    """
    postcode_area = str(data.get("postcode_area", ""))

    # IMD decile: 1 (most deprived) – 10 (least deprived)
    imd_decile_raw = _to_int(data.get("imd_decile", 5), default=5)
    imd_decile = max(1, min(10, imd_decile_raw))

    # Regional A/E index: local mortality vs national average (1.0 = national)
    rae_raw = _to_float(data.get("regional_ae_index", 1.0), default=1.0)
    regional_ae_index = max(0.5, min(2.0, rae_raw))

    # Hospital quality score: 0–100
    hqs_raw = _to_float(data.get("hospital_quality_score", 75.0), default=75.0)
    hospital_quality_score = max(0.0, min(100.0, hqs_raw))

    out = dict(data)
    out["_postcode_area"]         = postcode_area
    out["_imd_decile"]            = imd_decile
    out["_regional_ae_index"]     = regional_ae_index
    out["_hospital_quality_score"] = hospital_quality_score
    return out
