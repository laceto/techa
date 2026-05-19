"""
techa/actuarial/geo/snapshot.py — Geospatial/epidemiological risk snapshot orchestrator.

Public API
----------
build_geo_snapshot(geo_data, *, nan_to_none) -> dict
    Enrich an insurance applicant's profile with regional/epidemiological indices.

Input geo_data (dict)
----------------------
All fields are optional; defaults are shown in brackets.

    postcode_area          — str.   Informational postcode area, e.g. "SW1". [""]
    imd_decile             — int.   Index of Multiple Deprivation decile, 1–10.
                                    1 = most deprived, 10 = least deprived. [5]
    regional_ae_index      — float. Local mortality vs national average (1.0 = average).
                                    Source: ONS / CMI. Clamped to [0.5, 2.0]. [1.0]
    hospital_quality_score — float. 0–100 quality score. Clamped to [0, 100]. [75]

Output snapshot keys (10 keys)
-------------------------------
    postcode_area               — str,   pass-through
    imd_decile                  — int,   1–10
    imd_risk_band               — str,   "high" | "elevated" | "low"
    imd_loading_pct             — float, 15.0 | 5.0 | 0.0
    regional_ae_index           — float
    regional_ae_loading_pct     — float, (index − 1.0) × 100, clamped to [−20.0, +30.0]
    hospital_quality_score      — float
    hospital_access_loading_pct — float, 5.0 if score < 50 else 0.0
    geo_total_loading_pct       — float, sum of loadings, clamped to [0.0, 40.0]
    geo_risk_level              — str,   "low" | "elevated" | "high"
"""

from __future__ import annotations

import math

from techa.actuarial.geo._adapter import validate_geo_data
from techa.actuarial.geo.analysis import compute_geo_analysis

__all__ = ["build_geo_snapshot"]


def build_geo_snapshot(geo_data: dict, *, nan_to_none: bool = False) -> dict:
    """
    Compute a geospatial/epidemiological risk snapshot.

    Args:
        geo_data:    Dict — see module docstring for schema. All fields optional.
        nan_to_none: Replace float NaN with None for JSON-serialisable output.

    Returns:
        Flat dict of geo risk KPIs (10 keys).

    Example:
        from techa.actuarial.geo import build_geo_snapshot

        snap = build_geo_snapshot({
            "postcode_area": "E1",
            "imd_decile": 2,
            "regional_ae_index": 1.12,
            "hospital_quality_score": 45,
        }, nan_to_none=True)

        print(snap["imd_loading_pct"])             # 15.0
        print(snap["regional_ae_loading_pct"])     # 12.0
        print(snap["hospital_access_loading_pct"]) # 5.0
        print(snap["geo_total_loading_pct"])       # 32.0
        print(snap["geo_risk_level"])              # "high"
    """
    data = validate_geo_data(geo_data)
    result = compute_geo_analysis(data)

    if nan_to_none:
        result = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in result.items()
        }
    return result
