"""
techa/actuarial/ae/snapshot.py — A/E monitoring snapshot orchestrator.

Public API
----------
build_ae_snapshot(ae_data, *, nan_to_none) -> dict
    Compute actual vs expected experience monitoring KPIs.

Input ae_data (dict)
---------------------
Required:
    periods   — list of period records. Each must contain:
                  actual_claims (or actual_amount)   — observed events/amount.
                  expected_claims (or expected_amount) — modelled baseline.

Optional per period:
    exposed_lives   — policy count exposed during period.
    period          — string label, e.g. "2024-Q1".

Optional top-level:
    risk_type       — "mortality" | "morbidity" | "lapse" | "expense". Default "mortality".
    product_type    — string, e.g. "term_life".
    currency        — string, e.g. "GBP".

Output snapshot keys
--------------------
period_count, actual_total, expected_total,
aggregate_ae_ratio, aggregate_ae_pct,
ae_alert_level (green/amber/red),
credibility_weight, credibility_weighted_ae,
z_score, z_score_significant,
cumulative_deviation,
max_ae_ratio, min_ae_ratio, periods_above_expected,
ae_trend_slope, ae_trend_r2, ae_trend_direction (improving/stable/deteriorating),
ae_volatility,
risk_type, product_type.
"""

from __future__ import annotations

import math

from techa.actuarial.ae._adapter import validate_ae_data
from techa.actuarial.ae.analysis import compute_ae_analysis

__all__ = ["build_ae_snapshot"]


def build_ae_snapshot(ae_data: dict, *, nan_to_none: bool = False) -> dict:
    """
    Compute an A/E experience monitoring snapshot.

    Args:
        ae_data:     Dict — see module docstring for schema.
        nan_to_none: Replace float NaN with None for JSON-serialisable output.

    Returns:
        Flat dict of A/E KPIs (~20 keys).

    Raises:
        ValueError: If required fields are missing or periods list is empty.

    Example:
        from techa.actuarial.ae import build_ae_snapshot

        snap = build_ae_snapshot({
            "risk_type": "mortality",
            "periods": [
                {"period": "2023-Q1", "actual_claims": 45, "expected_claims": 42.0},
                {"period": "2023-Q2", "actual_claims": 48, "expected_claims": 43.5},
                {"period": "2023-Q3", "actual_claims": 52, "expected_claims": 44.0},
                {"period": "2023-Q4", "actual_claims": 55, "expected_claims": 44.5},
            ],
        }, nan_to_none=True)

        print(snap["aggregate_ae_pct"])     # 116.0
        print(snap["ae_alert_level"])       # "amber"
        print(snap["ae_trend_direction"])   # "deteriorating"
    """
    data = validate_ae_data(ae_data)
    result = compute_ae_analysis(data)
    result["risk_type"]    = data.get("risk_type",    "mortality")
    result["product_type"] = data.get("product_type", "unknown")

    if nan_to_none:
        result = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in result.items()
        }
    return result
