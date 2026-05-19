"""
techa/actuarial/ae/_adapter.py — A/E monitoring data validation and enrichment.

Required fields: periods (list, each with actual_claims or actual_amount
                 AND expected_claims or expected_amount).
Optional period fields: exposed_lives, period (label string).
Optional top-level: risk_type, product_type, currency.
"""

from __future__ import annotations

__all__ = ["validate_ae_data"]

_REQUIRED_TOP = frozenset({"periods"})


def _to_float(val, default: float = float("nan")) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def validate_ae_data(data: dict) -> dict:
    """
    Validate and normalise an A/E monitoring dataset.

    Args:
        data: Dict with "periods" list. Each period must supply actual and
              expected values (claim counts or amounts).

    Returns:
        Enriched dict with "_periods" list of validated period dicts.

    Raises:
        ValueError: If required fields are missing or periods list is empty.
    """
    missing = _REQUIRED_TOP - set(data)
    if missing:
        raise ValueError(f"Missing required A/E fields: {missing}")

    raw_periods = data.get("periods", [])
    if not raw_periods:
        raise ValueError("A/E 'periods' list must not be empty.")

    periods: list[dict] = []
    for i, p in enumerate(raw_periods):
        actual   = _to_float(p.get("actual_claims",   p.get("actual_amount")))
        expected = _to_float(p.get("expected_claims",  p.get("expected_amount")))
        exposed  = _to_float(p.get("exposed_lives", float("nan")))
        label    = str(p.get("period", f"P{i+1}"))

        # Accept amount-based A/E if count-based not provided
        actual_amt   = _to_float(p.get("actual_amount",   float("nan")))
        expected_amt = _to_float(p.get("expected_amount", float("nan")))

        periods.append({
            "label":        label,
            "actual":       actual,
            "expected":     expected,
            "exposed":      exposed,
            "actual_amt":   actual_amt,
            "expected_amt": expected_amt,
        })

    out = dict(data)
    out["_periods"] = periods
    out.setdefault("risk_type",    "mortality")
    out.setdefault("product_type", "unknown")
    out.setdefault("currency",     "GBP")
    return out
