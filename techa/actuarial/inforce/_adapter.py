"""
techa/actuarial/inforce/_adapter.py — In-force portfolio data validation.

Required fields: periods (list, each with policies_in_force and
                 gross_premium_income at minimum).
Optional per-period: new_policies, lapses, deaths, maturities,
                     best_estimate_liability, risk_margin,
                     solvency_capital_requirement, own_funds.
Optional top-level: periods_per_year, product_type.
"""

from __future__ import annotations

import math

__all__ = ["validate_inforce_data"]

_REQUIRED_TOP = frozenset({"periods"})


def _to_float(val, default: float = float("nan")) -> float:
    try:
        v = float(val)
        return v if not math.isnan(v) else default
    except (TypeError, ValueError):
        return default


def validate_inforce_data(data: dict) -> dict:
    """
    Validate and normalise an in-force portfolio dataset.

    Args:
        data: Dict with "periods" list. Each element must supply
              policies_in_force and gross_premium_income.

    Returns:
        Enriched dict with "_periods" list of validated period dicts.

    Raises:
        ValueError: If required fields are missing or periods list is empty.
    """
    missing = _REQUIRED_TOP - set(data)
    if missing:
        raise ValueError(f"Missing required in-force fields: {missing}")

    raw = data.get("periods", [])
    if not raw:
        raise ValueError("In-force 'periods' list must not be empty.")

    nan = float("nan")
    periods: list[dict] = []
    for i, p in enumerate(raw):
        periods.append({
            "label":      str(p.get("period", f"P{i+1}")),
            "pif":        _to_float(p.get("policies_in_force"),    nan),
            "new_pol":    _to_float(p.get("new_policies"),          nan),
            "lapses":     _to_float(p.get("lapses"),                nan),
            "deaths":     _to_float(p.get("deaths"),                nan),
            "maturities": _to_float(p.get("maturities"),            nan),
            "gwp":        _to_float(p.get("gross_premium_income"),   nan),
            "bel":        _to_float(p.get("best_estimate_liability", p.get("bel")), nan),
            "risk_margin":_to_float(p.get("risk_margin"),            nan),
            "scr":        _to_float(p.get("solvency_capital_requirement", p.get("scr")), nan),
            "own_funds":  _to_float(p.get("own_funds"),              nan),
        })

    out = dict(data)
    out["_periods"]         = periods
    out["_periods_per_year"] = int(data.get("periods_per_year", 4))
    out.setdefault("product_type", "unknown")
    return out
