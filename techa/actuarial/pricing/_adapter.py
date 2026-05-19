"""
techa/actuarial/pricing/_adapter.py — Reinsurance pricing data validation.

Required fields: cash_flows (list, each with ceded_premium and ceded_claims).
Optional per-year fields: expenses, commission, profit_commission.
Optional top-level: treaty_type, discount_rate, term_years, allocated_capital.
"""

from __future__ import annotations

import math

__all__ = ["validate_pricing_data"]

_REQUIRED_TOP = frozenset({"cash_flows"})


def _to_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if not math.isnan(v) else default
    except (TypeError, ValueError):
        return default


def validate_pricing_data(data: dict) -> dict:
    """
    Validate and normalise a reinsurance pricing dataset.

    Args:
        data: Dict with "cash_flows" list. Each element must supply at least
              ceded_premium and ceded_claims per treaty year.

    Returns:
        Enriched dict with "_flows" list of validated yearly dicts.

    Raises:
        ValueError: If required fields are missing or cash_flows list is empty.
    """
    missing = _REQUIRED_TOP - set(data)
    if missing:
        raise ValueError(f"Missing required pricing fields: {missing}")

    raw_flows = data.get("cash_flows", [])
    if not raw_flows:
        raise ValueError("Pricing 'cash_flows' list must not be empty.")

    flows: list[dict] = []
    for i, f in enumerate(raw_flows):
        flows.append({
            "year":              i + 1,
            "ceded_premium":     _to_float(f.get("ceded_premium")),
            "ceded_claims":      _to_float(f.get("ceded_claims")),
            "expenses":          _to_float(f.get("expenses")),
            "commission":        _to_float(f.get("commission",
                                          f.get("commission_paid"))),
            "profit_commission": _to_float(f.get("profit_commission")),
        })

    out = dict(data)
    out["_flows"]          = flows
    out["_discount_rate"]  = _to_float(data.get("discount_rate"), default=0.05)
    out["_allocated_capital"] = _to_float(data.get("allocated_capital"), default=0.0)
    out.setdefault("treaty_type", "unknown")
    out.setdefault("term_years",  len(flows))
    return out
