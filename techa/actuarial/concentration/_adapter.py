"""
techa/actuarial/concentration/_adapter.py — Portfolio concentration data validation.

Required fields:
    portfolio_total_sa      — total sum assured of the in-force book (£), must be > 0.
    portfolio_gwp           — gross written premium of the book (£).
    portfolio_policy_count  — number of in-force policies, must be > 0.
    policy_sum_assured      — this policy's sum assured (£), must be > 0.

Optional fields:
    policy_premium_annual   — this policy's annual premium (£). Defaults to 0.0.
"""

from __future__ import annotations

__all__ = ["validate_concentration_data"]


def _to_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _to_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def validate_concentration_data(data: dict) -> dict:
    """
    Validate and normalise a portfolio concentration context dict.

    Args:
        data: Dict with portfolio and policy fields. See module docstring for schema.

    Returns:
        Enriched dict with normalised numeric fields.

    Raises:
        ValueError: If required fields are missing, zero, or negative.
    """
    portfolio_total_sa = _to_float(data.get("portfolio_total_sa"), default=0.0)
    if portfolio_total_sa <= 0:
        raise ValueError(
            f"'portfolio_total_sa' must be > 0; got {portfolio_total_sa!r}."
        )

    portfolio_policy_count = _to_int(data.get("portfolio_policy_count"), default=0)
    if portfolio_policy_count <= 0:
        raise ValueError(
            f"'portfolio_policy_count' must be > 0; got {portfolio_policy_count!r}."
        )

    policy_sum_assured = _to_float(data.get("policy_sum_assured"), default=0.0)
    if policy_sum_assured <= 0:
        raise ValueError(
            f"'policy_sum_assured' must be > 0; got {policy_sum_assured!r}."
        )

    out = dict(data)
    out["portfolio_total_sa"]     = portfolio_total_sa
    out["portfolio_gwp"]          = _to_float(data.get("portfolio_gwp"), default=0.0)
    out["portfolio_policy_count"] = portfolio_policy_count
    out["policy_sum_assured"]     = policy_sum_assured
    out["policy_premium_annual"]  = _to_float(
        data.get("policy_premium_annual"), default=0.0
    )
    return out
