"""
techa/insurance/snapshot.py — Insurance accountant KPI snapshot builder.

Responsibility: thin orchestrator.
    - Input validation and MIN_PERIODS enforcement.
    - Delegates all KPI computation to domain modules (profitability, reserves, growth)
      via to_numpy_financials() from _adapter.
    - nan_to_none flag for JSON-serialisable output.

Public API
----------
build_kpi_snapshot(df, *, nan_to_none, periods_per_year, trend_lookback) -> dict
    Compute a last-period accountant KPI snapshot from an insurance financials time series.

Expected input DataFrame
------------------------
Index:   datetime (accounting period end date, sorted ascending — monthly or quarterly).
Required columns:
    gwp             — Gross Written Premium (£) for the period.
    claims_incurred — Net claims incurred in the period (£).
    expenses        — Operating expenses (management + commission) in the period (£).
Optional columns (return NaN KPIs when absent):
    nwp                — Net Written Premium after reinsurance cessions (£).
    claims_paid        — Cash claims settled in the period (£).
    reserve_held       — Closing reserve balance (£).
    reserve_required   — Actuarially required reserve at period end (£).
    policies_in_force  — Count of in-force policies at period end.
    new_policies       — New policies written in the period.
    lapsed_policies    — Policies cancelled or lapsed in the period.

Output snapshot keys
--------------------
Profitability:
    loss_ratio, expense_ratio, combined_ratio, underwriting_margin_pct,
    underwriting_profit, reinsurance_cession_pct, net_claims_ratio,
    loss_ratio_trend, loss_ratio_trend_r2,
    expense_ratio_trend, expense_ratio_trend_r2,
    combined_ratio_trend, combined_ratio_trend_r2.

Reserves:
    reserve_adequacy_ratio, reserve_adequacy_pct, reserve_surplus,
    reserve_to_gwp_pct, claims_settlement_ratio,
    claims_outstanding, claims_outstanding_ratio,
    reserve_adequacy_trend, reserve_adequacy_trend_r2.

Growth:
    gwp_latest, nwp_latest,
    premium_growth_pp, claims_growth_pp,
    premium_growth_yoy, claims_growth_yoy,
    gwp_cagr, gwp_trend, gwp_trend_r2,
    avg_premium, lapse_rate, new_business_ratio.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from techa.insurance._adapter import to_numpy_financials, MIN_PERIODS
from techa.insurance.profitability import compute_profitability
from techa.insurance.reserves import compute_reserves
from techa.insurance.growth import compute_growth

__all__ = ["build_kpi_snapshot"]

log = logging.getLogger(__name__)


def build_kpi_snapshot(
    df: pd.DataFrame,
    *,
    nan_to_none: bool = False,
    periods_per_year: int = 4,
    trend_lookback: int = 8,
) -> dict:
    """
    Compute a last-period accountant KPI snapshot from an insurance financials time series.

    Args:
        df:               DataFrame with datetime index (period end dates, ascending).
                          Required columns: gwp, claims_incurred, expenses.
                          Optional: nwp, claims_paid, reserve_held, reserve_required,
                          policies_in_force, new_policies, lapsed_policies.
        nan_to_none:      Replace float NaN with None for JSON-serialisable output.
                          Default False.
        periods_per_year: Accounting frequency for YoY comparisons and CAGR.
                          4 = quarterly (default), 12 = monthly.
        trend_lookback:   Number of periods for OLS trend fits. Default 8.

    Returns:
        Flat dict of scalars (float, None when nan_to_none=True).
        Keys documented in snapshot.py module docstring.

    Raises:
        ValueError: Missing required columns, or fewer than MIN_PERIODS rows.

    Example:
        import pandas as pd
        from techa.insurance import build_kpi_snapshot

        df = pd.DataFrame({
            "gwp":             [3000, 3100, 3200, 3400, 3350, 3500, 3600, 3800],
            "claims_incurred": [1700, 1800, 1750, 2000, 1900, 2100, 2000, 2300],
            "expenses":        [ 850,  880,  900,  950,  930,  980, 1000, 1050],
            "nwp":             [2700, 2800, 2900, 3100, 3000, 3200, 3300, 3500],
            "reserve_held":    [15000, 15200, 15500, 15800, 15600, 16000, 16200, 16800],
            "reserve_required":[14000, 14300, 14600, 15000, 14900, 15300, 15500, 16000],
            "claims_paid":     [1600, 1700, 1650, 1900, 1800, 2000, 1900, 2200],
        }, index=pd.date_range("2022-03-31", periods=8, freq="QE"))

        snap = build_kpi_snapshot(df, nan_to_none=True)
        print(snap["combined_ratio"])     # 0.863...
        print(snap["loss_ratio_trend"])   # positive = deteriorating
        print(snap["reserve_adequacy_pct"])  # > 100 = over-reserved
    """
    if len(df) < MIN_PERIODS:
        raise ValueError(
            f"build_kpi_snapshot requires at least {MIN_PERIODS} periods; got {len(df)}. "
            "Provide a longer financial history or lower MIN_PERIODS in _adapter.py."
        )

    arrays = to_numpy_financials(df)

    result: dict = {}
    result.update(compute_profitability(arrays, trend_lookback=trend_lookback))
    result.update(compute_reserves(arrays, trend_lookback=trend_lookback))
    result.update(compute_growth(
        arrays,
        periods_per_year=periods_per_year,
        trend_lookback=trend_lookback,
    ))

    log.debug(
        "build_kpi_snapshot: %d periods, %d KPI keys computed",
        len(df),
        len(result),
    )

    if nan_to_none:
        result = {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in result.items()
        }

    return result
