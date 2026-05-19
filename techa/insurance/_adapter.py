"""
techa/insurance/_adapter.py — DataFrame-to-numpy conversion for insurance KPI inputs.

All domain modules (profitability, reserves, growth) receive dict[str, np.ndarray]
from to_numpy_financials(). This module is the single conversion point; no other
file in techa.insurance should call .to_numpy() or .values on financial data.

Public API
----------
to_numpy_financials(df)  — validate columns, return dict of float64 arrays.
last_valid(arr)          — last non-NaN element of a period array.
nan_div(a, b)            — NaN-safe scalar division.
MIN_PERIODS              — hard minimum periods required by build_kpi_snapshot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["to_numpy_financials", "last_valid", "nan_div", "MIN_PERIODS"]

MIN_PERIODS = 4  # minimum accounting periods for a meaningful snapshot

_REQUIRED_COLS: frozenset[str] = frozenset({"gwp", "claims_incurred", "expenses"})

_OPTIONAL_COLS: frozenset[str] = frozenset({
    "nwp",                  # net written premium (after reinsurance cessions)
    "claims_paid",          # cash claims paid in the period
    "reserve_held",         # closing reserve balance
    "reserve_required",     # actuarially required reserve
    "policies_in_force",    # count of in-force policies at period end
    "new_policies",         # new policies written in the period
    "lapsed_policies",      # policies lapsed or cancelled in the period
})


def to_numpy_financials(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Validate and convert an insurance financials DataFrame to float64 arrays.

    The DataFrame index must be a datetime (accounting period end dates),
    sorted ascending. Required columns: gwp, claims_incurred, expenses.
    Optional columns are returned as all-NaN arrays when absent.

    Args:
        df: DataFrame indexed by period date, one row per accounting period.

    Returns:
        Dict mapping column name → float64 ndarray of length len(df).

    Raises:
        ValueError: If any required column is missing.
    """
    frame = df.copy()
    frame.columns = frame.columns.str.lower()

    missing = _REQUIRED_COLS - set(frame.columns)
    if missing:
        raise ValueError(
            f"build_kpi_snapshot: missing required columns: {sorted(missing)}. "
            f"Got: {sorted(frame.columns)}."
        )

    n = len(frame)
    arrays: dict[str, np.ndarray] = {}

    for col in _REQUIRED_COLS | _OPTIONAL_COLS:
        if col in frame.columns:
            arrays[col] = frame[col].to_numpy(dtype=np.float64)
        else:
            arrays[col] = np.full(n, np.nan)

    return arrays


def last_valid(arr: np.ndarray) -> float:
    """
    Return the last non-NaN element of a period array.

    Args:
        arr: float64 ndarray of per-period values.

    Returns:
        Last non-NaN value as float, or float("nan") if all NaN or empty.
    """
    if arr is None or len(arr) == 0:
        return float("nan")
    valid = arr[~np.isnan(arr)]
    return float(valid[-1]) if len(valid) > 0 else float("nan")


def nan_div(a: float, b: float) -> float:
    """
    NaN-safe scalar division.

    Returns float("nan") when b is zero, NaN, or when a is NaN.
    """
    if np.isnan(a) or np.isnan(b) or b == 0.0:
        return float("nan")
    return float(a / b)
