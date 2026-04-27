"""
techa/indicators/grouping.py — Group assignment for the indicator snapshot pipeline.

Responsibility: pure data shaping — no aggregation math lives here.

Public API
----------
assign_vol_regime(ticker_df) -> pd.DataFrame
    Add vol_regime column: "low_vol" / "mid_vol" / "high_vol" via ATR% terciles.

assign_user_groups(ticker_df, groups) -> pd.DataFrame
    Add user_groups column (list[str]) from a caller-supplied ticker→label mapping.
    Stores a list so downstream can explode directly — no comma-join encoding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["assign_vol_regime", "assign_user_groups"]


def assign_vol_regime(ticker_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add vol_regime column based on ATR% tercile rank across all tickers.

    Regimes: "low_vol" (bottom third), "mid_vol" (middle), "high_vol" (top third).
    Tickers with NaN atr_pct receive "unknown".

    Args:
        ticker_df: Per-ticker indicator DataFrame (index = ticker symbol).

    Returns:
        Copy of ticker_df with vol_regime column added.
    """
    df = ticker_df.copy()

    if "atr_pct" not in df.columns or df["atr_pct"].isna().all():
        df["vol_regime"] = "unknown"
        return df

    atr_valid = df["atr_pct"].dropna()
    low_cut  = atr_valid.quantile(1 / 3)
    high_cut = atr_valid.quantile(2 / 3)

    def _regime(v: float) -> str:
        if np.isnan(v):
            return "unknown"
        if v <= low_cut:
            return "low_vol"
        if v <= high_cut:
            return "mid_vol"
        return "high_vol"

    df["vol_regime"] = df["atr_pct"].map(_regime)
    return df


def assign_user_groups(
    ticker_df: pd.DataFrame,
    groups: dict[str, str | list[str]],
) -> pd.DataFrame:
    """
    Add user_groups column (list[str]) from a ticker→group mapping.

    Tickers absent from groups receive ["ungrouped"].
    Multi-group membership is stored as a list — not comma-joined — so
    aggregation.build_group_df can explode it directly.

    Args:
        ticker_df: Per-ticker indicator DataFrame (index = ticker symbol).
        groups:    Dict mapping ticker → group label or list of labels.
                   Example: {"AAPL": "tech", "AMZN": ["tech", "consumer"]}

    Returns:
        Copy of ticker_df with user_groups column added.
    """
    df = ticker_df.copy()

    def _normalise(v: str | list[str]) -> list[str]:
        return v if isinstance(v, list) else [str(v)]

    df["user_groups"] = [
        _normalise(groups[t]) if t in groups else ["ungrouped"]
        for t in df.index
    ]
    return df
