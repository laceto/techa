"""
techa/indicators/aggregation.py — Group aggregation for the indicator snapshot pipeline.

Responsibility: pure aggregation math — no group assignment lives here.

Public API
----------
aggregate_group(group_df) -> dict
    Aggregate a subset of the ticker indicator table into group-level metrics.

build_group_df(ticker_df, group_col, grouping_type, *, list_col=False) -> pd.DataFrame
    Aggregate ticker_df by group_col and return indexed group DataFrame.

Constants
---------
_BULL_SCORE_WEIGHTS — weight map for the composite bull_score.
_MEAN_COLS          — numeric indicator columns included in group means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["aggregate_group", "build_group_df"]

_BULL_SCORE_WEIGHTS: dict[str, float] = {
    "pct_above_sma50":  0.30,
    "pct_above_sma200": 0.25,
    "pct_macd_bullish": 0.25,
    "pct_rsi_neutral":  0.20,
}

_MEAN_COLS = [
    "rsi", "adx", "di_plus", "di_minus",
    "atr_pct", "bb_width", "bb_pct_b", "hist_vol_20d",
    "dist_sma20_pct", "dist_sma50_pct", "dist_sma200_pct",
    "macd_hist", "stoch_k", "stoch_d", "stoch_fast_k", "stoch_fast_d",
    "roc_10d", "roc_20d", "chg_1d", "chg_5d",
    "vol_vs_ma20", "ad", "adosc",
]

# Signal percentages derived from float columns (booleans removed from snapshot schema)
_SIGNAL_COMPUTATIONS: dict[str, callable] = {
    "pct_above_sma20":    lambda df: df["dist_sma20_pct"] > 0,
    "pct_above_sma50":    lambda df: df["dist_sma50_pct"] > 0,
    "pct_above_sma200":   lambda df: df["dist_sma200_pct"] > 0,
    "pct_golden_cross":   lambda df: df["golden_cross"].astype(bool),
    "pct_rsi_overbought": lambda df: df["rsi"] > 70,
    "pct_rsi_oversold":   lambda df: df["rsi"] < 30,
    "pct_macd_bullish":   lambda df: df["macd_hist"] > 0,
    "pct_trending":       lambda df: df["adx"] > 25,
    "pct_high_volume":    lambda df: df["vol_vs_ma20"] > 1.5,
}


def aggregate_group(group_df: pd.DataFrame) -> dict:
    """
    Aggregate a subset of the ticker indicator table into group-level metrics.

    Computes:
    - n_tickers
    - mean_{col} for each numeric column in _MEAN_COLS
    - pct_{signal} for each signal in _SIGNAL_COMPUTATIONS (% of tickers, 0-100)
    - pct_rsi_neutral (RSI in [40, 60]) — used in bull_score
    - bull_score: weighted composite of bullish signals (0-100)

    Args:
        group_df: Subset of the ticker indicator DataFrame for one group.

    Returns:
        Flat dict of group-level scalars.
    """
    result: dict = {"n_tickers": len(group_df)}

    for col in _MEAN_COLS:
        if col in group_df.columns:
            result[f"mean_{col}"] = float(group_df[col].mean(skipna=True))

    signal_pcts: dict[str, float] = {}
    for out_col, fn in _SIGNAL_COMPUTATIONS.items():
        try:
            mask = fn(group_df)
            pct = float(mask.mean(skipna=True) * 100)
            result[out_col] = pct
            signal_pcts[out_col] = pct
        except (KeyError, TypeError):
            pass

    pct_rsi_neutral = (
        float(group_df["rsi"].between(40, 60).mean() * 100)
        if "rsi" in group_df.columns
        else float("nan")
    )

    components = {
        "pct_above_sma50":  signal_pcts.get("pct_above_sma50",  float("nan")),
        "pct_above_sma200": signal_pcts.get("pct_above_sma200", float("nan")),
        "pct_macd_bullish": signal_pcts.get("pct_macd_bullish", float("nan")),
        "pct_rsi_neutral":  pct_rsi_neutral,
    }
    weighted_sum = sum(
        _BULL_SCORE_WEIGHTS[k] * v
        for k, v in components.items()
        if not np.isnan(v)
    )
    weight_total = sum(
        _BULL_SCORE_WEIGHTS[k]
        for k, v in components.items()
        if not np.isnan(v)
    )
    result["bull_score"] = weighted_sum / weight_total if weight_total > 0 else float("nan")

    return result


def build_group_df(
    ticker_df: pd.DataFrame,
    group_col: str,
    grouping_type: str,
    *,
    list_col: bool = False,
) -> pd.DataFrame:
    """
    Aggregate ticker_df by group_col and return a group-indexed DataFrame.

    Args:
        ticker_df:    Per-ticker indicator table with group_col present.
        group_col:    Column name holding the group label (str or list[str]).
        grouping_type: String label for the grouping dimension (e.g. "vol_regime", "user").
        list_col:     True when group_col holds list[str] values (user_groups).
                      Explodes the column before grouping; no string-split needed.

    Returns:
        DataFrame indexed by (group_label, grouping_type).
    """
    df = ticker_df.copy()

    if list_col:
        df = df.explode(group_col)
        df[group_col] = df[group_col].str.strip()

    rows: list[dict] = []
    for group_label, sub in df.groupby(group_col):
        row = aggregate_group(sub)
        row["group_label"] = str(group_label)
        row["grouping_type"] = grouping_type
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index(["group_label", "grouping_type"])
