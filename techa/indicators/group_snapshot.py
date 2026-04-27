"""
techa/indicators/group_snapshot.py — Multi-ticker group snapshot orchestrator.

Responsibility: thin orchestrator only.
    - Normalise input (dict or MultiIndex DataFrame) to {ticker: ohlcv_df}
    - Call build_snapshot per ticker, skip failures with a warning
    - Delegate group assignment to grouping module
    - Delegate aggregation to aggregation module
    - Return GroupSnapshot dataclass

Public API
----------
GroupSnapshot
    Frozen dataclass with .tickers (per-ticker table) and .groups (per-group table).

build_group_snapshot(data, groups=None, *, include_vol_regime=True) -> GroupSnapshot
    Full pipeline: per-ticker indicators → group assignment → aggregation.

build_ticker_snapshot(data) -> pd.DataFrame
    Per-ticker step only. Useful when group aggregation is not needed.

build_ticker_table(data) -> pd.DataFrame
    Deprecated alias for build_ticker_snapshot.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Union

import pandas as pd

from techa.indicators._adapter import MIN_BARS
from techa.indicators.aggregation import build_group_df
from techa.indicators.grouping import assign_user_groups, assign_vol_regime
from techa.indicators.snapshot import build_snapshot

__all__ = [
    "GroupSnapshot",
    "build_group_snapshot",
    "build_ticker_snapshot",
    "build_ticker_table",
]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupSnapshot:
    """
    Result of build_group_snapshot.

    Attributes:
        tickers: Per-ticker last-bar indicator table. Index = ticker symbol.
                 Includes vol_regime and user_groups columns when assigned.
        groups:  Per-(group_label, grouping_type) aggregated snapshot.
                 Index = (group_label, grouping_type).
    """

    tickers: pd.DataFrame
    groups: pd.DataFrame


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

def _to_ticker_dict(
    data: Union[dict[str, pd.DataFrame], pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    if isinstance(data, dict):
        return data

    if not isinstance(data.index, pd.MultiIndex):
        raise ValueError(
            "DataFrame input must have a MultiIndex of (ticker, date) or (date, ticker). "
            "Use a dict[str, pd.DataFrame] instead for non-MultiIndex data."
        )

    names = list(data.index.names)
    if names[0].lower() in ("ticker", "symbol"):
        ticker_level = 0
    elif len(names) > 1 and names[1].lower() in ("ticker", "symbol"):
        ticker_level = 1
    else:
        ticker_level = 0
        log.warning(
            "Cannot infer ticker/date levels from MultiIndex names %s; assuming level 0 = ticker.",
            names,
        )

    result: dict[str, pd.DataFrame] = {}
    for ticker in data.index.get_level_values(ticker_level).unique():
        sub = data.xs(ticker, level=ticker_level)
        if not sub.index.is_monotonic_increasing:
            sub = sub.sort_index()
        result[str(ticker)] = sub

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ticker_snapshot(
    data: Union[dict[str, pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    """
    Compute last-bar technical indicators for every ticker.

    Tickers with fewer than MIN_BARS bars are skipped with a warning.
    Individual failures are logged and skipped so one bad ticker never aborts a scan.

    Args:
        data: Dict[str, pd.DataFrame] mapping ticker → OHLCV DataFrame, OR
              MultiIndex DataFrame indexed by (ticker, date) or (date, ticker).

    Returns:
        DataFrame indexed by ticker with all last-bar indicator columns.
        Empty DataFrame if no tickers could be processed.
    """
    ticker_dict = _to_ticker_dict(data)
    rows: list[dict] = []

    for ticker, ohlcv in ticker_dict.items():
        if len(ohlcv) < MIN_BARS:
            log.warning(
                "Skipping %s: fewer than %d bars (%d available).", ticker, MIN_BARS, len(ohlcv)
            )
            continue
        try:
            row = build_snapshot(ohlcv)
            row["ticker"] = ticker
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            log.warning("build_snapshot failed for %s: %s", ticker, exc)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("ticker")


def build_group_snapshot(
    data: Union[dict[str, pd.DataFrame], pd.DataFrame],
    groups: dict[str, str | list[str]] | None = None,
    *,
    include_vol_regime: bool = True,
) -> GroupSnapshot:
    """
    Compute per-ticker last-bar indicators and aggregate them into group summaries.

    Args:
        data:               Dict[str, pd.DataFrame] or MultiIndex DataFrame.
                            OHLCV columns: open/high/low/close/volume (case-insensitive).
        groups:             Optional dict mapping ticker → group label or list of labels.
                            Example: {"AAPL": "tech", "AMZN": ["tech", "consumer"]}
                            If None, only vol_regime grouping is produced (when enabled).
        include_vol_regime: Include automatic ATR%-tercile vol_regime grouping. Default True.

    Returns:
        GroupSnapshot with:
          .tickers — per-ticker last-bar indicator table with group columns.
          .groups  — per-(group_label, grouping_type) aggregated snapshot.
                     grouping_type values: "vol_regime" and/or "user".
    """
    ticker_df = build_ticker_snapshot(data)
    if ticker_df.empty:
        return GroupSnapshot(tickers=ticker_df, groups=pd.DataFrame())

    group_frames: list[pd.DataFrame] = []

    if include_vol_regime:
        ticker_df = assign_vol_regime(ticker_df)
        group_frames.append(build_group_df(ticker_df, "vol_regime", "vol_regime"))

    if groups:
        ticker_df = assign_user_groups(ticker_df, groups)
        group_frames.append(
            build_group_df(ticker_df, "user_groups", "user", list_col=True)
        )

    if group_frames:
        group_df = pd.concat(group_frames)
        metric_cols = sorted(c for c in group_df.columns if c not in ("n_tickers", "bull_score"))
        ordered = ["n_tickers"] + metric_cols + ["bull_score"]
        group_df = group_df[[c for c in ordered if c in group_df.columns]]
    else:
        group_df = pd.DataFrame()

    return GroupSnapshot(tickers=ticker_df, groups=group_df)


def build_ticker_table(
    data: Union[dict[str, pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    """Deprecated. Use build_ticker_snapshot()."""
    warnings.warn(
        "build_ticker_table is deprecated; use build_ticker_snapshot().",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_ticker_snapshot(data)
