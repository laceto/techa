"""
techa/indicators — Technical indicator snapshot builder (ta-lib backed).

Primary entry points
--------------------
build_snapshot(ohlcv, *, nan_to_none=False) -> dict
    Last-bar indicator snapshot for one ticker from raw OHLCV history.

build_snapshot_from_parquet(ticker, data_path, *, ticker_col, date_col) -> dict
    Convenience loader for parquet-stored OHLCV data.

build_group_snapshot(data, groups=None, *, include_vol_regime=True) -> GroupSnapshot
    Per-group aggregated snapshot across many tickers.

build_ticker_snapshot(data) -> pd.DataFrame
    Per-ticker last-bar indicator table without group aggregation.

GroupSnapshot
    Frozen dataclass: .tickers (per-ticker table) and .groups (per-group table).

Deprecated aliases (kept for one minor version)
------------------------------------------------
compute_last_bar  → build_snapshot
build_ticker_table → build_ticker_snapshot
"""

from techa.indicators.snapshot import (
    build_snapshot,
    build_snapshot_from_parquet,
    compute_last_bar,
)
from techa.indicators.group_snapshot import (
    GroupSnapshot,
    build_group_snapshot,
    build_ticker_snapshot,
    build_ticker_table,
)

__all__ = [
    "build_snapshot",
    "build_snapshot_from_parquet",
    "GroupSnapshot",
    "build_group_snapshot",
    "build_ticker_snapshot",
    # deprecated
    "compute_last_bar",
    "build_ticker_table",
]
