"""
agents/_common.py — Shared constants and low-level helpers for all agent subpackages.

Imported by:
  techa.agents.ta._tools.prepare_tools
  techa.agents.patterns._tools.prepare_tools
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_PATH: Path = Path("data/results/it/analysis_results.parquet")
HISTORY_BARS: int  = 300   # max rows per ticker kept from parquet; enough for ADX(14)/MA(150)/RSI(14)


def _read_parquet_dated(path: Path, analysis_date: str | None) -> pd.DataFrame:
    """
    Open analysis_results.parquet, parse the date column, and apply an
    optional upper-bound date ceiling.

    Args:
        path:          Path to the parquet file.
        analysis_date: ISO date string ceiling (inclusive); None → no cutoff.

    Returns:
        Full DataFrame filtered to rows up to analysis_date.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])

    if analysis_date is not None:
        df = df[df["date"] <= pd.Timestamp(analysis_date)]

    return df
