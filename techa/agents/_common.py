"""
agents/_common.py — Shared constants and low-level helpers for all agent subpackages.

Imported by:
  techa.agents.ta._tools.prepare_tools
  techa.agents.patterns._tools.prepare_tools
  techa.agents.ta.graph_state
  techa.agents.patterns.graph_state

WorkerResult is defined in techa.agents.schema and re-exported here for
backward compatibility.  New code should import from techa.agents.schema
directly; _common imports it so existing callers do not need to change.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Re-export from the canonical schema module.
from techa.agents.schema import WorkerResult  # noqa: F401

RESULTS_PATH: Path = Path("data/results/it/analysis_results.parquet")
HISTORY_BARS: int  = 300   # max rows per ticker kept from parquet; enough for ADX(14)/MA(150)/RSI(14)


def get_result_by_id(results: list[WorkerResult], agent_id: str) -> "WorkerResult | None":
    """Return the first WorkerResult for agent_id, or None if not found."""
    return next((r for r in results if r["agent_id"] == agent_id), None)


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
