"""
agents/patterns/graph_state.py — Typed state for the PatternScan LangGraph.

Single source of truth for all fields that flow between nodes.
Every field uses the _last reducer so parallel branches can merge without conflict.
"""

from __future__ import annotations

from typing import Annotated, Optional

from typing_extensions import TypedDict


def _last(a, b):  # noqa: ANN001
    """Reducer: always keep the most recent value. Required for parallel fan-out/fan-in."""
    return b


class PatternScanState(TypedDict, total=False):
    # ── Caller inputs ──────────────────────────────────────────────────────────
    tickers:       Annotated[list[str],      _last]  # tickers to scan (required)
    signal_filter: Annotated[str,            _last]  # "all" | "bull" | "bear"
    data_source:   Annotated[str,            _last]  # "parquet" (default) | "live"
    analysis_date: Annotated[Optional[str],  _last]  # ISO date anchor; None → latest bar (parquet mode)
    lookback_days: Annotated[int,            _last]  # calendar days of OHLCV history to download (live mode)
    lookback_bars: Annotated[int,            _last]  # trading bars of recent pattern history to include (default 20)
    benchmark:     Annotated[str,            _last]  # accepted for API consistency; not used by pattern nodes
    fx:            Annotated[Optional[str],  _last]  # accepted for API consistency; not used by pattern nodes

    # ── Set by prepare_node ────────────────────────────────────────────────────
    scan_date:    Annotated[str,            _last]   # ISO date of the scan run
    payload_json: Annotated[str,            _last]
    # payload_json shape:
    # {
    #   "tickers":       list[str],
    #   "scan_date":     "YYYY-MM-DD",
    #   "signal_filter": str,
    #   "hits":          list[{"ticker", "date", "display_name", "signal"}],  # last-bar only
    #   "total_hits":    int,
    #   "recent_hits":   list[{"ticker", "date", "display_name", "signal"}],  # last lookback_bars bars
    #   "lookback_bars": int,
    # }

    # ── Set by pattern worker ──────────────────────────────────────────────────
    pattern_result: Annotated[Optional[dict], _last]  # PatternScanAnalysis.model_dump()
    # {"error": str} when the worker caught an exception

    # ── Set by synthesise_node ─────────────────────────────────────────────────
    final_output: Annotated[str, _last]  # formatted scan report
