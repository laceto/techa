"""
agents/graph_state.py — Typed state for the TechnicalAnalysis LangGraph.

Single source of truth for all fields that flow between nodes.
Scalar/dict fields use _last; the results accumulator uses the add reducer
so parallel worker_node invocations (via Send) can each append without conflict.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Optional

from typing_extensions import TypedDict

from techa.agents._common import WorkerResult


def _last(a, b):  # noqa: ANN001
    """Reducer: always keep the most recent value. Required for parallel fan-out/fan-in."""
    return b


class TechnicalAnalysisState(TypedDict, total=False):
    # ── Caller inputs ──────────────────────────────────────────────────────────
    symbol:        Annotated[str,           _last]  # single ticker to analyse (required)
    analysis_date: Annotated[Optional[str], _last]  # ISO date; None → latest bar
    data_source:   Annotated[str,           _last]  # "parquet" (default) | "live"
    benchmark:     Annotated[str,           _last]  # benchmark ticker
    # Mode A default: "FTSEMIB.MI" (excluded from result set)
    # Mode B default: user-supplied (used for calculate_relative_prices)
    fx:            Annotated[Optional[str], _last]  # FX ticker for currency conversion (e.g. "EURUSD=X")
    # None → no FX conversion (stock and benchmark share the same currency)

    # ── Injected by Send dispatcher ────────────────────────────────────────────
    agent_id: Annotated[Optional[str], _last]  # set per-dispatch; identifies the active worker

    # ── Set by prepare_node ────────────────────────────────────────────────────
    resolved_date: Annotated[str,            _last]  # actual date resolved from the data
    payload:       Annotated[Optional[dict], _last]
    # payload shape:
    # {
    #   "date":    "YYYY-MM-DD",
    #   "symbol":  str,
    #   "raw_df":  list[dict],   ← df.to_dict(orient="records"); workers reconstruct with pd.DataFrame()
    # }

    # ── Accumulated by worker_node (one entry per dispatched agent) ────────────
    results: Annotated[list[WorkerResult], add]
    # Each WorkerResult: {"agent_id": str, "data": dict, "error": str | None}

    # ── Set by synthesise_node ─────────────────────────────────────────────────
    final_output: Annotated[str, _last]  # both AI reports formatted side by side
