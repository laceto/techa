"""
agents/orchestrator/graph_state.py — Typed state for the Orchestrator LangGraph.

Single source of truth for all fields that flow between nodes.
Scalar/dict fields use _last; the results accumulator uses the add reducer
so parallel runner_node invocations (via Send) can each append without conflict.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Optional

from typing_extensions import TypedDict

from techa.agents._common import WorkerResult


def _last(a, b):  # noqa: ANN001
    """Reducer: always keep the most recent value. Required for parallel fan-out/fan-in."""
    return b


class OrchestratorState(TypedDict, total=False):
    # ── Caller inputs ──────────────────────────────────────────────────────
    symbol:        Annotated[str,           _last]  # single ticker (required)
    data_source:   Annotated[str,           _last]  # "parquet" (default) | "live"
    analysis_date: Annotated[Optional[str], _last]  # ISO date; None → latest bar (parquet mode only)
    lookback_days: Annotated[int,           _last]  # calendar days of history for live mode (default 365)
    benchmark:     Annotated[str,           _last]  # benchmark ticker for live ta data (default "FTSEMIB.MI")
    fx:            Annotated[Optional[str], _last]  # FX ticker for currency conversion; None = same currency
    relative:      Annotated[bool,          _last]  # True = relative prices; False = absolute (live mode only)

    # ── Injected by Send dispatcher ────────────────────────────────────────
    agent_id: Annotated[Optional[str], _last]       # "indicators" | "patterns" | "ta"

    # ── Set by prepare_node ────────────────────────────────────────────────
    raw_df:        Annotated[Optional[list], _last]  # df.reset_index().to_dict(orient="records")
    resolved_date: Annotated[str,            _last]  # last bar ISO date
    ta_df:         Annotated[Optional[list], _last]  # ta enriched df serialised as df.to_dict(orient="records")

    # ── Accumulated by runner_node (one entry per dispatched agent) ────────
    results: Annotated[list[WorkerResult], add]
    # Each WorkerResult: {"agent_id": str, "data": dict, "error": str | None}

    # ── Set by synthesise_node ─────────────────────────────────────────────
    final_output: Annotated[str, _last]  # combined indicators + patterns + ta plain-text report
