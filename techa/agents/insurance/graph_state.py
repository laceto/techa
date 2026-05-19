"""
agents/insurance/graph_state.py — Typed state for the InsuranceAnalysis LangGraph.

Single source of truth for all fields that flow between nodes.
Scalar/dict fields use _last; the results accumulator uses the add reducer
so parallel worker_node invocations (via Send) can each append without conflict.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Optional

from typing_extensions import TypedDict

from techa.agents.schema import WorkerResult


def _last(a, b):  # noqa: ANN001
    """Reducer: always keep the most recent value. Required for parallel fan-out/fan-in."""
    return b


class InsuranceAnalysisState(TypedDict, total=False):
    # ── Caller inputs ──────────────────────────────────────────────────────────
    policy_id:    Annotated[str,           _last]  # unique policy / application reference (required)
    risk_profile: Annotated[Optional[dict], _last]  # full policy application data dict; None → demo profile

    # ── Injected by Send dispatcher ────────────────────────────────────────────
    agent_id: Annotated[Optional[str], _last]  # set per-dispatch; identifies the active worker

    # ── Set by prepare_node ────────────────────────────────────────────────────
    payload: Annotated[Optional[dict], _last]
    # payload shape:
    # {
    #   "policy_id":         str,
    #   "product_type":      str,          e.g. "whole_life" | "term_life"
    #   "assessment_date":   "YYYY-MM-DD",
    #   "applicant":         dict,          age, gender, smoker, bmi, medical_history, …
    #   "coverage":          dict,          sum_assured, premium_annual, term_years, …
    #   "claims_history":    dict,          total_claims_count, total_claims_paid, …
    #   "financial_metrics": dict,          loss_ratio, expense_ratio, reserve_adequacy_pct, …
    # }

    # ── Accumulated by worker_node (one entry per dispatched specialist) ───────
    results: Annotated[list[WorkerResult], add]
    # Each WorkerResult: {"agent_id": str, "data": dict, "error": str | None}

    # ── Set by synthesise_node ─────────────────────────────────────────────────
    final_output: Annotated[str,           _last]  # narrative underwriting brief (Life Head of Business)
    decision:     Annotated[Optional[dict], _last]  # DecisionRecord.model_dump() — machine-readable verdict
    # decision shape:
    # {
    #   "decision":                "accept" | "refer" | "decline",
    #   "recommended_loading_pct": float,   # blended: actuarial×0.40 + medical×0.35 + claims×0.25
    #   "conditions":              list[str],  # exclusions + evidence requirements
    #   "review_date":             "YYYY-MM-DD" | None,
    #   "primary_driver":          str,
    #   "confidence":              "high" | "medium" | "low",
    # }
