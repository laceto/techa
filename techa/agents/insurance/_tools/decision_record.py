"""
agents/insurance/_tools/decision_record.py — Deterministic underwriting decision record.

derive_decision() reads the four WorkerResult dicts and produces a DecisionRecord
without any additional LLM call. It is called at the start of synthesise_node so
the structured decision is always available alongside the narrative final_output.

Decision hierarchy (first match wins):
  decline — actuarial risk_classification=decline, OR medical underwriting_decision=decline,
             OR claims very_high + suspicious_patterns
  refer   — actuarial postpone, OR medical postpone, OR claims very_high (non-suspicious),
             OR accountant financial_health=weak, OR any worker conviction=low
  accept  — all others

Loading blend (individual risk only):
  recommended_loading_pct = actuarial * 0.40 + medical * 0.35 + claims * 0.25
  Financial loading from the accountant is a portfolio-level surcharge, not blended here.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal, Optional

from pydantic import BaseModel, Field

from techa.agents.schema import WorkerResult


class DecisionRecord(BaseModel):
    decision:                Literal["accept", "refer", "decline"]
    recommended_loading_pct: float = Field(
        description=(
            "Blended individual-risk loading: actuarial×0.40 + medical×0.35 + claims×0.25. "
            "Does not include the accountant's portfolio-level financial_loading_pct."
        )
    )
    conditions:  list[str] = Field(
        description=(
            "Exclusion clauses, evidence requirements, and monitoring conditions "
            "attached to an accept or refer decision. Empty for a clean accept."
        )
    )
    review_date: Optional[str] = Field(
        description=(
            "ISO date for mandatory policy re-underwriting. "
            "None for clean accept or decline."
        )
    )
    primary_driver: str = Field(
        description="One sentence identifying the single factor that determined the decision."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "high: all four workers returned high conviction. "
            "low: at least one worker returned low conviction. "
            "medium: otherwise."
        )
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _data(results: list[WorkerResult], agent_id: str) -> dict:
    for r in results:
        if r["agent_id"] == agent_id and not r.get("error"):
            return r.get("data") or {}
    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def derive_decision(
    results: list[WorkerResult],
    assessment_date: str,
) -> DecisionRecord:
    """
    Derive a DecisionRecord deterministically from the four worker results.

    Args:
        results:         state["results"] list — one WorkerResult per dispatched worker.
        assessment_date: ISO date string from payload; used to compute review_date.

    Returns:
        DecisionRecord Pydantic model.
    """
    act = _data(results, "actuarial")
    acc = _data(results, "accountant")
    med = _data(results, "medical_underwriting")
    clm = _data(results, "claims_assessor")

    # ── Signals ─────────────────────────────────────────────────────────────
    act_class  = act.get("risk_classification", "standard")
    med_dec    = med.get("underwriting_decision", "standard")
    clm_level  = clm.get("claims_risk_level", "low")
    clm_sus    = clm.get("suspicious_patterns", False)
    acc_health = acc.get("financial_health", "adequate")

    # ── Decision ─────────────────────────────────────────────────────────────
    if (act_class == "decline"
            or med_dec == "decline"
            or (clm_level == "very_high" and clm_sus)):
        decision: Literal["accept", "refer", "decline"] = "decline"
    elif (act_class == "postpone"
          or med_dec == "postpone"
          or clm_level == "very_high"
          or acc_health == "weak"):
        decision = "refer"
    else:
        decision = "accept"

    # ── Loading blend ─────────────────────────────────────────────────────────
    act_load = float(act.get("mortality_loading_pct") or 0.0)
    med_load = float(med.get("medical_loading_pct")   or 0.0)
    clm_load = float(clm.get("claims_loading_pct")    or 0.0)
    recommended_loading_pct = round(act_load * 0.40 + med_load * 0.35 + clm_load * 0.25, 1)

    # ── Conditions ───────────────────────────────────────────────────────────
    conditions: list[str] = []
    for excl in (med.get("exclusions") or []):
        conditions.append(f"Exclusion: {excl}")
    for req in (med.get("additional_requirements") or []):
        conditions.append(f"Required before issuance: {req}")
    if acc_health in ("marginal", "weak"):
        conditions.append("Financial monitoring: quarterly portfolio review required")
    if clm_sus:
        notes = clm.get("suspicious_pattern_notes", "")
        conditions.append(
            f"Claims investigation: {notes}" if notes else "Claims: refer to Special Investigations Unit"
        )

    # ── Review date ───────────────────────────────────────────────────────────
    try:
        base = date.fromisoformat(assessment_date)
    except (ValueError, TypeError):
        base = date.today()

    if decision == "decline":
        review_date: Optional[str] = None
    elif decision == "refer":
        review_date = (base + timedelta(days=365)).isoformat()
    elif act_class == "substandard" or med_dec == "rated":
        review_date = (base + timedelta(days=730)).isoformat()
    else:
        review_date = None

    # ── Confidence ────────────────────────────────────────────────────────────
    convictions = [
        act.get("conviction", "medium"),
        acc.get("conviction", "medium"),
        med.get("conviction", "medium"),
        clm.get("conviction", "medium"),
    ]
    if all(c == "high" for c in convictions):
        confidence: Literal["high", "medium", "low"] = "high"
    elif any(c == "low" for c in convictions):
        confidence = "low"
    else:
        confidence = "medium"

    # ── Primary driver (worker with highest individual loading) ───────────────
    fin_load = float(acc.get("financial_loading_pct") or 0.0)
    candidates = [
        (act_load, act.get("verdict", "")),
        (med_load, med.get("verdict", "")),
        (clm_load, clm.get("verdict", "")),
        (fin_load, acc.get("verdict", "")),
    ]
    best_load, best_verdict = max(candidates, key=lambda x: x[0])
    primary_driver = best_verdict or f"Blended loading {recommended_loading_pct:.0f}% across all dimensions"

    return DecisionRecord(
        decision=decision,
        recommended_loading_pct=recommended_loading_pct,
        conditions=conditions,
        review_date=review_date,
        primary_driver=primary_driver,
        confidence=confidence,
    )
