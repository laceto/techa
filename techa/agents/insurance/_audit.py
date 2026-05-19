"""
agents/insurance/_audit.py — Audit trail serialisation for underwriting decisions.

append_audit_record() is called at the end of every synthesise_node run.
It appends one row to data/results/insurance/decisions.parquet so every
underwriting decision is reproducible for regulatory audit and model comparison.

The parquet file is created on first write. Concurrent writes from batch runs
are safe because each write reads the existing file, appends, and overwrites —
this is intentionally single-writer; for high-concurrency use a proper database.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_AUDIT_PATH = Path(__file__).parents[4] / "data" / "results" / "insurance" / "decisions.parquet"

_COLUMNS = [
    "run_id",
    "policy_id",
    "assessment_date",
    "decision",
    "recommended_loading_pct",
    "confidence",
    "fraud_risk_level",
    "worker_results_json",
    "conditions_json",
    "review_date",
    "primary_driver",
    "final_output_length",
    "model",
]


def append_audit_record(
    *,
    policy_id: str,
    assessment_date: str,
    decision_dict: dict | None,
    fraud_risk_level: str | None,
    results: list[dict],
    final_output: str,
    model: str,
) -> None:
    """
    Append one audit row to the decisions parquet file.

    Safe to call even when decision_dict is None (derive_decision failed) —
    the row is still written with None fields so the attempt is recorded.

    Args:
        policy_id:        Policy / application reference.
        assessment_date:  ISO date string from payload.
        decision_dict:    DecisionRecord.model_dump() or None.
        fraud_risk_level: From state["fraud_risk_level"] or None.
        results:          state["results"] list of WorkerResult dicts.
        final_output:     Narrative synthesis string.
        model:            MODEL constant at run time.
    """
    try:
        import pandas as pd
    except ImportError:
        log.warning("[audit] pandas not available — audit record not written")
        return

    try:
        row: dict[str, Any] = {
            "run_id":                  str(uuid.uuid4()),
            "policy_id":               policy_id,
            "assessment_date":         assessment_date or str(date.today()),
            "decision":                (decision_dict or {}).get("decision"),
            "recommended_loading_pct": (decision_dict or {}).get("recommended_loading_pct"),
            "confidence":              (decision_dict or {}).get("confidence"),
            "fraud_risk_level":        fraud_risk_level or "low",
            "worker_results_json":     json.dumps(results, default=str),
            "conditions_json":         json.dumps((decision_dict or {}).get("conditions", []), default=str),
            "review_date":             (decision_dict or {}).get("review_date"),
            "primary_driver":          (decision_dict or {}).get("primary_driver"),
            "final_output_length":     len(final_output),
            "model":                   model,
        }

        new_df = pd.DataFrame([row])

        _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

        if _AUDIT_PATH.exists():
            existing = pd.read_parquet(_AUDIT_PATH)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined.to_parquet(_AUDIT_PATH, index=False)
        log.info(
            "[audit] record appended: policy=%s decision=%s run_id=%s",
            policy_id,
            row["decision"],
            row["run_id"],
        )
    except Exception as exc:
        log.error("[audit] failed to write audit record: %s", exc, exc_info=True)
