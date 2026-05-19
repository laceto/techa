"""
agents/insurance/batch.py — Concurrent batch runner for the insurance agent.

run_batch() processes a list of (policy_id, risk_profile) pairs concurrently
using asyncio + graph.ainvoke(), with a semaphore to cap concurrency.

Each result dict contains:
    policy_id    — str
    final_output — str (narrative brief) or error message
    decision     — dict (DecisionRecord) or None
    error        — str | None

Usage:
    import asyncio
    from techa.agents.insurance.batch import run_batch

    cases = [
        ("POL-001", None),   # uses demo profile
        ("POL-002", {"applicant": {"age": 55, "smoker": True}}),
    ]
    results = asyncio.run(run_batch(cases, max_concurrency=5))
    for r in results:
        print(r["policy_id"], r["decision"])
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)


async def run_batch(
    cases: list[tuple[str, dict | None]],
    max_concurrency: int = 10,
) -> list[dict[str, Any]]:
    """
    Run the insurance agent concurrently for a list of (policy_id, risk_profile) pairs.

    Args:
        cases:           List of (policy_id, risk_profile) tuples.
                         Pass risk_profile=None to use the built-in demo profile.
        max_concurrency: Maximum number of simultaneous graph invocations.
                         Each invocation makes several OpenAI calls; keep this ≤ 20
                         to avoid rate-limiting.

    Returns:
        List of result dicts in the same order as `cases`:
        [
            {
                "policy_id":    str,
                "final_output": str,    # narrative brief or error message
                "decision":     dict | None,  # DecisionRecord.model_dump()
                "error":        str | None,
            },
            ...
        ]
    """
    from techa.agents.insurance.agent import create_insurance_agent

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_one(policy_id: str, risk_profile: dict | None) -> dict[str, Any]:
        async with semaphore:
            try:
                graph = create_insurance_agent(policy_id, risk_profile=risk_profile)
                final_state = await graph.ainvoke(graph._initial_state)
                return {
                    "policy_id":    policy_id,
                    "final_output": final_state.get("final_output", ""),
                    "decision":     final_state.get("decision"),
                    "error":        None,
                }
            except Exception as exc:
                log.error("[batch] %s failed: %s", policy_id, exc, exc_info=True)
                return {
                    "policy_id":    policy_id,
                    "final_output": f"Error: {exc}",
                    "decision":     None,
                    "error":        str(exc),
                }

    tasks = [_run_one(pid, profile) for pid, profile in cases]
    results = await asyncio.gather(*tasks)
    log.info("[batch] completed %d/%d cases successfully",
             sum(1 for r in results if r["error"] is None), len(results))
    return list(results)
