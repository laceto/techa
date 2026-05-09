"""
agents/patterns/_subagents.py — Worker registry.

WORKER_NAMES is the single source of truth for which workers run.
Adding a new worker requires only adding one entry here — the dispatcher
in agent.py fans out dynamically via Send; no graph wiring changes needed.
"""

from __future__ import annotations

from techa.agents.patterns._tools.ask_pattern_trader import ask_pattern_trader

WORKER_NAMES: list[str] = ["pattern"]


def _run_pattern(payload: dict):
    return ask_pattern_trader(payload, tickers=payload.get("tickers", []))


WORKER_REGISTRY: dict = {
    "pattern": _run_pattern,
}
