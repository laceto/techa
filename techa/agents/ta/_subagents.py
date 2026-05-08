"""
agents/_subagents.py — Worker registry.

WORKER_NAMES is the single source of truth for which strategies run.
Adding a new strategy requires only adding one entry here — the dispatcher
in agent.py fans out dynamically via Send; no graph wiring changes needed.
"""

from __future__ import annotations

WORKER_NAMES: list[str] = ["breakout", "ma"]
