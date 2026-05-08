"""
agents/indicators/_subagents.py — Worker registry.

WORKER_NAMES is the single source of truth for which indicator dimensions run.
Adding a new dimension requires only adding one entry here — the dispatcher
in agent.py fans out dynamically via Send; no graph wiring changes needed.
"""

from __future__ import annotations

WORKER_NAMES: list[str] = ["trend", "momentum", "volatility"]
