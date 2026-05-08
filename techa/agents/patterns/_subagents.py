"""
agents/patterns/_subagents.py — Worker registry.

WORKER_NAMES is the single source of truth for which workers run.
Adding a new worker = add one entry here; no wiring changes needed in agent.py.
"""

from __future__ import annotations

from techa.agents.patterns.graph_nodes import create_subgraph

WORKER_NAMES: list[str] = ["pattern"]


def build_subgraphs() -> dict:
    """Compile one subgraph per worker name."""
    return {name: create_subgraph(name) for name in WORKER_NAMES}
