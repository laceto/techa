"""
agents/schema.py — Canonical type definitions shared across all agent subpackages.

Single source of truth for WorkerResult and any other inter-node data contracts.
Import from here, not from _common, when you need just the TypedDict definition.

_common.py re-exports WorkerResult from here for backward compatibility.
"""

from __future__ import annotations

from typing import Optional

from typing_extensions import TypedDict


class WorkerResult(TypedDict):
    """Standardized result envelope written by every worker_node / runner_node."""
    agent_id: str            # identifies which worker produced this result
    data:     dict           # structured output, serialised from the Pydantic model
    error:    Optional[str]  # populated when the worker caught an exception; None otherwise
