"""
agents/indicators/_subagents.py — Worker registry.

WORKER_NAMES is the single source of truth for which indicator dimensions run.
Adding a new dimension requires only adding one entry here — the dispatcher
in agent.py fans out dynamically via Send; no graph wiring changes needed.
"""

from __future__ import annotations

from techa.agents.indicators._tools.ask_trend_analyst import ask_trend_analyst
from techa.agents.indicators._tools.ask_momentum_analyst import ask_momentum_analyst
from techa.agents.indicators._tools.ask_volatility_analyst import ask_volatility_analyst

WORKER_NAMES: list[str] = ["trend", "momentum", "volatility"]

WORKER_REGISTRY: dict = {
    "trend":      ask_trend_analyst,
    "momentum":   ask_momentum_analyst,
    "volatility": ask_volatility_analyst,
}
