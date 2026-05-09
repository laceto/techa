"""
agents/_subagents.py — Worker registry.

WORKER_NAMES is the single source of truth for which strategies run.
Adding a new strategy requires only adding one entry here — the dispatcher
in agent.py fans out dynamically via Send; no graph wiring changes needed.
"""

from __future__ import annotations

import pandas as pd

from techa.agents.ta._tools.ask_bo_trader import ask_bo_trader
from techa.agents.ta._tools.ask_ma_trader import ask_ma_trader

WORKER_NAMES: list[str] = ["breakout", "ma"]


def _run_breakout(df: pd.DataFrame, symbol: str):
    from techa.breakout.bo_snapshot import build_snapshot as build
    return ask_bo_trader(build(df), ticker=symbol)


def _run_ma(df: pd.DataFrame, symbol: str):
    from techa.ma.ma_snapshot import build_snapshot as build
    return ask_ma_trader(build(df), ticker=symbol)


WORKER_REGISTRY: dict = {
    "breakout": _run_breakout,
    "ma":       _run_ma,
}
