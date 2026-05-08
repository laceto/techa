"""
agents/indicators/agent.py — create_indicator_agent() graph factory.

Builds the LangGraph StateGraph:
  START → prepare_node → (Send dispatcher) → worker_node → synthesise_node → END

The dispatcher fans out to worker_node once per entry in WORKER_NAMES via Send,
injecting agent_id into each copy of the state. Adding a new worker requires only
updating WORKER_NAMES in _subagents.py — no edge wiring changes here.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from techa.agents.indicators.graph_state import IndicatorAnalysisState
from techa.agents.indicators.graph_nodes import prepare_node, synthesise_node, worker_node
from techa.agents.indicators._subagents import WORKER_NAMES


def _dispatcher(state: IndicatorAnalysisState) -> list[Send]:
    """Fan out to worker_node once per registered agent, injecting agent_id via Send."""
    return [Send("worker_node", {"agent_id": name, **state}) for name in WORKER_NAMES]


def create_indicator_agent(
    symbol: str,
    analysis_date: str | None = None,
    data_source: str = "live",    # "live" | "parquet"
    lookback_days: int = 365,
    checkpointer=None,
):
    """
    Build and compile the IndicatorAnalysis LangGraph for a single ticker.

    Args:
        symbol:        Ticker to analyse (required, e.g. "PST.MI").
        analysis_date: ISO date string ceiling; None → latest available bar (parquet mode only).
        data_source:   "live" (default, downloads via yfinance) or "parquet"
                       (reads ropen/rhigh/rlow/rclose from analysis_results.parquet).
        lookback_days: Calendar days of OHLCV history to fetch (live mode only, default 365).
        checkpointer:  Optional LangGraph checkpointer for persistence / resumption.

    Returns:
        CompiledStateGraph ready for .invoke() / .stream().
    """
    builder = StateGraph(IndicatorAnalysisState)

    builder.add_node("prepare",     prepare_node)
    builder.add_node("worker_node", worker_node)
    builder.add_node("synthesise",  synthesise_node)

    builder.add_edge(START, "prepare")
    builder.add_conditional_edges("prepare", _dispatcher)
    builder.add_edge("worker_node", "synthesise")
    builder.add_edge("synthesise", END)

    initial_state: IndicatorAnalysisState = {
        "symbol":        symbol,
        "analysis_date": analysis_date,
        "data_source":   data_source,
        "lookback_days": lookback_days,
        "results":       [],
    }

    graph = builder.compile(checkpointer=checkpointer)
    graph._initial_state = initial_state
    return graph
