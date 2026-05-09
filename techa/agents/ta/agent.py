"""
agents/agent.py — create_manager() graph factory.

Builds the LangGraph StateGraph:
  START → prepare_node → (Send dispatcher) → worker_node → synthesise_node → END

The dispatcher fans out to worker_node once per entry in WORKER_NAMES via Send,
injecting agent_id into each copy of the state. Adding a new worker requires only
updating WORKER_NAMES in _subagents.py — no edge wiring changes here.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from techa.agents.ta.graph_state import TechnicalAnalysisState
from techa.agents.ta.graph_nodes import prepare_node, synthesise_node, worker_node
from techa.agents.ta._subagents import WORKER_NAMES


def _dispatcher(state: TechnicalAnalysisState) -> list[Send]:
    """Fan out to worker_node once per registered agent, injecting agent_id via Send."""
    return [Send("worker_node", {"agent_id": name, **state}) for name in WORKER_NAMES]


def create_manager(
    symbol: str,
    analysis_date: str | None = None,
    data_source: str = "parquet",   # "parquet" | "live"
    benchmark: str = "FTSEMIB.MI",
    fx: str | None = None,
    relative: bool = False,
    checkpointer=None,
):
    """
    Build and compile the TechnicalAnalysis LangGraph for a single ticker.

    Args:
        symbol:        Ticker to analyse (required, e.g. "A2A.MI").
        analysis_date: ISO date string; None → latest available bar.
        data_source:   "parquet" (default) or "live" (downloads via YFinanceDataHandler).
        benchmark:     Benchmark ticker.
                       Mode A: excluded from result set (default "FTSEMIB.MI").
                       Mode B: used for calculate_relative_prices (e.g. "H4ZX.DE").
        fx:            Optional FX ticker for currency conversion (e.g. "EURUSD=X").
                       Pass None when stock and benchmark share the same currency.
        relative:      If True, signals use relative prices (stock / benchmark).
                       If False (default, matches config.json), absolute prices are
                       used. Applies to live mode only; parquet data is pre-computed.
        checkpointer:  Optional LangGraph checkpointer for persistence / resumption.

    Returns:
        CompiledStateGraph ready for .invoke() / .stream().
    """
    builder = StateGraph(TechnicalAnalysisState)

    builder.add_node("prepare",      prepare_node)
    builder.add_node("worker_node",  worker_node)
    builder.add_node("synthesise",   synthesise_node)

    builder.add_edge(START, "prepare")
    builder.add_conditional_edges("prepare", _dispatcher)
    builder.add_edge("worker_node", "synthesise")
    builder.add_edge("synthesise", END)

    initial_state: TechnicalAnalysisState = {
        "symbol":        symbol,
        "analysis_date": analysis_date,
        "data_source":   data_source,
        "benchmark":     benchmark,
        "fx":            fx,
        "relative":      relative,
        "results":       [],
    }

    graph = builder.compile(checkpointer=checkpointer)
    graph._initial_state = initial_state
    return graph
