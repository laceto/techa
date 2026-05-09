"""
agents/orchestrator/agent.py — create_orchestrator() graph factory.

Builds the LangGraph StateGraph:
  START → prepare_node → (_dispatcher → Send) → runner_node → synthesise_node → END

prepare_node loads raw OHLCV once for the given symbol and serialises it into
state["raw_df"]. The dispatcher fans out to runner_node twice — once with
agent_id="indicators" and once with agent_id="patterns" — so both domain agents
run in parallel against the same already-loaded data. synthesise_node collects
both WorkerResult entries and formats a combined plain-text report.

Adding a new runner requires only adding an entry to RUNNER_NAMES — no edge
wiring changes are needed.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Send

from techa.agents.orchestrator.graph_state import OrchestratorState
from techa.agents.orchestrator.graph_nodes import prepare_node, runner_node, synthesise_node

RUNNER_NAMES: list[str] = ["indicators", "patterns"]


def _dispatcher(state: OrchestratorState) -> list[Send]:
    """Fan out to runner_node once per registered runner, injecting agent_id via Send."""
    return [Send("runner_node", {"agent_id": name, **state}) for name in RUNNER_NAMES]


def create_orchestrator(
    symbol: str,
    data_source: str = "live",
    analysis_date: str | None = None,
    lookback_days: int = 365,
    checkpointer=None,
) -> CompiledGraph:
    """
    Build and compile the Orchestrator LangGraph for a single ticker.

    Loads raw OHLCV once and fans out in parallel to the indicators agent
    (trend / momentum / volatility) and the patterns agent (candlestick scan).
    Results are combined into a single plain-text report by synthesise_node
    without an additional LLM call.

    Args:
        symbol:        Ticker to analyse (required, e.g. "PST.MI").
        data_source:   "live" (default, downloads via yfinance) or "parquet"
                       (reads ropen/rhigh/rlow/rclose from analysis_results.parquet).
        analysis_date: ISO date string ceiling; None → latest available bar.
                       Only applied in parquet mode to anchor the data window.
        lookback_days: Calendar days of OHLCV history to fetch (live mode only,
                       default 365).
        checkpointer:  Optional LangGraph checkpointer for persistence / resumption.

    Returns:
        CompiledGraph ready for .invoke() / .stream(). The graph exposes
        ._initial_state so callers can do: graph.invoke(graph._initial_state).

    Example:
        from techa.agents.orchestrator import create_orchestrator

        # Live mode (default)
        graph = create_orchestrator("PST.MI")
        result = graph.invoke(graph._initial_state)
        print(result["final_output"])

        # Parquet mode
        graph = create_orchestrator("A2A.MI", data_source="parquet",
                                    analysis_date="2024-06-30")
        result = graph.invoke(graph._initial_state)
        print(result["final_output"])
    """
    builder = StateGraph(OrchestratorState)

    builder.add_node("prepare",     prepare_node)
    builder.add_node("runner_node", runner_node)
    builder.add_node("synthesise",  synthesise_node)

    builder.add_edge(START, "prepare")
    builder.add_conditional_edges("prepare", _dispatcher)
    builder.add_edge("runner_node", "synthesise")
    builder.add_edge("synthesise", END)

    initial_state: OrchestratorState = {
        "symbol":        symbol,
        "data_source":   data_source,
        "analysis_date": analysis_date,
        "lookback_days": lookback_days,
        "results":       [],
    }

    graph = builder.compile(checkpointer=checkpointer)
    graph._initial_state = initial_state
    return graph
