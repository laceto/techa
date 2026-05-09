"""
agents/orchestrator/agent.py — create_orchestrator() graph factory.

Builds the LangGraph StateGraph:
  START → prepare_node → (_dispatcher → Send) → runner_node → synthesise_node → END

prepare_node loads raw OHLCV once for the given symbol (serialised into
state["raw_df"]) and also loads the ta-enriched DataFrame (serialised into
state["ta_df"]). The dispatcher fans out to runner_node three times — once each
for agent_id="indicators", agent_id="patterns", and agent_id="ta" — so all three
domain agents run in parallel against the already-loaded data. synthesise_node
collects all three WorkerResult entries and calls an LLM to format a combined
plain-text report.

Adding a new runner requires only adding an entry to RUNNER_NAMES — no edge
wiring changes are needed.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from techa.agents.orchestrator.graph_state import OrchestratorState
from techa.agents.orchestrator.graph_nodes import prepare_node, runner_node, synthesise_node

RUNNER_NAMES: list[str] = ["indicators", "patterns", "ta"]


def _dispatcher(state: OrchestratorState) -> list[Send]:
    """Fan out to runner_node once per registered runner, injecting agent_id via Send."""
    return [Send("runner_node", {"agent_id": name, **state}) for name in RUNNER_NAMES]


def create_orchestrator(
    symbol: str,
    data_source: str = "live",
    analysis_date: str | None = None,
    lookback_days: int = 365,
    benchmark: str = "FTSEMIB.MI",
    fx: str | None = None,
    checkpointer=None,
):
    """
    Build and compile the Orchestrator LangGraph for a single ticker.

    Loads raw OHLCV (for indicators and patterns) and the ta-enriched DataFrame
    (for breakout + MA crossover analysis) once in prepare_node, then fans out in
    parallel to three runners: indicators (trend / momentum / volatility), patterns
    (candlestick scan), and ta (breakout + MA crossover). Results are compiled into
    a single structured markdown brief by synthesise_node via an LLM call.

    Args:
        symbol:        Ticker to analyse (required, e.g. "PST.MI").
        data_source:   "live" (default, downloads via yfinance) or "parquet"
                       (reads enriched data from analysis_results.parquet).
        analysis_date: ISO date string ceiling; None → latest available bar.
                       Only applied in parquet mode to anchor the data window.
        lookback_days: Calendar days of OHLCV history to fetch (live mode only,
                       default 365).
        benchmark:     Benchmark ticker used when computing relative prices for
                       the ta runner in live mode (default "FTSEMIB.MI").
        fx:            Optional FX ticker for currency conversion when the stock
                       and benchmark trade in different currencies (e.g. "EURUSD=X").
                       Pass None (default) when they share the same currency.
        checkpointer:  Optional LangGraph checkpointer for persistence / resumption.

    Returns:
        Compiled graph ready for .invoke() / .stream(). The graph exposes
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

        # Live mode with FX conversion
        graph = create_orchestrator("TCEHY", benchmark="SPY", fx="EURUSD=X")
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
        "benchmark":     benchmark,
        "fx":            fx,
        "results":       [],
    }

    graph = builder.compile(checkpointer=checkpointer)
    graph._initial_state = initial_state
    return graph
