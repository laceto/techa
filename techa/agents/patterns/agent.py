"""
agents/patterns/agent.py — create_pattern_agent() graph factory.

Builds the LangGraph StateGraph:
  START → prepare_node → pattern_worker → synthesise_node → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from techa.agents.patterns.graph_state import PatternScanState
from techa.agents.patterns.graph_nodes import prepare_node, synthesise_node
from techa.agents.patterns._subagents import WORKER_NAMES, build_subgraphs


def create_pattern_agent(
    tickers: list[str],
    analysis_date: str | None = None,
    data_source: str = "parquet",
    benchmark: str = "FTSEMIB.MI",
    fx: str | None = None,
    signal_filter: str = "all",
    lookback_days: int = 365,
    lookback_bars: int = 20,
    checkpointer=None,
):
    """
    Build and compile the PatternScan LangGraph for a list of tickers.

    Args:
        tickers:       Ticker symbols to scan (required, e.g. ["A2A.MI", "ENI.MI"]).
        analysis_date: ISO date string; None → latest available bar.
                       Only applied in parquet mode to anchor the scan window.
        data_source:   "parquet" (default) or "live".
                       parquet — reads ropen/rhigh/rlow/rclose from
                                 data/results/it/analysis_results.parquet.
                       live    — downloads raw OHLCV via yfinance.
        benchmark:     Accepted for API consistency with create_manager.
                       Not used by the pattern nodes (patterns scan raw OHLCV).
        fx:            Accepted for API consistency with create_manager.
                       Not used by the pattern nodes.
        signal_filter: "all" (default), "bull" (+100 only), or "bear" (-100 only).
        lookback_days: Calendar days of OHLCV history to download (live mode only).
        lookback_bars: Trading bars of recent pattern history included in the LLM payload
                       alongside the last-bar hits. Default 20 (≈ 1 calendar month).
                       Gives the model context on pattern clustering and recurrence.
        checkpointer:  Optional LangGraph checkpointer for persistence / resumption.

    Returns:
        CompiledStateGraph ready for .invoke() / .stream().

    Example:
        # Parquet mode (default)
        graph = create_pattern_agent(["A2A.MI", "ENI.MI"], analysis_date="2024-06-30")
        result = graph.invoke(graph._initial_state)
        print(result["final_output"])

        # Live mode
        graph = create_pattern_agent(["A2A.MI", "ENI.MI"], data_source="live")
        result = graph.invoke(graph._initial_state)
        print(result["final_output"])
    """
    builder = StateGraph(PatternScanState)

    builder.add_node("prepare", prepare_node)
    builder.add_node("synthesise", synthesise_node)

    subgraphs = build_subgraphs()
    for name, subgraph in subgraphs.items():
        builder.add_node(name, subgraph)

    builder.add_edge(START, "prepare")

    for name in WORKER_NAMES:
        builder.add_edge("prepare", name)

    for name in WORKER_NAMES:
        builder.add_edge(name, "synthesise")

    builder.add_edge("synthesise", END)

    initial_state: PatternScanState = {
        "tickers":       tickers,
        "analysis_date": analysis_date,
        "data_source":   data_source,
        "benchmark":     benchmark,
        "fx":            fx,
        "signal_filter": signal_filter,
        "lookback_days": lookback_days,
        "lookback_bars": lookback_bars,
    }

    graph = builder.compile(checkpointer=checkpointer)
    graph._initial_state = initial_state
    return graph
