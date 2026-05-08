"""
agents/patterns/graph_nodes.py — Node implementations for the PatternScan graph.

Nodes:
  prepare_node    — loads OHLCV for each ticker (parquet or live), calls scan_last_bar,
                    serialises the hits payload to payload_json.
  create_subgraph — factory: returns a compiled single-node subgraph that calls
                    ask_pattern_trader.
  synthesise_node — formats pattern_result into a readable final text report.

Invariant: payload_json is the sole data channel from prepare_node to workers.
           No worker reads from disk or network.
"""

from __future__ import annotations

import json
import logging
import time

import pandas as pd
from langgraph.graph import END, START, StateGraph

from techa.agents._common import RESULTS_PATH
from techa.agents.patterns.graph_state import PatternScanState
from techa.agents.patterns._tools.ask_pattern_trader import ask_pattern_trader
from techa.agents.patterns._tools.prepare_tools import (
    load_ohlcv_from_parquet,
    download_ohlcv_live,
)
from techa.patterns.scanner import scan_last_bar, scan_patterns

log = logging.getLogger(__name__)

_DEFAULT_LOOKBACK_DAYS = 365


# ---------------------------------------------------------------------------
# prepare_node
# ---------------------------------------------------------------------------


def prepare_node(state: PatternScanState) -> dict:
    """
    Load OHLCV for each ticker, run scan_last_bar, serialise to payload_json.

    Mode A (parquet): reads ropen/rhigh/rlow/rclose from analysis_results.parquet,
                      anchored to analysis_date (or latest bar if None).
    Mode B (live):    downloads raw OHLCV via yfinance over the lookback_days window.

    Args:
        state: Must contain "tickers". Optional: "data_source", "analysis_date",
               "signal_filter", "lookback_days", "lookback_bars".

    Returns:
        Dict updating "payload_json" and "scan_date".
    """
    t0            = time.perf_counter()
    tickers       = state["tickers"]
    signal_filter = state.get("signal_filter", "all")
    data_source   = state.get("data_source", "parquet")
    analysis_date = state.get("analysis_date")
    lookback_days = state.get("lookback_days", _DEFAULT_LOOKBACK_DAYS)
    lookback_bars = state.get("lookback_bars", 20)

    if data_source == "parquet":
        log.info(
            "[prepare] loading %d tickers from parquet (analysis_date=%s)",
            len(tickers), analysis_date,
        )
        ohlcv_by_ticker, scan_date = load_ohlcv_from_parquet(
            RESULTS_PATH, tickers, analysis_date
        )
    else:  # live
        ohlcv_by_ticker, scan_date = download_ohlcv_live(tickers, lookback_days)

    if not ohlcv_by_ticker:
        log.error("[prepare] no OHLCV data loaded — all tickers failed")

    log.info("[prepare] loaded %d / %d tickers", len(ohlcv_by_ticker), len(tickers))

    hits_df = scan_last_bar(ohlcv_by_ticker, signal_filter=signal_filter)

    # Build recent_hits: scan_patterns over the last lookback_bars rows per ticker
    recent_frames = []
    for ticker, ohlcv in ohlcv_by_ticker.items():
        recent_ohlcv = ohlcv.iloc[-lookback_bars:]
        recent_df = scan_patterns(recent_ohlcv, signal_filter=signal_filter)
        if not recent_df.empty:
            recent_df = recent_df.assign(
                ticker=ticker,
                date=recent_df["date"].astype(str),
                direction=recent_df["signal"].apply(lambda s: "bullish" if s > 0 else "bearish"),
            )[["ticker", "date", "display_name", "signal", "direction"]]
            recent_frames.append(recent_df)

    recent_hits_records = (
        pd.concat(recent_frames, ignore_index=True).to_dict(orient="records")
        if recent_frames else []
    )

    log.info(
        "[prepare] scans complete in %.2fs — last-bar: %d hits, recent (%d bars): %d hits (data_source=%s)",
        time.perf_counter() - t0,
        len(hits_df),
        lookback_bars,
        len(recent_hits_records),
        data_source,
    )

    hits_records = hits_df.assign(
        date=hits_df["date"].astype(str),
        direction=hits_df["signal"].apply(lambda s: "bullish" if s > 0 else "bearish"),
    ).to_dict(orient="records")

    payload = {
        "tickers":       tickers,
        "scan_date":     scan_date,
        "signal_filter": signal_filter,
        "hits":          hits_records,
        "total_hits":    len(hits_df),
        "recent_hits":   recent_hits_records,
        "lookback_bars": lookback_bars,
    }

    return {
        "payload_json": json.dumps(payload),
        "scan_date":    scan_date,
    }


# ---------------------------------------------------------------------------
# Subgraph factory
# ---------------------------------------------------------------------------


def create_subgraph(worker_name: str):
    """
    Build a compiled single-node subgraph that calls one worker tool.

    The subgraph never raises — exceptions are caught and returned as
    {"error": str(exc)} so the graph can continue to synthesise_node.

    Args:
        worker_name: Currently only "pattern" is supported.
    """
    result_key = f"{worker_name}_result"

    def run_worker(state: PatternScanState) -> dict:
        try:
            payload = json.loads(state["payload_json"])
            tickers = payload.get("tickers", [])
            if worker_name == "pattern":
                result = ask_pattern_trader(payload, tickers=tickers)
                log.info("[%s] analysis complete (%d tickers)", worker_name, len(tickers))
                return {result_key: result.model_dump()}
            else:
                raise ValueError(f"Unknown worker: {worker_name!r}")
        except Exception as exc:
            log.error("[%s] worker failed: %s", worker_name, exc, exc_info=True)
            return {result_key: {"error": str(exc)}}

    graph = StateGraph(PatternScanState)
    graph.add_node("run_worker", run_worker)
    graph.add_edge(START, "run_worker")
    graph.add_edge("run_worker", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# synthesise_node
# ---------------------------------------------------------------------------


def synthesise_node(state: PatternScanState) -> dict:
    """
    Format the pattern worker's structured result into a readable text report.

    Never raises — missing or errored results produce a partial report.
    """
    tickers        = state.get("tickers", [])
    scan_date      = state.get("scan_date", "unknown")
    pattern_result = state.get("pattern_result") or {}

    sep = "=" * 60

    if not pattern_result:
        return {"final_output": f"{sep}\n  Pattern scan returned no results.\n{sep}"}

    if "error" in pattern_result:
        return {"final_output": f"{sep}\n  Pattern scan failed: {pattern_result['error']}\n{sep}"}

    lines = [
        sep,
        f"  Candlestick Pattern Scan — {scan_date}",
        f"  Tickers: {', '.join(tickers)}",
        sep,
        "",
        f"  {pattern_result.get('description', '')}",
        "",
        (
            f"  Total hits: {pattern_result.get('total_hits', 0)}"
            f"  |  Bullish: {pattern_result.get('bullish_count', 0)}"
            f"  |  Bearish: {pattern_result.get('bearish_count', 0)}"
        ),
        "",
    ]

    for ts in pattern_result.get("ticker_summaries", []):
        lines.append(f"  {ts['ticker']}  ({ts['date']})")
        lines.append(f"    Bias: {ts['net_bias']}  Conviction: {ts['conviction']}")
        if ts.get("bullish_patterns"):
            lines.append(f"    Bullish : {', '.join(ts['bullish_patterns'])}")
        if ts.get("bearish_patterns"):
            lines.append(f"    Bearish : {', '.join(ts['bearish_patterns'])}")
        lines.append(f"    → {ts['verdict']}")
        lines.append("")

    if pattern_result.get("top_actionable"):
        lines.append(f"  Top actionable : {', '.join(pattern_result['top_actionable'])}")
    if pattern_result.get("watchlist"):
        lines.append(f"  Watchlist      : {', '.join(pattern_result['watchlist'])}")

    lines += ["", f"  Summary: {pattern_result.get('summary', '')}", sep]

    final = "\n".join(lines)
    log.info("[synthesise] report generated (%d chars)", len(final))
    return {"final_output": final}
