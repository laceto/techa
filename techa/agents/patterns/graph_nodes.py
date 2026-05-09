"""
agents/patterns/graph_nodes.py — Node implementations for the PatternScan graph.

Nodes:
  prepare_node    — loads OHLCV for each ticker (parquet or live), calls scan_last_bar,
                    stores the result as a native dict in state["payload"].
  worker_node     — single shared node dispatched by Send with agent_id injected;
                    calls ask_pattern_trader and appends a WorkerResult.
  synthesise_node — reads state["results"], formats the structured output into a text report.

Invariant: state["payload"] is the sole data channel from prepare_node to worker_node.
           No worker reads from disk or network.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from techa.agents._common import RESULTS_PATH, get_result_by_id
from techa.agents.patterns.graph_state import PatternScanState
from techa.agents.patterns._subagents import WORKER_REGISTRY
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
    Load OHLCV for each ticker, run scan_last_bar, store payload as native dict.

    Mode A (parquet): reads ropen/rhigh/rlow/rclose from analysis_results.parquet,
                      anchored to analysis_date (or latest bar if None).
    Mode B (live):    downloads raw OHLCV via yfinance over the lookback_days window.

    Args:
        state: Must contain "tickers". Optional: "data_source", "analysis_date",
               "signal_filter", "lookback_days", "lookback_bars".

    Returns:
        Dict updating "payload" and "scan_date".
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
        "payload":   payload,
        "scan_date": scan_date,
    }


# ---------------------------------------------------------------------------
# worker_node — single shared node dispatched by Send
# ---------------------------------------------------------------------------


def worker_node(state: PatternScanState) -> dict:
    """
    Call ask_pattern_trader based on agent_id injected by the Send dispatcher.

    Never raises — exceptions are caught and stored as a WorkerResult with error set,
    so synthesise_node always runs.

    Args:
        state: Must contain "agent_id" (injected by Send) and "payload" (set by prepare_node).

    Returns:
        {"results": [WorkerResult]} — appended to state via the add reducer.
    """
    agent_id = state["agent_id"]
    payload  = state["payload"]
    tickers  = payload.get("tickers", [])

    try:
        worker_func = WORKER_REGISTRY.get(agent_id)
        if worker_func is None:
            raise ValueError(f"Unknown agent_id: {agent_id!r}")
        result = worker_func(payload)
        log.info("[worker] %s analysis complete (%d tickers)", agent_id, len(tickers))

        return {"results": [{"agent_id": agent_id, "data": result.model_dump(), "error": None}]}

    except Exception as exc:
        log.error("[worker] %s failed: %s", agent_id, exc, exc_info=True)
        return {"results": [{"agent_id": agent_id, "data": {}, "error": str(exc)}]}


# ---------------------------------------------------------------------------
# synthesise_node
# ---------------------------------------------------------------------------


def synthesise_node(state: PatternScanState) -> dict:
    """
    Format the pattern worker's structured result into a readable text report.

    Never raises — missing or errored results produce a graceful partial report.
    """
    tickers   = state.get("tickers", [])
    scan_date = state.get("scan_date", "unknown")

    pattern_r = get_result_by_id(state.get("results", []), "pattern")

    sep = "=" * 60

    if not pattern_r:
        return {"final_output": f"{sep}\n  Pattern scan returned no results.\n{sep}"}

    if pattern_r.get("error"):
        return {"final_output": f"{sep}\n  Pattern scan failed: {pattern_r['error']}\n{sep}"}

    pattern_result = pattern_r["data"]

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
