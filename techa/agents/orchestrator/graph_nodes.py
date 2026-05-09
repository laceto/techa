"""
agents/orchestrator/graph_nodes.py — Node implementations for the Orchestrator graph.

Nodes:
  prepare_node    — loads raw OHLCV once for the symbol (parquet or live); serialises
                    the DataFrame into state["raw_df"] as a list of records so the
                    DatetimeIndex is preserved as a "date" column in each record.
  runner_node     — single shared node dispatched by Send with agent_id injected;
                    reconstructs the DataFrame from state["raw_df"] and runs the
                    domain-specific logic for "indicators" or "patterns"; appends
                    one WorkerResult to state["results"].
  synthesise_node — iterates state["results"] keyed by agent_id; formats a
                    plain-text combined report. Never raises.

Invariant: state["raw_df"] is the sole data channel from prepare_node to runner_node —
           no runner reads from disk or network.
"""

from __future__ import annotations

import json
import logging
import time

import pandas as pd

from techa.agents._common import RESULTS_PATH
from techa.agents.orchestrator.graph_state import OrchestratorState

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# prepare_node
# ---------------------------------------------------------------------------


def prepare_node(state: OrchestratorState) -> dict:
    """
    Load raw OHLCV for one symbol and serialise it into state["raw_df"].

    Mode A (parquet): reuses load_ohlcv_from_parquet from the indicators prepare_tools.
    Mode B (live):    downloads via YFinanceDataHandler (same pipeline as all agents).

    The DataFrame is stored as df.reset_index().to_dict(orient="records") so the
    DatetimeIndex is preserved as a "date" column that runner_node can restore.

    Args:
        state: Must contain "symbol". Optional: "data_source", "analysis_date",
               "lookback_days".

    Returns:
        Dict updating "raw_df" and "resolved_date".

    Raises:
        ValueError: If the symbol cannot be found or returns no data.
        FileNotFoundError: If the parquet file is missing (parquet mode).
    """
    symbol        = state["symbol"]
    data_source   = state.get("data_source", "parquet")
    analysis_date = state.get("analysis_date")
    lookback_days = state.get("lookback_days", 365)

    if data_source == "live":
        from datetime import date, timedelta
        from algoshort.yfinance_handler import YFinanceDataHandler

        end_dt   = date.today()
        start_dt = end_dt - timedelta(days=lookback_days)
        handler  = YFinanceDataHandler(cache_dir="data/ohlc/it", enable_logging=False, chunk_size=20)
        handler.download_data(
            symbols=[symbol],
            start=str(start_dt),
            end=end_dt,
            interval="1d",
            use_cache=False,
            threads=True,
        )
        df = handler.get_ohlc_data(symbol)

        if df.empty:
            raise ValueError(
                f"YFinanceDataHandler returned no data for '{symbol}'. "
                f"Check the ticker symbol."
            )

        df.columns = df.columns.str.lower()
        resolved_date = df.index[-1].strftime("%Y-%m-%d")
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        df = df[keep]
    else:
        from techa.agents.indicators._tools.prepare_tools import load_ohlcv_from_parquet

        df, resolved_date = load_ohlcv_from_parquet(RESULTS_PATH, symbol, analysis_date)

    log.info("[prepare] symbol=%s resolved_date=%s rows=%d", symbol, resolved_date, len(df))

    raw_df = df.reset_index().to_dict(orient="records")

    return {
        "raw_df":        raw_df,
        "resolved_date": resolved_date,
    }


# ---------------------------------------------------------------------------
# runner_node — single shared node dispatched by Send
# ---------------------------------------------------------------------------


def runner_node(state: OrchestratorState) -> dict:
    """
    Run domain-specific logic for a single agent_id injected by the Send dispatcher.

    Reconstructs the DataFrame from state["raw_df"] at the top of each branch.
    Never raises — exceptions are caught and stored as a WorkerResult with error set,
    so the other dispatched runner can continue and synthesise_node always runs.

    Args:
        state: Must contain "agent_id" (injected by Send) and "raw_df"
               (set by prepare_node).

    Returns:
        {"results": [WorkerResult]} — appended to state via the add reducer.
    """
    agent_id      = state["agent_id"]
    symbol        = state.get("symbol", "unknown")
    resolved_date = state.get("resolved_date", "unknown")

    try:
        # Reconstruct the DataFrame from the serialised records in state
        df = pd.DataFrame(state["raw_df"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        if agent_id == "indicators":
            from techa.indicators import build_snapshot
            from techa.agents.indicators._tools.ask_trend_analyst     import ask_trend_analyst
            from techa.agents.indicators._tools.ask_momentum_analyst  import ask_momentum_analyst
            from techa.agents.indicators._tools.ask_volatility_analyst import ask_volatility_analyst

            t0       = time.perf_counter()
            snapshot = build_snapshot(df, nan_to_none=True)
            payload  = {"symbol": symbol, "date": resolved_date, "snapshot": snapshot}

            trend_r      = ask_trend_analyst(payload, ticker=symbol)
            momentum_r   = ask_momentum_analyst(payload, ticker=symbol)
            volatility_r = ask_volatility_analyst(payload, ticker=symbol)

            log.info(
                "[runner] indicators complete for %s in %.2fs",
                symbol,
                time.perf_counter() - t0,
            )

            data = {
                "trend":      trend_r.model_dump(),
                "momentum":   momentum_r.model_dump(),
                "volatility": volatility_r.model_dump(),
            }
            return {"results": [{"agent_id": "indicators", "data": data, "error": None}]}

        elif agent_id == "patterns":
            from techa.agents.patterns._tools.ask_pattern_trader import ask_pattern_trader
            from techa.patterns.scanner import scan_last_bar, scan_patterns

            lookback_bars   = 20
            ohlcv_by_ticker = {symbol: df}

            hits_df = scan_last_bar(ohlcv_by_ticker, signal_filter="all")

            recent_ohlcv = df.iloc[-lookback_bars:]
            recent_df    = scan_patterns(recent_ohlcv, signal_filter="all")

            hits_records = (
                hits_df.assign(
                    date=hits_df["date"].astype(str),
                    direction=hits_df["signal"].apply(
                        lambda s: "bullish" if s > 0 else "bearish"
                    ),
                ).to_dict(orient="records")
                if not hits_df.empty else []
            )

            recent_hits_records = []
            if not recent_df.empty:
                recent_hits_records = (
                    recent_df.assign(
                        ticker=symbol,
                        date=recent_df["date"].astype(str),
                        direction=recent_df["signal"].apply(
                            lambda s: "bullish" if s > 0 else "bearish"
                        ),
                    )[["ticker", "date", "display_name", "signal", "direction"]]
                    .to_dict(orient="records")
                )

            scan_payload = {
                "tickers":       [symbol],
                "scan_date":     resolved_date,
                "signal_filter": "all",
                "hits":          hits_records,
                "total_hits":    len(hits_df),
                "recent_hits":   recent_hits_records,
                "lookback_bars": lookback_bars,
            }

            result = ask_pattern_trader(scan_payload, tickers=[symbol])
            log.info("[runner] patterns complete for %s", symbol)
            return {"results": [{"agent_id": "patterns", "data": result.model_dump(), "error": None}]}

        else:
            raise ValueError(f"Unknown agent_id: {agent_id!r}")

    except Exception as exc:
        log.error("[runner] %s failed: %s", agent_id, exc, exc_info=True)
        return {"results": [{"agent_id": agent_id, "data": {}, "error": str(exc)}]}


# ---------------------------------------------------------------------------
# synthesise_node
# ---------------------------------------------------------------------------


def synthesise_node(state: OrchestratorState) -> dict:
    """
    Format a plain-text combined report from the indicators and patterns runner results.

    Never raises — missing or errored runner results produce a graceful partial report.

    Args:
        state: Must contain "symbol", "resolved_date", and "results".

    Returns:
        Dict updating "final_output" with the combined report string.
    """
    symbol        = state.get("symbol", "unknown")
    resolved_date = state.get("resolved_date", "unknown")
    results_by_id = {r["agent_id"]: r for r in state.get("results", [])}
    sep = "=" * 60

    def _section(agent_id: str, label: str) -> str:
        r = results_by_id.get(agent_id)
        if not r:
            return f"  {label}: no results returned."
        if r.get("error"):
            return f"  {label}: failed — {r['error']}"
        return f"  {label}:\n{json.dumps(r['data'], indent=4)}"

    lines = [
        sep,
        f"  Combined Analysis — {symbol}  ({resolved_date})",
        sep,
        "",
        _section("indicators", "Indicators (trend / momentum / volatility)"),
        "",
        _section("patterns", "Candlestick Patterns"),
        "",
        sep,
    ]

    final = "\n".join(lines)
    log.info("[synthesise] report generated (%d chars)", len(final))
    return {"final_output": final}
