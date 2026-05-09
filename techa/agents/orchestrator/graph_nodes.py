"""
agents/orchestrator/graph_nodes.py — Node implementations for the Orchestrator graph.

Nodes:
  prepare_node    — loads raw OHLCV once for the symbol (parquet or live); serialises
                    the DataFrame into state["raw_df"] as a list of records so the
                    DatetimeIndex is preserved as a "date" column in each record.
                    Also loads the ta-enriched DataFrame and serialises it into
                    state["ta_df"] as a list of records.
  runner_node     — single shared node dispatched by Send with agent_id injected;
                    reconstructs the appropriate DataFrame from state and runs the
                    domain-specific logic for "indicators", "patterns", or "ta";
                    appends one WorkerResult to state["results"].
  synthesise_node — iterates state["results"] keyed by agent_id; calls an LLM to
                    format a combined markdown brief. Never raises.

Invariant: state["raw_df"] and state["ta_df"] are the sole data channels from
           prepare_node to runner_node — no runner reads from disk or network.
"""

from __future__ import annotations

import json
import logging
import time

import pandas as pd

from techa.agents._common import RESULTS_PATH
from techa.agents._llm import SYNTHESIS_MODEL
from techa.agents.orchestrator.graph_state import OrchestratorState

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 3 — Report compilation prompt
# ---------------------------------------------------------------------------

_REPORT_SYSTEM = """\
You are a senior technical analyst at a long/short equity proprietary trading desk.
You have received four independent technical assessments for {ticker}:
a trend analysis (MA alignment, slope, golden cross), a momentum analysis
(MACD, stochastic, ROC), a volatility & volume flow analysis (ATR regime,
Bollinger Bands, Chaikin oscillator), and a candlestick pattern scan
(last-bar hits and recent activity).

Your audience is a portfolio manager who needs actionable entry/exit conviction,
not a retail summary. Be direct, precise, and opinionated. Every claim must be
backed by a specific signal or metric from the assessments. Do not fabricate data.
If a figure is unavailable, say so explicitly.

---

## Position Recommendation

State clearly: LONG · SHORT · NEUTRAL — and the conviction level: High / Medium / Low.

Two to three paragraphs covering:
- The primary technical catalyst driving the recommended position.
- The key signal that would invalidate the thesis (stop-loss trigger).
- Suggested holding horizon: short-term (< 4 weeks), medium-term (1–3 months),
  or structural (> 3 months).

---

## Signal Confluence Scorecard

Summary table with columns:
| Dimension | Trend | Momentum | Volatility | TA | Patterns | Confluence |

Where Confluence is one of: ✅ Aligned · ⚠️ Mixed · 🔴 Diverging

Include these dimensions: Direction bias, Conviction level, Entry timing,
Volume/flow signal, Risk/stop level.

---

## Indicators Deep-Dive

### Trend (MA)
1. SMA alignment — full stack status (bullish/bearish/mixed), golden/death cross.
2. Slope — direction, quality (R²), distance from SMA20 and SMA50.
3. Trigger to watch — exact cross or price level that changes the call.

### Momentum
1. MACD — histogram value and direction, signal cross bias.
2. Stochastic — %K value, zone (overbought/oversold/neutral).
3. Rate of change — roc_20d and chg_5d alignment.
4. Trigger to watch — specific oscillator cross or ROC reversal.

### Volatility & Volume Flow
1. ATR regime — level (low/normal/high) and atr_pct value.
2. Bollinger Bands — %B position, squeeze status (bb_squeeze), band width.
3. Volume flow — accumulation/distribution/neutral from Chaikin oscillator.
4. Risk sizing — stop sizing implication from ATR level.

---

## TA Analysis (Breakout + MA Crossover)

### Breakout
1. Confluence — full_long / full_short / mixed / flat across rbo_20/50/150.
2. Range quality — setup (sideways?), compression, touch count.
3. Turtle signal — rtt_5020 alignment with rbo_20.
4. Stop levels — rlo_20 (short-term long stop), rhi_20 (short-term short stop).
5. Trigger to watch — specific signal flip that changes the call.

### MA Crossover
1. Triple confluence — rema_50100150 and rsma_50100150 agreement.
2. Trend strength — ADX level, RSI, MA gap slope.
3. Volume confirmation — vol_trend at and after the crossover flip.
4. Stop level — rema_50100150_stop_loss.
5. Trigger to watch — specific MA cross or ADX threshold to monitor.

---

## Candlestick Patterns
1. Last-bar hits — bullish/bearish count, pattern names, net bias.
2. Recent activity — activity trend (increasing/stable/decreasing), notable patterns.
3. Confluence with indicators — do patterns confirm or contradict the indicator bias?
4. Conviction — high (3+ same-direction) / medium / low.

---

## Entry & Exit Plan

- **Entry trigger**: specific signal flip or price event that confirms entry.
- **Stop-loss**: signal reversal or ATR level that invalidates the thesis.
- **First target**: nearest resistance or measured-move projection.

---

## Bottom Line

One paragraph: net technical conviction, single most important signal to monitor,
and the specific event or price level that would change the call.
"""

_REPORT_HUMAN = """\
Trend analysis:             {trend_analysis}

Momentum analysis:          {momentum_analysis}

Volatility/volume analysis: {volatility_analysis}

Candlestick patterns:       {patterns_analysis}

Breakout analysis:          {breakout_analysis}

MA crossover analysis:      {ma_analysis}
"""


# ---------------------------------------------------------------------------
# prepare_node
# ---------------------------------------------------------------------------


def prepare_node(state: OrchestratorState) -> dict:
    """
    Load raw OHLCV and the ta-enriched DataFrame for one symbol.

    Mode A (parquet): reuses load_ohlcv_from_parquet from the indicators prepare_tools
                      for raw OHLCV; uses load_analysis_data from the ta prepare_tools
                      for the enriched ta DataFrame.
    Mode B (live):    downloads via YFinanceDataHandler (same pipeline as all agents)
                      for raw OHLCV; uses load_live_data from the ta prepare_tools for
                      the enriched ta DataFrame (relative prices, signals, stop-losses).

    The raw OHLCV DataFrame is stored as df.reset_index().to_dict(orient="records") so
    the DatetimeIndex is preserved as a "date" column that runner_node can restore.
    The ta DataFrame is stored as df.to_dict(orient="records") (date is a plain column).

    Args:
        state: Must contain "symbol". Optional: "data_source", "analysis_date",
               "lookback_days", "benchmark", "fx".

    Returns:
        Dict updating "raw_df", "ta_df", and "resolved_date".

    Raises:
        ValueError: If the symbol cannot be found or returns no data.
        FileNotFoundError: If the parquet file is missing (parquet mode).
    """
    symbol        = state["symbol"]
    data_source   = state.get("data_source", "parquet")
    analysis_date = state.get("analysis_date")
    lookback_days = state.get("lookback_days", 365)
    benchmark     = state.get("benchmark", "FTSEMIB.MI")
    fx            = state.get("fx")

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

    # ── Load ta-enriched DataFrame ──────────────────────────────────────────
    if data_source == "live":
        from techa.agents.ta._tools.prepare_tools import load_live_data
        _, ta_df_raw = load_live_data(symbol, benchmark=benchmark, fx=fx)
    else:
        from techa.agents.ta._tools.prepare_tools import load_analysis_data
        _, ta_df_raw = load_analysis_data(RESULTS_PATH, symbol, analysis_date)

    ta_df = ta_df_raw.to_dict(orient="records")
    log.info("[prepare] ta_df serialised: %d rows", len(ta_df))

    return {
        "raw_df":        raw_df,
        "ta_df":         ta_df,
        "resolved_date": resolved_date,
    }


# ---------------------------------------------------------------------------
# runner_node — single shared node dispatched by Send
# ---------------------------------------------------------------------------


def runner_node(state: OrchestratorState) -> dict:
    """
    Run domain-specific logic for a single agent_id injected by the Send dispatcher.

    Reconstructs the appropriate DataFrame from state at the top of each branch:
    "indicators" and "patterns" use state["raw_df"]; "ta" uses state["ta_df"].
    Never raises — exceptions are caught and stored as a WorkerResult with error set,
    so the other dispatched runners can continue and synthesise_node always runs.

    Args:
        state: Must contain "agent_id" (injected by Send), "raw_df" (set by
               prepare_node for indicators/patterns), and "ta_df" (set by
               prepare_node for the ta runner).

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

        elif agent_id == "ta":
            from techa.agents.ta._subagents import WORKER_REGISTRY as TA_REGISTRY

            df = pd.DataFrame(state["ta_df"])
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

            bo_result = TA_REGISTRY["breakout"](df, symbol)
            ma_result = TA_REGISTRY["ma"](df, symbol)

            log.info("[runner] ta complete for %s", symbol)
            data = {
                "breakout": bo_result.model_dump(),
                "ma":       ma_result.model_dump(),
            }
            return {"results": [{"agent_id": "ta", "data": data, "error": None}]}

        else:
            raise ValueError(f"Unknown agent_id: {agent_id!r}")

    except Exception as exc:
        log.error("[runner] %s failed: %s", agent_id, exc, exc_info=True)
        return {"results": [{"agent_id": agent_id, "data": {}, "error": str(exc)}]}


# ---------------------------------------------------------------------------
# synthesise_node
# ---------------------------------------------------------------------------


def _call_synthesis_llm(
    ticker: str,
    trend_analysis: str,
    momentum_analysis: str,
    volatility_analysis: str,
    patterns_analysis: str,
    breakout_analysis: str,
    ma_analysis: str,
) -> str:
    """
    Call the LLM once to compile all six assessments into a final technical brief.

    Isolated into its own function so tests can mock the LLM call without
    wrestling with LangChain's LCEL pipe internals.

    Args:
        ticker:              Ticker symbol (e.g. "A2A.MI") — injected into the system prompt.
        trend_analysis:      JSON string of the trend sub-result (or "unavailable").
        momentum_analysis:   JSON string of the momentum sub-result (or "unavailable").
        volatility_analysis: JSON string of the volatility sub-result (or "unavailable").
        patterns_analysis:   JSON string of the patterns runner result (or "unavailable").
        breakout_analysis:   JSON string of the breakout sub-result (or "unavailable").
        ma_analysis:         JSON string of the MA crossover sub-result (or "unavailable").

    Returns:
        Markdown string containing the compiled report.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages([
        ("system", _REPORT_SYSTEM),
        ("human",  _REPORT_HUMAN),
    ])
    llm = ChatOpenAI(model=SYNTHESIS_MODEL, temperature=0)

    response = (prompt | llm).invoke({
        "ticker":              ticker,
        "trend_analysis":      trend_analysis,
        "momentum_analysis":   momentum_analysis,
        "volatility_analysis": volatility_analysis,
        "patterns_analysis":   patterns_analysis,
        "breakout_analysis":   breakout_analysis,
        "ma_analysis":         ma_analysis,
    })
    return response.content if hasattr(response, "content") else str(response)


def synthesise_node(state: OrchestratorState) -> dict:
    """
    Compile all assessments into a final technical brief via an LLM call.

    Never raises — missing or errored runner results are passed as "unavailable"
    to the LLM so it can still produce a partial report.

    Args:
        state: Must contain "symbol", "resolved_date", and "results".

    Returns:
        Dict updating "final_output" with the compiled markdown brief.
    """
    ticker        = state.get("symbol", "unknown")
    results_by_id = {r["agent_id"]: r for r in state.get("results", [])}

    def _fmt_indicators(r) -> tuple:
        """Return (trend_str, momentum_str, volatility_str) or three "unavailable" strings."""
        if not r:
            return "unavailable", "unavailable", "unavailable"
        if r.get("error"):
            msg = f"unavailable — {r['error']}"
            return msg, msg, msg
        data = r["data"]
        trend_str      = json.dumps(data.get("trend",      {}), indent=2)
        momentum_str   = json.dumps(data.get("momentum",   {}), indent=2)
        volatility_str = json.dumps(data.get("volatility", {}), indent=2)
        return trend_str, momentum_str, volatility_str

    def _fmt_patterns(r) -> str:
        if not r:
            return "unavailable"
        if r.get("error"):
            return f"unavailable — {r['error']}"
        return json.dumps(r["data"], indent=2)

    def _fmt_ta(r) -> tuple:
        """Return (breakout_str, ma_str) or two "unavailable" strings."""
        if not r:
            return "unavailable", "unavailable"
        if r.get("error"):
            msg = f"unavailable — {r['error']}"
            return msg, msg
        data = r["data"]
        return json.dumps(data.get("breakout", {}), indent=2), json.dumps(data.get("ma", {}), indent=2)

    trend_str, momentum_str, volatility_str = _fmt_indicators(
        results_by_id.get("indicators")
    )
    patterns_str = _fmt_patterns(results_by_id.get("patterns"))
    breakout_str, ma_str = _fmt_ta(results_by_id.get("ta"))

    log.info("[synthesise] generating report for %s", ticker)
    try:
        brief = _call_synthesis_llm(
            ticker,
            trend_str,
            momentum_str,
            volatility_str,
            patterns_str,
            breakout_str,
            ma_str,
        )
    except Exception as exc:
        log.error("[synthesise] LLM call failed: %s", exc, exc_info=True)
        raw = json.dumps(
            {r["agent_id"]: r for r in state.get("results", [])},
            indent=2,
            default=str,
        )
        brief = f"Synthesis failed: {exc}\n\nRaw results:\n{raw}"

    log.info("[synthesise] report generated (%d chars)", len(brief))
    return {"final_output": brief}
