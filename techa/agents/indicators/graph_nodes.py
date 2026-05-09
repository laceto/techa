"""
agents/indicators/graph_nodes.py — Node implementations for the IndicatorAnalysis graph.

Nodes:
  prepare_node        — downloads or loads OHLCV, calls build_snapshot, stores payload.
  worker_node         — single shared node dispatched by Send with agent_id injected;
                        calls ask_trend_analyst / ask_momentum_analyst /
                        ask_volatility_analyst and appends a WorkerResult.
  _call_synthesis_llm — compiles all three AI reports into a final technical brief.
  synthesise_node     — reads state["results"], formats inputs, delegates to LLM.

Invariant: state["payload"] is the sole data channel from prepare_node to worker_node.
           No worker reads from disk or network.
"""

from __future__ import annotations

import json
import logging
import time

from techa.agents._common import RESULTS_PATH, get_result_by_id
from techa.agents._llm import SYNTHESIS_MODEL
from techa.agents.indicators.graph_state import IndicatorAnalysisState
from techa.agents.indicators._tools.prepare_tools import (
    load_ohlcv_from_parquet,
    download_ohlcv_live,
)
from techa.agents.indicators._subagents import WORKER_REGISTRY
from techa.indicators import build_snapshot

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 3 — Report compilation prompt
# ---------------------------------------------------------------------------

_REPORT_SYSTEM = """\
You are a senior technical analyst at a long/short equity proprietary trading desk.
You have received three independent technical assessments for {ticker}:
a trend analysis, a momentum analysis, and a volatility & volume flow analysis.

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
| Dimension | Trend | Momentum | Volatility/Volume | Confluence |

Where Confluence is one of: ✅ Aligned · ⚠️ Mixed · 🔴 Diverging

Include these dimensions: Trend direction, Momentum bias, Volatility regime,
BB position, Volume flow, Entry timing.

---

## Trend Analysis Deep-Dive

1. **Verdict** — direction and conviction from the trend assessment.
2. **SMA alignment** — full stack (bullish/bearish/mixed), golden/death cross status.
3. **Slope quality** — slope direction, R² reliability, SMA20 distance.
4. **Key levels** — distance from SMA50 and SMA200.
5. **Trigger to watch** — the exact MA cross or price level that would change the call.

---

## Momentum Analysis Deep-Dive

1. **Verdict** — direction and conviction from the momentum assessment.
2. **MACD** — histogram direction, signal cross status.
3. **Stochastic** — zone (overbought/oversold/neutral), %K value.
4. **Rate of change** — roc_20d and chg_5d alignment.
5. **Trigger to watch** — the exact oscillator cross or ROC reversal to monitor.

---

## Volatility & Volume Flow Analysis Deep-Dive

1. **Verdict** — regime and conviction from the volatility assessment.
2. **ATR regime** — normalised ATR level (low/normal/high), historical vol comparison.
3. **Bollinger Bands** — position (%B), squeeze status, band width.
4. **Volume flow** — accumulation/distribution signal from Chaikin oscillator.
5. **Risk sizing** — ATR-based stop sizing implication.

---

## Entry & Exit Plan

- **Entry trigger**: the specific signal flip or price level that confirms entry.
- **Stop-loss**: the ATR level or MA level that invalidates the thesis.
- **First target**: nearest resistance or measured-move projection from band width.

---

## Bottom Line

One paragraph: net technical conviction, the single most important signal to
monitor, and the specific event or price level that would change the call.
"""

_REPORT_HUMAN = """\
Trend analysis:             {trend_analysis}

Momentum analysis:          {momentum_analysis}

Volatility/volume analysis: {volatility_analysis}
"""


# ---------------------------------------------------------------------------
# prepare_node
# ---------------------------------------------------------------------------


def prepare_node(state: IndicatorAnalysisState) -> dict:
    """
    Load OHLCV for one symbol, compute build_snapshot, store as native dict.

    Raises:
        ValueError: If the symbol cannot be found or snapshot cannot be built.
        FileNotFoundError: If the parquet file is missing (parquet mode).
    """
    t0 = time.perf_counter()
    symbol        = state["symbol"]
    data_source   = state.get("data_source", "parquet")
    analysis_date = state.get("analysis_date")
    lookback_days = state.get("lookback_days", 365)

    if data_source == "live":
        df, resolved_date = download_ohlcv_live(symbol, lookback_days=lookback_days)
    else:
        df, resolved_date = load_ohlcv_from_parquet(RESULTS_PATH, symbol, analysis_date)

    log.info("[prepare] symbol=%s resolved_date=%s rows=%d", symbol, resolved_date, len(df))

    snapshot = build_snapshot(df, nan_to_none=True)

    log.info("[prepare] snapshot built in %.2fs  keys=%d", time.perf_counter() - t0, len(snapshot))

    payload = {
        "symbol":   symbol,
        "date":     resolved_date,
        "snapshot": snapshot,
    }

    return {
        "payload":       payload,
        "resolved_date": resolved_date,
    }


# ---------------------------------------------------------------------------
# worker_node — single shared node dispatched by Send
# ---------------------------------------------------------------------------


def worker_node(state: IndicatorAnalysisState) -> dict:
    """
    Call the appropriate AI analyst based on agent_id injected by the Send dispatcher.

    Never raises — exceptions are caught and stored as a WorkerResult with error set,
    so the other dispatched workers can continue and synthesise_node always runs.

    Args:
        state: Must contain "agent_id" (injected by Send) and "payload" (set by prepare_node).

    Returns:
        {"results": [WorkerResult]} — appended to state via the add reducer.
    """
    agent_id = state["agent_id"]
    payload  = state["payload"]
    symbol   = payload["symbol"]

    try:
        worker_func = WORKER_REGISTRY.get(agent_id)
        if worker_func is None:
            raise ValueError(f"Unknown agent_id: {agent_id!r}")
        result = worker_func(payload, ticker=symbol)

        log.info("[worker] %s analysis complete for %s", agent_id, symbol)
        return {"results": [{"agent_id": agent_id, "data": result.model_dump(), "error": None}]}

    except Exception as exc:
        log.error("[worker] %s failed: %s", agent_id, exc, exc_info=True)
        return {"results": [{"agent_id": agent_id, "data": {}, "error": str(exc)}]}


# ---------------------------------------------------------------------------
# synthesise_node
# ---------------------------------------------------------------------------


def _call_synthesis_llm(
    ticker: str,
    trend_analysis: str,
    momentum_analysis: str,
    volatility_analysis: str,
) -> str:
    """
    Call the LLM once to compile all three AI reports into a final technical brief.

    Args:
        ticker:              Ticker symbol — injected into the system prompt.
        trend_analysis:      JSON string of the trend worker's result (or "unavailable").
        momentum_analysis:   JSON string of the momentum worker's result (or "unavailable").
        volatility_analysis: JSON string of the volatility worker's result (or "unavailable").

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
    })
    return response.content if hasattr(response, "content") else str(response)


def synthesise_node(state: IndicatorAnalysisState) -> dict:
    """
    Compile all three AI reports into a final technical brief via an LLM call.

    Never raises — missing or errored worker results are passed as "unavailable"
    to the LLM so it can still produce a partial report.
    """
    ticker = state.get("symbol", "unknown")

    def _fmt(agent_id: str) -> str:
        r = get_result_by_id(state.get("results", []), agent_id)
        if not r:
            return "unavailable"
        if r.get("error"):
            return f"unavailable — {r['error']}"
        return json.dumps(r["data"], indent=2)

    trend_analysis      = _fmt("trend")
    momentum_analysis   = _fmt("momentum")
    volatility_analysis = _fmt("volatility")

    log.info("[synthesise] generating report for %s", ticker)
    brief = _call_synthesis_llm(ticker, trend_analysis, momentum_analysis, volatility_analysis)
    log.info("[synthesise] report generated (%d chars)", len(brief))

    return {"final_output": brief}
