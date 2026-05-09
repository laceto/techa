"""
agents/graph_nodes.py — Node implementations for the TechnicalAnalysis graph.

Nodes:
  prepare_node        — loads raw data for one symbol and stores the raw DataFrame
                        as a serialised dict in state["payload"]["raw_df"].
  worker_node         — single shared node; dispatched by Send with agent_id injected;
                        reconstructs the DataFrame, builds its own snapshot, calls
                        ask_bo_trader or ask_ma_trader, and appends a WorkerResult.
  _call_synthesis_llm — calls the LLM once to compile both AI reports into a final brief.
  synthesise_node     — reads state["results"], formats inputs, delegates to _call_synthesis_llm.

Invariant: state["payload"] is the sole data channel from prepare_node to worker_node.
           No worker reads from disk or network.
"""

from __future__ import annotations

import json
import logging
import time

from techa.agents._common import RESULTS_PATH, get_result_by_id
from techa.agents._llm import SYNTHESIS_MODEL
from techa.agents.ta.graph_state import TechnicalAnalysisState
from techa.agents.ta._tools.prepare_tools import load_analysis_data, load_live_data
from techa.agents.ta._subagents import WORKER_REGISTRY

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 3 — Report compilation prompt
# ---------------------------------------------------------------------------

_REPORT_SYSTEM = """\
You are a senior technical analyst at a long/short equity proprietary trading desk.
You have received two independent technical assessments for {ticker}:
a range-breakout analysis and an MA-crossover analysis.

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
| Dimension | Breakout | MA Crossover | Confluence |

Where Confluence is one of: ✅ Aligned · ⚠️ Mixed · 🔴 Diverging

Include these dimensions: Trend direction, Regime (rrg), Entry timing, \
Volume confirmation, Risk/Stop level.

---

## Breakout Analysis Deep-Dive

1. **Verdict** — direction and conviction from the breakout assessment.
2. **Key signals** — bullet list of every active signal, age, and flip status.
3. **Range quality** — range setup, volatility compression, touch count.
4. **Volume** — breakout volume confirmation or consolidation quiet.
5. **Stop level** — the specific stop-loss level from the assessment.
6. **Trigger to watch** — the exact signal or price event that would change the call.

---

## MA Crossover Analysis Deep-Dive

1. **Verdict** — direction and conviction from the MA assessment.
2. **Key signals** — EMA/SMA alignment, triple-confluence status, signal age/flip.
3. **Trend quality** — ADX level and slope, RSI, MA gap and gap slope.
4. **Volume** — crossover volume confirmation and post-crossover sustainability.
5. **Stop level** — the specific stop-loss level from the assessment.
6. **Trigger to watch** — the exact signal or metric that would change the call.

---

## Entry & Exit Plan

- **Entry trigger**: the specific signal flip or price level that confirms entry.
- **Stop-loss**: the signal reversal or price level that invalidates the thesis.
- **First target**: the nearest resistance or measured-move projection.

---

## Bottom Line

One paragraph: net technical conviction, the single most important signal to
monitor, and the specific event or price level that would change the call.
"""

_REPORT_HUMAN = """\
Breakout analysis:    {breakout_analysis}

MA crossover analysis: {ma_analysis}
"""


# ---------------------------------------------------------------------------
# prepare_node
# ---------------------------------------------------------------------------


def prepare_node(state: TechnicalAnalysisState) -> dict:
    """
    Load raw data for one symbol and store the DataFrame as a serialised dict in payload.

    Raises:
        ValueError: If the symbol cannot be found in the data.
        FileNotFoundError: If the parquet file is missing (Mode A).
    """
    t0 = time.perf_counter()
    symbol        = state["symbol"]
    data_source   = state.get("data_source", "parquet")
    analysis_date = state.get("analysis_date")
    benchmark     = state.get("benchmark", "FTSEMIB.MI")
    fx            = state.get("fx")
    relative      = state.get("relative", False)

    if data_source == "live":
        resolved_date, df = load_live_data(symbol, benchmark=benchmark, fx=fx, relative=relative)
    else:
        if relative is not False:
            log.warning("[prepare] relative=%s has no effect in parquet mode — data is already relative", relative)
        resolved_date, df = load_analysis_data(RESULTS_PATH, symbol, analysis_date)

    log.info("[prepare] symbol=%s resolved_date=%s rows=%d", symbol, resolved_date, len(df))
    log.info("[prepare] raw_df stored, %.2fs", time.perf_counter() - t0)

    payload = {
        "date":   resolved_date,
        "symbol": symbol,
        "raw_df": df.to_dict(orient="records"),
    }

    return {
        "payload":       payload,
        "resolved_date": resolved_date,
    }


# ---------------------------------------------------------------------------
# worker_node — single shared node dispatched by Send
# ---------------------------------------------------------------------------


def worker_node(state: TechnicalAnalysisState) -> dict:
    """
    Call the appropriate AI trader based on agent_id injected by the Send dispatcher.

    Never raises — exceptions are caught and stored as a WorkerResult with error set,
    so the other dispatched worker can continue and synthesise_node always runs.

    Args:
        state: Must contain "agent_id" (injected by Send) and "payload" (set by prepare_node).

    Returns:
        {"results": [WorkerResult]} — appended to state via the add reducer.
    """
    agent_id = state["agent_id"]
    payload  = state["payload"]
    symbol   = payload["symbol"]

    try:
        import pandas as pd
        df = pd.DataFrame(payload["raw_df"])
        worker_func = WORKER_REGISTRY.get(agent_id)
        if worker_func is None:
            raise ValueError(f"Unknown agent_id: {agent_id!r}")
        result = worker_func(df, symbol)
        log.info("[worker] %s analysis complete for %s", agent_id, symbol)

        return {"results": [{"agent_id": agent_id, "data": result.model_dump(), "error": None}]}

    except Exception as exc:
        log.error("[worker] %s failed: %s", agent_id, exc, exc_info=True)
        return {"results": [{"agent_id": agent_id, "data": {}, "error": str(exc)}]}


# ---------------------------------------------------------------------------
# synthesise_node
# ---------------------------------------------------------------------------


def _call_synthesis_llm(ticker: str, breakout_analysis: str, ma_analysis: str) -> str:
    """
    Call the LLM once to compile both AI reports into a final technical brief.

    Isolated into its own function so tests can mock the LLM call without
    wrestling with LangChain's LCEL pipe internals.

    Args:
        ticker:            Ticker symbol (e.g. "A2A.MI") — injected into the system prompt.
        breakout_analysis: JSON string of the breakout worker's result (or "unavailable").
        ma_analysis:       JSON string of the MA worker's result (or "unavailable").

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
        "ticker":             ticker,
        "breakout_analysis":  breakout_analysis,
        "ma_analysis":        ma_analysis,
    })
    return response.content if hasattr(response, "content") else str(response)


def synthesise_node(state: TechnicalAnalysisState) -> dict:
    """
    Compile both AI reports into a final technical brief via an LLM call.

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

    bo_analysis = _fmt("breakout")
    ma_analysis = _fmt("ma")

    log.info("[synthesise] generating report for %s", ticker)
    brief = _call_synthesis_llm(ticker, bo_analysis, ma_analysis)
    log.info("[synthesise] report generated (%d chars)", len(brief))

    return {"final_output": brief}
