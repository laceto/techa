# techa/agents — Full Source Code (pre-refactor snapshot)

> **Stale.** This file was generated before the Dynamic Scalability Patterns refactor.
> For the current architecture see `DYNAMIC_PATTERNS_REVIEW.md` and read the source files directly.

## Directory structure

```
techa/agents/
├── _common.py
├── ta/
│   ├── agent.py
│   ├── graph_state.py
│   ├── graph_nodes.py
│   ├── _subagents.py
│   └── _tools/
│       ├── prepare_tools.py
│       ├── ask_bo_trader.py
│       └── ask_ma_trader.py
└── patterns/
    ├── agent.py
    ├── graph_state.py
    ├── graph_nodes.py
    ├── _subagents.py
    └── _tools/
        ├── prepare_tools.py
        └── ask_pattern_trader.py
```

---

## `_common.py`

```python
"""
agents/_common.py — Shared constants and low-level helpers for all agent subpackages.

Imported by:
  techa.agents.ta._tools.prepare_tools
  techa.agents.patterns._tools.prepare_tools
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_PATH: Path = Path("data/results/it/analysis_results.parquet")
HISTORY_BARS: int  = 300   # max rows per ticker kept from parquet; enough for ADX(14)/MA(150)/RSI(14)


def _read_parquet_dated(path: Path, analysis_date: str | None) -> pd.DataFrame:
    """
    Open analysis_results.parquet, parse the date column, and apply an
    optional upper-bound date ceiling.

    Args:
        path:          Path to the parquet file.
        analysis_date: ISO date string ceiling (inclusive); None → no cutoff.

    Returns:
        Full DataFrame filtered to rows up to analysis_date.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])

    if analysis_date is not None:
        df = df[df["date"] <= pd.Timestamp(analysis_date)]

    return df
```

---

## `ta/agent.py`

```python
"""
agents/agent.py — create_manager() graph factory.

Builds the LangGraph StateGraph:
  START → prepare_node → [breakout_worker, ma_worker] → synthesise_node → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from techa.agents.ta.graph_state import TechnicalAnalysisState
from techa.agents.ta.graph_nodes import prepare_node, synthesise_node
from techa.agents.ta._subagents import WORKER_NAMES, build_subgraphs


def create_manager(
    symbol: str,
    analysis_date: str | None = None,
    data_source: str = "parquet",   # "parquet" | "live"
    benchmark: str = "FTSEMIB.MI",
    fx: str | None = None,
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
        checkpointer:  Optional LangGraph checkpointer for persistence / resumption.

    Returns:
        CompiledStateGraph ready for .invoke() / .stream().
    """
    builder = StateGraph(TechnicalAnalysisState)

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

    initial_state: TechnicalAnalysisState = {
        "symbol":        symbol,
        "analysis_date": analysis_date,
        "data_source":   data_source,
        "benchmark":     benchmark,
        "fx":            fx,
    }

    graph = builder.compile(checkpointer=checkpointer)
    graph._initial_state = initial_state
    return graph
```

---

## `ta/graph_state.py`

```python
"""
agents/graph_state.py — Typed state for the TechnicalAnalysis LangGraph.

Single source of truth for all fields that flow between nodes.
Every field uses the _last reducer so parallel branches can merge without conflict.
"""

from __future__ import annotations

from typing import Annotated, Optional

from typing_extensions import TypedDict


def _last(a, b):  # noqa: ANN001
    """Reducer: always keep the most recent value. Required for parallel fan-out/fan-in."""
    return b


class TechnicalAnalysisState(TypedDict, total=False):
    # ── Caller inputs ──────────────────────────────────────────────────────────
    symbol:        Annotated[str,           _last]  # single ticker to analyse (required)
    analysis_date: Annotated[Optional[str], _last]  # ISO date; None → latest bar
    data_source:   Annotated[str,           _last]  # "parquet" (default) | "live"
    benchmark:     Annotated[str,           _last]  # benchmark ticker
    # Mode A default: "FTSEMIB.MI" (excluded from result set)
    # Mode B default: user-supplied (used for calculate_relative_prices)
    fx:            Annotated[Optional[str], _last]  # FX ticker for currency conversion (e.g. "EURUSD=X")
    # None → no FX conversion (stock and benchmark share the same currency)

    # ── Set by prepare_node ────────────────────────────────────────────────────
    resolved_date: Annotated[str, _last]  # actual date resolved from the data
    payload_json:  Annotated[str, _last]
    # payload_json shape:
    # {
    #   "date":               "YYYY-MM-DD",
    #   "symbol":             str,
    #   "breakout_snapshot":  dict,   ← ta.breakout.bo_snapshot.build_snapshot()
    #   "ma_snapshot":        dict,   ← ta.ma.ma_snapshot.build_snapshot()
    # }

    # ── Set by subgraph workers ────────────────────────────────────────────────
    breakout_result: Annotated[Optional[dict], _last]  # TraderAnalysis.model_dump()
    ma_result:       Annotated[Optional[dict], _last]  # MATraderAnalysis.model_dump()
    # {"error": str} when the worker caught an exception

    # ── Set by synthesise_node ─────────────────────────────────────────────────
    final_output: Annotated[str, _last]  # both AI reports formatted side by side
```

---

## `ta/graph_nodes.py`

```python
"""
agents/graph_nodes.py — Node implementations for the TechnicalAnalysis graph.

Nodes:
  prepare_node         — loads data and builds breakout + MA snapshots for one symbol,
                         serialises everything to payload_json.
  create_subgraph()    — factory: returns a compiled single-node subgraph that calls
                         one AI trader (ask_bo_trader or ask_ma_trader).
  _call_synthesis_llm  — calls the LLM once to compile both AI reports into a final brief.
  synthesise_node      — formats inputs and delegates to _call_synthesis_llm.

Invariant: payload_json is the sole data channel from prepare_node to workers.
           No worker reads from disk or network.
"""

from __future__ import annotations

import json
import logging
import time

from langgraph.graph import END, START, StateGraph

from techa.agents._common import RESULTS_PATH
from techa.agents.ta.graph_state import TechnicalAnalysisState
from techa.agents.ta._tools.prepare_tools import load_analysis_data, load_live_data
from techa.agents.ta._tools.ask_bo_trader import ask_bo_trader
from techa.agents.ta._tools.ask_ma_trader import ask_ma_trader
from techa.breakout.bo_snapshot import build_snapshot as bo_build_snapshot
from techa.ma.ma_snapshot import build_snapshot as ma_build_snapshot

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o"

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
    Load data for one symbol, build breakout and MA snapshots, serialise to payload_json.

    Raises:
        ValueError: If the symbol cannot be found or snapshots cannot be built.
        FileNotFoundError: If the parquet file is missing (Mode A).
    """
    t0 = time.perf_counter()
    symbol        = state["symbol"]
    data_source   = state.get("data_source", "parquet")
    analysis_date = state.get("analysis_date")
    benchmark     = state.get("benchmark", "FTSEMIB.MI")
    fx            = state.get("fx")

    if data_source == "live":
        resolved_date, df = load_live_data(symbol, benchmark=benchmark, fx=fx)
    else:
        resolved_date, df = load_analysis_data(RESULTS_PATH, symbol, analysis_date)

    log.info("[prepare] symbol=%s resolved_date=%s rows=%d", symbol, resolved_date, len(df))

    breakout_snapshot = bo_build_snapshot(df)
    ma_snapshot       = ma_build_snapshot(df)

    log.info("[prepare] snapshots built in %.2fs", time.perf_counter() - t0)

    payload = {
        "date":              resolved_date,
        "symbol":            symbol,
        "breakout_snapshot": breakout_snapshot,
        "ma_snapshot":       ma_snapshot,
    }

    return {
        "payload_json":  json.dumps(payload),
        "resolved_date": resolved_date,
    }


# ---------------------------------------------------------------------------
# Subgraph factory
# ---------------------------------------------------------------------------


def create_subgraph(worker_name: str):
    """
    Build a compiled single-node subgraph that calls one AI trader.

    The subgraph never raises — exceptions are caught and returned as
    {"error": str(exc)} so the other worker can continue.

    Args:
        worker_name: "breakout" → calls ask_bo_trader; "ma" → calls ask_ma_trader.
    """
    result_key = f"{worker_name}_result"

    def run_worker(state: TechnicalAnalysisState) -> dict:
        try:
            payload = json.loads(state["payload_json"])
            symbol  = payload["symbol"]
            if worker_name == "breakout":
                snapshot = payload["breakout_snapshot"]
                result   = ask_bo_trader(snapshot, ticker=symbol)
                log.info("[%s] analysis complete for %s", worker_name, symbol)
                return {result_key: result.model_dump()}
            elif worker_name == "ma":
                snapshot = payload["ma_snapshot"]
                result   = ask_ma_trader(snapshot, ticker=symbol)
                log.info("[%s] analysis complete for %s", worker_name, symbol)
                return {result_key: result.model_dump()}
            else:
                raise ValueError(f"Unknown worker: {worker_name!r}")
        except Exception as exc:
            log.error("[%s] worker failed: %s", worker_name, exc, exc_info=True)
            return {result_key: {"error": str(exc)}}

    graph = StateGraph(TechnicalAnalysisState)
    graph.add_node("run_worker", run_worker)
    graph.add_edge(START, "run_worker")
    graph.add_edge("run_worker", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# synthesise_node
# ---------------------------------------------------------------------------


def _call_synthesis_llm(ticker: str, breakout_analysis: str, ma_analysis: str) -> str:
    """
    Call the LLM once to compile both AI reports into a final technical brief.

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
    llm = ChatOpenAI(model=_DEFAULT_MODEL, temperature=0)

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
    ticker          = state.get("symbol", "unknown")
    breakout_result = state.get("breakout_result") or {}
    ma_result       = state.get("ma_result") or {}

    def _fmt(result: dict) -> str:
        if not result:
            return "unavailable"
        if "error" in result:
            return f"unavailable — {result['error']}"
        return json.dumps(result, indent=2)

    bo_analysis = _fmt(breakout_result)
    ma_analysis = _fmt(ma_result)

    log.info("[synthesise] generating report for %s", ticker)
    brief = _call_synthesis_llm(ticker, bo_analysis, ma_analysis)
    log.info("[synthesise] report generated (%d chars)", len(brief))

    return {"final_output": brief}
```

---

## `ta/_subagents.py`

```python
"""
agents/_subagents.py — Worker registry.

WORKER_NAMES is the single source of truth for which strategies run.
Adding a new strategy = add one entry here; no wiring changes needed in agent.py.
"""

from __future__ import annotations

from techa.agents.ta.graph_nodes import create_subgraph

WORKER_NAMES: list[str] = ["breakout", "ma"]


def build_subgraphs() -> dict:
    """Compile one subgraph per worker name."""
    return {name: create_subgraph(name) for name in WORKER_NAMES}
```

---

## `ta/_tools/prepare_tools.py`

```python
"""
agents/_tools/prepare_tools.py — Data loading for both pipeline modes.

Mode A (parquet): reads analysis_results.parquet, filters to one symbol.
Mode B (live):    downloads via YFinanceDataHandler, enriches with signals and stops.

Public API:
    load_analysis_data(path, symbol, analysis_date) -> (resolved_date, df)
    load_live_data(symbol, benchmark, fx, config_path) -> (resolved_date, df)

Both return a tuple of (ISO-date-string, DataFrame) for the requested symbol.
The DataFrame is ready to pass to bo_snapshot.build_snapshot() / ma_snapshot.build_snapshot().
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from techa.agents._common import RESULTS_PATH, HISTORY_BARS, _read_parquet_dated

log = logging.getLogger(__name__)

BENCHMARK = "FTSEMIB.MI"


# ---------------------------------------------------------------------------
# Mode A — parquet
# ---------------------------------------------------------------------------


def load_analysis_data(
    path: Path,
    symbol: str,
    analysis_date: str | None,
) -> tuple[str, pd.DataFrame]:
    """
    Load a single symbol's history from the analysis parquet.

    Args:
        path:          Path to analysis_results.parquet.
        symbol:        Ticker to load (e.g. "A2A.MI").
        analysis_date: ISO date to anchor the snapshot window; None → latest bar.

    Returns:
        (resolved_date, df) where df contains at most HISTORY_BARS rows up to
        resolved_date, ready for build_snapshot().

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError:        If resolved_date is not in the parquet, or symbol not found.
    """
    log.info("[prepare_tools] loading %s", path)
    df_all = _read_parquet_dated(path, analysis_date)

    df = df_all[df_all["symbol"] == symbol].copy()

    if df.empty:
        raise ValueError(f"Symbol {symbol!r} not found in parquet.")

    dates = df["date"].dt.date
    if analysis_date is not None:
        target = pd.Timestamp(analysis_date).date()
        if target not in set(dates):
            available = sorted(set(dates))
            raise ValueError(
                f"analysis_date={analysis_date!r} not in parquet for {symbol!r}. "
                f"Available range: {available[0]} → {available[-1]}"
            )
        resolved = target
    else:
        resolved = dates.max()

    resolved_date = str(resolved)
    log.info("[prepare_tools] symbol=%s resolved_date=%s", symbol, resolved_date)

    df = df[df["date"].dt.date <= resolved]
    df = df.tail(HISTORY_BARS).copy()

    return resolved_date, df


# ---------------------------------------------------------------------------
# Mode B — live download
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _build_search_spaces(cfg: dict) -> tuple[dict, list, dict]:
    turtle = cfg["regimes"]["turtle"]
    tt_search_space = {
        "fast": [turtle["fast_window"]],
        "slow": [turtle["slow_window"]],
    }
    bo_search_space = [cfg["regimes"]["breakout"]["bo_window"]]
    ma = cfg["regimes"]["ma_crossover"]
    ma_search_space = {
        "short_ma":  [ma["short_window"]],
        "medium_ma": [ma["medium_window"]],
        "long_ma":   [ma["long_window"]],
    }
    return tt_search_space, bo_search_space, ma_search_space


def load_live_data(
    symbol: str,
    benchmark: str = BENCHMARK,
    fx: str | None = None,
    config_path: Path = Path("config.json"),
) -> tuple[str, pd.DataFrame]:
    """
    Download live OHLC for a single symbol, enrich with signals and stop-losses.

    Args:
        symbol:      Yahoo Finance ticker to analyse (e.g. "TCEHY").
        benchmark:   Benchmark ticker for calculate_relative_prices (default "FTSEMIB.MI").
        fx:          Optional FX ticker for currency conversion (e.g. "EURUSD=X").
                     Pass None when stock and benchmark share the same currency.
        config_path: Path to config.json for search-space parameters.

    Returns:
        (today_iso, df) where df is ready for build_snapshot().

    Raises:
        ImportError: If algoshort or pipeline modules are not installed.
        ValueError:  If the symbol could not be enriched.
    """
    from algoshort.yfinance_handler import YFinanceDataHandler
    from algoshort.ohlcprocessor import OHLCProcessor
    from algoshort.wrappers import generate_signals, calculate_return
    from algoshort.stop_loss import StopLossCalculator

    cfg = _load_config(config_path)
    tt_search_space, bo_search_space, ma_search_space = _build_search_spaces(cfg)

    today = date.today()
    ticker_list = list(dict.fromkeys([symbol, benchmark] + ([fx] if fx else [])))

    handler = YFinanceDataHandler(
        cache_dir="data/ohlc/it",
        enable_logging=False,
        chunk_size=20,
    )
    handler.download_data(
        symbols=ticker_list,
        start="2016-01-01",
        end=today,
        interval="1d",
        use_cache=False,
        threads=True,
    )

    stop_loss_cfg = cfg["stop_loss"]

    df_benchmark = handler.get_ohlc_data(benchmark)
    df_stock     = handler.get_ohlc_data(symbol)

    df_bm = df_benchmark.copy()

    if fx:
        df_fx = (
            handler.get_ohlc_data(fx)
            .reset_index()
            .rename(columns={"close": "fx_close"})[["date", "fx_close"]]
        )
        df_stock = df_stock.reset_index()
        df_stock = pd.merge(df_stock, df_fx, how="left", on="date")
        df_stock["fx_close"] = df_stock["fx_close"].ffill()
        if df_stock["fx_close"].isna().any():
            raise ValueError(
                f"FX series {fx!r} has leading NaN values with no prior rate to forward-fill."
            )
        for col in ("open", "high", "low", "close"):
            df_stock[col] = df_stock[col] / df_stock["fx_close"]
        df_stock = df_stock.drop(columns=["fx_close"])

    processor = OHLCProcessor()
    dfs = processor.calculate_relative_prices(df_stock, df_bm)
    dfs, signal_columns = generate_signals(
        df=dfs,
        config_path=str(config_path),
        tt_search_space=tt_search_space,
        bo_search_space=bo_search_space,
        ma_search_space=ma_search_space,
    )
    dfs = calculate_return(dfs, signal_columns)

    calc = StopLossCalculator(dfs)
    for sig in signal_columns:
        calc.data = calc.atr_stop_loss(
            signal=sig,
            window=stop_loss_cfg["atr_window"],
            multiplier=stop_loss_cfg["atr_multiplier"],
        )
    dfs = calc.data
    dfs["symbol"] = symbol

    return str(today), dfs.tail(HISTORY_BARS).copy()
```

---

## `ta/_tools/ask_bo_trader.py`

```python
"""
ask_bo_trader.py — Range breakout trader AI assistant.

Scope: range breakout analysis only.
    Moving average crossover signals (rema_*, rsma_*) are intentionally excluded.
    The turtle signal (rtt_5020) is kept as an independent price-channel breakout confirmation.

Environment:
    OPENAI_API_KEY must be set.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

from techa.breakout.bo_snapshot import RESULTS_PATH, build_snapshot_from_parquet

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class TimeframeAnalysis(BaseModel):
    signal: int = Field(description="Breakout signal value: +1 long, 0 exit/flat, -1 short.")
    signal_age: int = Field(description="Bars the signal has held its current value.")
    fresh_flip: bool = Field(description="True if the signal changed value on the last bar.")
    resistance: float = Field(description="Relative resistance level for this window.")
    support: float = Field(description="Relative support level for this window.")
    dist_to_resistance_pct: float = Field(description="% distance from rclose to resistance. Negative = already above.")
    dist_to_support_pct: float = Field(description="% distance from rclose to support.")
    momentum_pct: float | None = Field(description="% change in rclose over this window's lookback period.")
    commentary: str = Field(description="One sentence on the signal state and key level for this timeframe.")


class TurtleSignal(BaseModel):
    signal: int = Field(description="rtt_5020: +1 = broke above 20-day high, -1 = below 20-day low, 0 = inside channel.")
    aligned_with_rbo_20: bool = Field(description="True if rtt_5020 and rbo_20 have the same non-zero sign.")
    commentary: str = Field(description="One sentence on turtle signal and whether it agrees with rbo_20.")


class RiskLevels(BaseModel):
    long_stop:            float = Field(description="rlo_20 — short-term stop for a long.")
    long_structural_stop: float = Field(description="rlo_150 — major structural stop for longs.")
    short_stop:            float = Field(description="rhi_20 — short-term stop for a short.")
    short_structural_stop: float = Field(description="rhi_150 — major structural stop for shorts.")
    peak_resistance:       float = Field(description="rh4 — absolute highest swing high.")
    major_floor:           float = Field(description="rl4 — absolute deepest swing low.")


class VolatilityCompression(BaseModel):
    band_width_pct:      float = Field(description="(rhi_20 - rlo_20) / rclose * 100 at the last bar.")
    band_width_slope:    float = Field(description="OLS slope of band_width_pct over the last 40 bars (%/bar).")
    band_width_pct_rank: float = Field(description="Percentile rank of band_width_pct vs 252-bar history (0–100).")
    is_compressed:       bool  = Field(description="True when band_width_slope < 0 AND band_width_pct_rank < 25.")
    commentary:          str   = Field(description="One sentence on compression state.")


class RangeQuality(BaseModel):
    n_resistance_touches: int   = Field(description="Distinct times price tested rhi_20 without breaking through.")
    n_support_touches:    int   = Field(description="Distinct times price tested rlo_20 without breaking through.")
    is_sideways:          bool  = Field(description="True when OLS slope of rclose is below 0.15%/day.")
    slope_pct_per_day:    float = Field(description="Signed OLS slope of rclose (%/day).")
    consolidation_bars:   int   = Field(description="Total consecutive bars rbo_20 has held 0.")
    band_width_pct:       float = Field(description="(rhi_20 - rlo_20) / rclose * 100.")
    commentary:           str   = Field(description="One sentence assessing setup quality.")


class VolumeQuality(BaseModel):
    vol_trend_mean:     float       = Field(description="Mean vol_trend over the consolidation window.")
    vol_trend_slope:    float       = Field(description="OLS slope of vol_trend per bar.")
    is_quiet:           bool        = Field(description="True when vol_trend_mean < 1.0.")
    is_declining:       bool        = Field(description="True when vol_trend_slope < 0.")
    breakout_confirmed: bool | None = Field(description="True = flip bar with vol_trend >= 1.2. None = not a flip bar.")
    commentary:         str         = Field(description="One sentence on volume behaviour.")


class TraderAnalysis(BaseModel):
    description: str = Field(description="3-5 sentence narrative summary of the full technical picture.")
    regime: int = Field(description="rrg: +1 bullish, 0 sideways, -1 bearish.")
    confluence: Literal["full_long", "full_short", "mixed", "flat"] = Field(
        description="full_long: rbo_20/50/150 all +1. full_short: all -1. mixed: disagree. flat: all 0."
    )
    short_term: TimeframeAnalysis = Field(description="Analysis of the 20-day window.")
    medium_term: TimeframeAnalysis = Field(description="Analysis of the 50-day window.")
    long_term: TimeframeAnalysis = Field(description="Analysis of the 150-day window.")
    turtle: TurtleSignal = Field(description="Turtle price-channel breakout signal.")
    vol_trend: float = Field(description="Volume ratio vs 20-bar average.")
    range_quality: RangeQuality | None = Field(description="Range setup quality. null when ticker in trend.")
    volatility_compression: VolatilityCompression | None = Field(description="Volatility compression state. null when insufficient history.")
    volume_quality: VolumeQuality | None = Field(description="Volume behaviour during consolidation. null when unavailable.")
    risk: RiskLevels
    verdict: str = Field(description="Actionable one-sentence conclusion. Exact numbers.")


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

MODEL = "gpt-4.1-nano"


def ask_bo_trader(snapshot: dict, ticker: str, question: str | None = None) -> TraderAnalysis:
    """
    Send the ticker snapshot to an OpenAI model and return a structured analysis.

    Args:
        snapshot: Dict from build_snapshot — the last-bar data payload.
        ticker:   Ticker symbol string.
        question: Optional follow-up question.

    Returns:
        TraderAnalysis Pydantic model parsed directly from the model response.
    """
    client = openai.OpenAI()

    user_content = f"Ticker: {ticker}\n\nSnapshot:\n{json.dumps(snapshot, indent=2)}"
    if question:
        user_content += f"\n\nQuestion: {question}"

    log.info("Sending snapshot for %s to %s (%d fields)", ticker, MODEL, len(snapshot))

    response = client.beta.chat.completions.parse(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        response_format=TraderAnalysis,
    )

    return response.choices[0].message.parsed
```

---

## `ta/_tools/ask_ma_trader.py`

```python
"""
ask_ma_trader.py — Moving average crossover trader AI assistant.

Scope: MA crossover analysis only.
    Range breakout signals (rbo_*, rhi_*, rlo_*, rtt_5020) are intentionally excluded.

Environment:
    OPENAI_API_KEY must be set.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Literal

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from techa.ma.ma_snapshot import RESULTS_PATH, build_snapshot_from_parquet

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class MATimeframeAnalysis(BaseModel):
    ema_signal:       int   = Field(description="EMA signal: +1 long, 0 flat, -1 short.")
    sma_signal:       int   = Field(description="SMA signal (confirmation).")
    ema_sma_agree:    bool  = Field(description="True if EMA and SMA signal have the same non-zero sign.")
    signal_age:       int   = Field(description="Bars the EMA signal has held its current value.")
    fresh_flip:       bool  = Field(description="True if EMA signal changed on the last bar.")
    dist_fast_ma_pct: float = Field(description="% distance from rclose to fast MA. Negative = above MA.")
    dist_slow_ma_pct: float = Field(description="% distance from rclose to slow MA. Negative = above MA.")
    commentary:       str   = Field(description="One sentence on the signal state and key MA level context.")


class MATripleConfluence(BaseModel):
    ema_signal: int   = Field(description="rema_50100150: +1 all EMAs bullish, -1 all bearish, 0 mixed.")
    sma_signal: int   = Field(description="rsma_50100150: same for SMAs.")
    agree:      bool  = Field(description="True if EMA and SMA triple signals agree.")
    signal_age: int   = Field(description="Bars rema_50100150 has held its current value.")
    fresh_flip: bool  = Field(description="True if rema_50100150 changed on the last bar.")
    commentary: str   = Field(description="One sentence on triple confluence quality and conviction.")


class MATrendStrengthOutput(BaseModel):
    rsi:             float = Field(description="RSI (14-period) on rclose. 0–100.")
    adx:             float = Field(description="ADX at last bar. > 25 = institutional trend strength.")
    adx_slope:       float = Field(description="ADX slope (%/bar). Positive = strengthening.")
    adx_slope_r2:    float = Field(description="R² of the ADX slope OLS fit (0–1).")
    ma_gap_pct:      float = Field(description="(EMA50 - EMA150) / rclose * 100. MACD proxy.")
    ma_gap_slope:    float = Field(description="OLS slope of ma_gap_pct over 20 bars.")
    ma_gap_slope_r2: float = Field(description="R² of the MA gap slope OLS fit (0–1).")
    is_trending:     bool  = Field(description="True when ADX > 25 and ADX not declining.")
    commentary:      str   = Field(description="One sentence on trend quality.")


class MAVolumeQuality(BaseModel):
    vol_trend:        float       = Field(description="Current volume / 20-bar average.")
    vol_on_crossover: float | None = Field(description="vol_trend at the most recent crossover flip bar.")
    is_confirmed:     bool  | None = Field(description="True = flip bar with vol_trend >= 1.2.")
    is_sustained:     bool  | None = Field(description="True = post-flip vol mean >= 1.0.")
    commentary:       str          = Field(description="One sentence on crossover volume confirmation.")


class MARiskLevels(BaseModel):
    long_stop:             float = Field(description="rema_50100150_stop_loss — ATR stop for a long.")
    long_structural_stop:  float = Field(description="rl4 — deepest swing low.")
    short_stop:            float = Field(description="rema_50100150_stop_loss — ATR stop for a short.")
    short_structural_stop: float = Field(description="rh4 — peak resistance.")
    peak_resistance:       float = Field(description="rh4 — absolute highest swing high.")
    major_floor:           float = Field(description="rl4 — absolute deepest swing low.")


class MATraderAnalysis(BaseModel):
    description:       str                                              = Field(description="3-5 sentence narrative summary.")
    regime:            int                                              = Field(description="rrg: +1 bullish, 0 sideways, -1 bearish.")
    confluence:        Literal["full_long", "full_short", "mixed", "flat"] = Field(description="EMA/SMA triple confluence agreement.")
    short_term:        MATimeframeAnalysis  = Field(description="50/100 crossover analysis.")
    medium_term:       MATimeframeAnalysis  = Field(description="100/150 crossover analysis.")
    triple_confluence: MATripleConfluence   = Field(description="Triple EMA/SMA confluence analysis.")
    trend_strength:    MATrendStrengthOutput | None = Field(description="ADX/RSI/gap trend quality. null when insufficient history.")
    volume_quality:    MAVolumeQuality      = Field(description="Volume confirmation at and after crossover.")
    risk:              MARiskLevels         = Field(description="Stop-loss and swing level targets.")
    verdict:           str                  = Field(description="Actionable one-sentence conclusion. Exact numbers.")


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

MODEL = "gpt-4.1-nano"


def ask_ma_trader(snapshot: dict, ticker: str, question: str | None = None) -> MATraderAnalysis:
    """
    Send the ticker snapshot to an OpenAI model and return a structured MA analysis.

    Args:
        snapshot: Dict from build_snapshot — the last-bar MA data payload.
        ticker:   Ticker symbol string.
        question: Optional follow-up question.

    Returns:
        MATraderAnalysis Pydantic model parsed from the model response.
    """
    client = openai.OpenAI()

    user_content = f"Ticker: {ticker}\n\nSnapshot:\n{json.dumps(snapshot, indent=2)}"
    if question:
        user_content += f"\n\nQuestion: {question}"

    log.info("Sending MA snapshot for %s to %s (%d fields)", ticker, MODEL, len(snapshot))

    response = client.beta.chat.completions.parse(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        response_format=MATraderAnalysis,
    )

    return response.choices[0].message.parsed
```

---

## `patterns/agent.py`

```python
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
        data_source:   "parquet" (default) or "live".
        benchmark:     Accepted for API consistency; not used by pattern nodes.
        fx:            Accepted for API consistency; not used by pattern nodes.
        signal_filter: "all" (default), "bull" (+100 only), or "bear" (-100 only).
        lookback_days: Calendar days of OHLCV history to download (live mode only).
        lookback_bars: Trading bars of recent pattern history in the LLM payload.
        checkpointer:  Optional LangGraph checkpointer.

    Returns:
        CompiledStateGraph ready for .invoke() / .stream().
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
```

---

## `patterns/graph_state.py`

```python
"""
agents/patterns/graph_state.py — Typed state for the PatternScan LangGraph.

Every field uses the _last reducer so parallel branches can merge without conflict.
"""

from __future__ import annotations

from typing import Annotated, Optional

from typing_extensions import TypedDict


def _last(a, b):  # noqa: ANN001
    """Reducer: always keep the most recent value. Required for parallel fan-out/fan-in."""
    return b


class PatternScanState(TypedDict, total=False):
    # ── Caller inputs ──────────────────────────────────────────────────────────
    tickers:       Annotated[list[str],      _last]  # tickers to scan (required)
    signal_filter: Annotated[str,            _last]  # "all" | "bull" | "bear"
    data_source:   Annotated[str,            _last]  # "parquet" (default) | "live"
    analysis_date: Annotated[Optional[str],  _last]  # ISO date anchor; None → latest bar
    lookback_days: Annotated[int,            _last]  # calendar days of OHLCV history (live mode)
    lookback_bars: Annotated[int,            _last]  # trading bars of recent pattern history (default 20)
    benchmark:     Annotated[str,            _last]  # API consistency; not used by pattern nodes
    fx:            Annotated[Optional[str],  _last]  # API consistency; not used by pattern nodes

    # ── Set by prepare_node ────────────────────────────────────────────────────
    scan_date:    Annotated[str,            _last]
    payload_json: Annotated[str,            _last]
    # payload_json shape:
    # {
    #   "tickers":       list[str],
    #   "scan_date":     "YYYY-MM-DD",
    #   "signal_filter": str,
    #   "hits":          list[{"ticker", "date", "display_name", "signal"}],
    #   "total_hits":    int,
    #   "recent_hits":   list[{"ticker", "date", "display_name", "signal"}],
    #   "lookback_bars": int,
    # }

    # ── Set by pattern worker ──────────────────────────────────────────────────
    pattern_result: Annotated[Optional[dict], _last]  # PatternScanAnalysis.model_dump()

    # ── Set by synthesise_node ─────────────────────────────────────────────────
    final_output: Annotated[str, _last]
```

---

## `patterns/graph_nodes.py`

```python
"""
agents/patterns/graph_nodes.py — Node implementations for the PatternScan graph.

Nodes:
  prepare_node    — loads OHLCV per ticker, calls scan_last_bar, serialises to payload_json.
  create_subgraph — factory: returns a compiled single-node subgraph for ask_pattern_trader.
  synthesise_node — formats pattern_result into a readable final text report.

Invariant: payload_json is the sole data channel from prepare_node to workers.
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


def prepare_node(state: PatternScanState) -> dict:
    t0            = time.perf_counter()
    tickers       = state["tickers"]
    signal_filter = state.get("signal_filter", "all")
    data_source   = state.get("data_source", "parquet")
    analysis_date = state.get("analysis_date")
    lookback_days = state.get("lookback_days", _DEFAULT_LOOKBACK_DAYS)
    lookback_bars = state.get("lookback_bars", 20)

    if data_source == "parquet":
        ohlcv_by_ticker, scan_date = load_ohlcv_from_parquet(RESULTS_PATH, tickers, analysis_date)
    else:
        ohlcv_by_ticker, scan_date = download_ohlcv_live(tickers, lookback_days)

    hits_df = scan_last_bar(ohlcv_by_ticker, signal_filter=signal_filter)

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


def create_subgraph(worker_name: str):
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


def synthesise_node(state: PatternScanState) -> dict:
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

    return {"final_output": "\n".join(lines)}
```

---

## `patterns/_subagents.py`

```python
"""
agents/patterns/_subagents.py — Worker registry.

WORKER_NAMES is the single source of truth for which workers run.
"""

from __future__ import annotations

from techa.agents.patterns.graph_nodes import create_subgraph

WORKER_NAMES: list[str] = ["pattern"]


def build_subgraphs() -> dict:
    """Compile one subgraph per worker name."""
    return {name: create_subgraph(name) for name in WORKER_NAMES}
```

---

## `patterns/_tools/prepare_tools.py`

```python
"""
agents/patterns/_tools/prepare_tools.py — Data loading for the PatternScan agent.

Mode A (parquet): reads ropen/rhigh/rlow/rclose from analysis_results.parquet.
Mode B (live):    downloads raw OHLCV via yfinance per ticker.

Public API:
    load_ohlcv_from_parquet(path, tickers, analysis_date) -> (ohlcv_by_ticker, resolved_date)
    download_ohlcv_live(tickers, lookback_days)           -> (ohlcv_by_ticker, scan_date)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import yfinance as yf
from pathlib import Path

from techa.agents._common import RESULTS_PATH, HISTORY_BARS, _read_parquet_dated

log = logging.getLogger(__name__)


def load_ohlcv_from_parquet(
    path: Path,
    tickers: list[str],
    analysis_date: str | None,
) -> tuple[dict, str]:
    """
    Load OHLCV for each ticker from analysis_results.parquet.

    Uses ropen/rhigh/rlow/rclose columns, renamed to open/high/low/close.

    Returns:
        (ohlcv_by_ticker, resolved_date) — resolved_date is the latest bar across all tickers.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    df_all = _read_parquet_dated(path, analysis_date)

    ohlcv_by_ticker: dict = {}
    resolved_dates:  list = []

    for ticker in tickers:
        df = df_all[df_all["symbol"] == ticker].copy()
        if df.empty:
            log.warning("[prepare] ticker %s not found in parquet — skipped", ticker)
            continue

        df = df.sort_values("date").tail(HISTORY_BARS)
        resolved_dates.append(df["date"].iloc[-1])

        rename = {k: v for k, v in {
            "ropen":  "open",
            "rhigh":  "high",
            "rlow":   "low",
            "rclose": "close",
        }.items() if k in df.columns}
        df = df.rename(columns=rename).set_index("date").sort_index()

        keep = [c for c in ("open", "high", "low", "close") if c in df.columns]
        ohlcv_by_ticker[ticker] = df[keep]

    resolved = (
        max(resolved_dates).strftime("%Y-%m-%d")
        if resolved_dates
        else str(analysis_date or "unknown")
    )
    return ohlcv_by_ticker, resolved


def download_ohlcv_live(
    tickers: list[str],
    lookback_days: int = 365,
) -> tuple[dict, str]:
    """
    Download raw OHLCV for each ticker via yfinance.

    Returns:
        (ohlcv_by_ticker, scan_date) — scan_date is today's ISO date.
        Tickers that fail to download are silently skipped.
    """
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=lookback_days)
    start    = start_dt.strftime("%Y-%m-%d")
    end      = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    scan_date = end_dt.strftime("%Y-%m-%d")

    ohlcv_by_ticker: dict = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, multi_level_index=False, progress=False)
            if df.empty:
                log.warning("[prepare] no data returned for %s — skipped", ticker)
                continue
            ohlcv_by_ticker[ticker] = df
        except Exception as exc:
            log.warning("[prepare] download failed for %s: %s", ticker, exc)

    return ohlcv_by_ticker, scan_date
```

---

## `patterns/_tools/ask_pattern_trader.py`

```python
"""
ask_pattern_trader.py — Candlestick pattern scan AI assistant.

Input payload shape (from prepare_node):
    {
      "tickers":       list[str],
      "scan_date":     "YYYY-MM-DD",
      "signal_filter": "all" | "bull" | "bear",
      "hits":          [{"ticker", "date", "display_name", "signal"}, ...],
      "total_hits":    int,
      "recent_hits":   [{"ticker", "date", "display_name", "signal"}, ...],
      "lookback_bars": int,
    }
    signal values: +100 = bullish, -100 = bearish.

Environment:
    OPENAI_API_KEY must be set.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Literal

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class RecentPatternActivity(BaseModel):
    ticker:            str                                            = Field(description="Ticker symbol.")
    total_recent_hits: int                                            = Field(description="Total pattern hits in the last lookback_bars bars.")
    bullish_recent:    int                                            = Field(description="Bullish (+100) hits in the recent window.")
    bearish_recent:    int                                            = Field(description="Bearish (-100) hits in the recent window.")
    pattern_names:     list[str]                                      = Field(description="Distinct pattern display names in the recent window.")
    activity_trend:    Literal["increasing", "stable", "decreasing"] = Field(
        description="increasing: more hits in second half. decreasing: fewer. stable: roughly equal."
    )
    commentary:        str                                            = Field(description="One sentence: is activity building, fading, or steady?")


class TickerPatternSummary(BaseModel):
    ticker:           str                                            = Field(description="Ticker symbol.")
    date:             str                                            = Field(description="ISO date of the last bar that fired.")
    bullish_patterns: list[str]                                      = Field(description="Display names of bullish (+100) patterns.")
    bearish_patterns: list[str]                                      = Field(description="Display names of bearish (-100) patterns.")
    net_bias:         Literal["bullish", "bearish", "mixed", "none"] = Field(description="bullish / bearish / mixed / none.")
    conviction:       Literal["high", "medium", "low"]               = Field(
        description="high: 3+ same-direction. medium: 2 same-direction. low: 1 or mixed."
    )
    verdict:          str                                            = Field(description="One actionable sentence.")


class PatternScanAnalysis(BaseModel):
    description:      str                       = Field(description="3-5 sentence narrative summary of the full scan.")
    total_hits:       int                        = Field(description="Total pattern hits on the last bar.")
    bullish_count:    int                        = Field(description="Number of bullish (+100) last-bar hits.")
    bearish_count:    int                        = Field(description="Number of bearish (-100) last-bar hits.")
    recent_activity:  list[RecentPatternActivity] = Field(description="One entry per ticker in recent_hits.")
    ticker_summaries: list[TickerPatternSummary]  = Field(description="One entry per ticker with a last-bar hit.")
    top_actionable:   list[str]                  = Field(description="Tickers with medium or high conviction.")
    watchlist:        list[str]                  = Field(description="Tickers with low conviction or mixed signals.")
    summary:          str                        = Field(description="One sentence. Net market message from this scan.")


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

MODEL = "gpt-4.1-nano"


def ask_pattern_trader(
    payload: dict,
    tickers: list[str],
    question: str | None = None,
) -> PatternScanAnalysis:
    """
    Send the scan_last_bar payload to an OpenAI model and return a structured analysis.

    Args:
        payload:  Dict from prepare_node — the scan_last_bar results payload.
        tickers:  List of ticker symbols scanned.
        question: Optional follow-up question.

    Returns:
        PatternScanAnalysis Pydantic model parsed from the model response.
    """
    client = openai.OpenAI()

    user_content = (
        f"Tickers scanned: {', '.join(tickers)}\n\n"
        f"Scan payload:\n{json.dumps(payload, indent=2)}"
    )
    if question:
        user_content += f"\n\nQuestion: {question}"

    log.info("Sending pattern scan payload to %s — %d hits across %d tickers",
             MODEL, payload.get("total_hits", 0), len(tickers))

    response = client.beta.chat.completions.parse(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        response_format=PatternScanAnalysis,
    )

    return response.choices[0].message.parsed
```
