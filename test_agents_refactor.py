"""
test_agents_refactor.py — Validates the Dynamic Scalability Patterns refactor.

Sections
--------
1. Structural tests  — import checks, state schema, graph topology (no API calls)
2. Unit tests        — mocked OpenAI, end-to-end graph execution
3. Live smoke test   — real API call (requires OPENAI_API_KEY + data)

Run all:        python test_agents_refactor.py
Run live only:  python test_agents_refactor.py live
"""

from __future__ import annotations

import sys
import json
import logging
import unittest
from unittest.mock import MagicMock, patch

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

PASS = "[PASS]"
FAIL = "[FAIL]"

results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    icon = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"  {icon}  {name}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        raise AssertionError(f"FAILED: {name}  {detail}")


# -----------------------------------------------------------------------------
# 1. STRUCTURAL TESTS — no network, no API
# -----------------------------------------------------------------------------

print("\n-- 1. Structural checks ---------------------------------------------")

# Principle 3 — _llm.py exists and exports the right names
from techa.agents._llm import invoke_structured, MODEL, _client
check("_llm.MODEL is gpt-4.1-nano",   MODEL == "gpt-4.1-nano")
check("_llm.invoke_structured callable", callable(invoke_structured))
check("_llm._client is OpenAI instance", _client.__class__.__name__ == "OpenAI")

# Principle 2 — WorkerResult defined in _common
from techa.agents._common import WorkerResult
wr: WorkerResult = {"agent_id": "test", "data": {"x": 1}, "error": None}
check("WorkerResult has agent_id/data/error keys",
      set(wr.keys()) == {"agent_id", "data", "error"})

# ta state — new fields present, old fields gone
from techa.agents.ta.graph_state import TechnicalAnalysisState
ta_hints = TechnicalAnalysisState.__annotations__
check("ta state has 'payload'",          "payload"   in ta_hints)
check("ta state has 'results'",          "results"   in ta_hints)
check("ta state has 'agent_id'",         "agent_id"  in ta_hints)
check("ta state no 'payload_json'",      "payload_json"    not in ta_hints)
check("ta state no 'breakout_result'",   "breakout_result" not in ta_hints)
check("ta state no 'ma_result'",         "ma_result"       not in ta_hints)

# patterns state — same checks
from techa.agents.patterns.graph_state import PatternScanState
ps_hints = PatternScanState.__annotations__
check("patterns state has 'payload'",    "payload"        in ps_hints)
check("patterns state has 'results'",    "results"        in ps_hints)
check("patterns state has 'agent_id'",   "agent_id"       in ps_hints)
check("patterns state no 'payload_json'","payload_json"   not in ps_hints)
check("patterns state no 'pattern_result'","pattern_result" not in ps_hints)

# Principle 1 — _subagents no longer exports build_subgraphs
import techa.agents.ta._subagents as ta_sub
import techa.agents.patterns._subagents as pat_sub
check("ta _subagents has WORKER_NAMES",        hasattr(ta_sub,  "WORKER_NAMES"))
check("ta _subagents no build_subgraphs",      not hasattr(ta_sub,  "build_subgraphs"))
check("pat _subagents has WORKER_NAMES",       hasattr(pat_sub, "WORKER_NAMES"))
check("pat _subagents no build_subgraphs",     not hasattr(pat_sub, "build_subgraphs"))
check("ta WORKER_NAMES == ['breakout','ma']",  ta_sub.WORKER_NAMES  == ["breakout", "ma"])
check("pat WORKER_NAMES == ['pattern']",       pat_sub.WORKER_NAMES == ["pattern"])

# Principle 4 — graph_nodes no longer import json for payload (json still imported for synthesise)
import techa.agents.ta.graph_nodes as ta_nodes
import techa.agents.patterns.graph_nodes as pat_nodes
check("ta graph_nodes has worker_node",        hasattr(ta_nodes,  "worker_node"))
check("pat graph_nodes has worker_node",       hasattr(pat_nodes, "worker_node"))
check("ta graph_nodes no create_subgraph",     not hasattr(ta_nodes,  "create_subgraph"))
check("pat graph_nodes no create_subgraph",    not hasattr(pat_nodes, "create_subgraph"))

# ask_* files no longer define MODEL or openai directly
import techa.agents.ta._tools.ask_bo_trader  as bo
import techa.agents.ta._tools.ask_ma_trader  as ma
import techa.agents.patterns._tools.ask_pattern_trader as pt
check("ask_bo_trader has no module-level MODEL",       not hasattr(bo, "MODEL") or bo.MODEL is MODEL)
check("ask_ma_trader has no module-level MODEL",       not hasattr(ma, "MODEL") or ma.MODEL is MODEL)
check("ask_pattern_trader has no module-level MODEL",  not hasattr(pt, "MODEL") or pt.MODEL is MODEL)

# Graph factory imports Send (not build_subgraphs)
import techa.agents.ta.agent as ta_agent
import techa.agents.patterns.agent as pat_agent
check("ta agent has _dispatcher",              hasattr(ta_agent,  "_dispatcher"))
check("pat agent has _dispatcher",             hasattr(pat_agent, "_dispatcher"))


# -----------------------------------------------------------------------------
# 2. UNIT TESTS — mocked OpenAI, end-to-end graph execution
# -----------------------------------------------------------------------------

print("\n-- 2. Unit tests (mocked OpenAI) ------------------------------------")


def _make_parsed_mock(model_cls, **overrides):
    """Return a MagicMock that walks like a parsed Pydantic object."""
    m = MagicMock(spec=model_cls)
    m.model_dump.return_value = {"mocked": True, "agent_id": overrides.get("agent_id", "unknown")}
    return m


# -- 2a. ta graph (breakout + ma workers, parallel Send) ----------------------

from techa.agents.ta.agent import create_manager
from techa.agents.ta._tools.ask_bo_trader import TraderAnalysis
from techa.agents.ta._tools.ask_ma_trader import MATraderAnalysis

_FAKE_DF_ROWS = 50

def _fake_load_analysis_data(path, symbol, analysis_date):
    import pandas as pd, numpy as np
    idx = pd.date_range("2024-01-01", periods=_FAKE_DF_ROWS, freq="B")
    df = pd.DataFrame({
        "date":   idx,
        "symbol": symbol,
        "rclose": np.linspace(1.0, 1.05, _FAKE_DF_ROWS),
        "ropen":  np.linspace(0.99, 1.04, _FAKE_DF_ROWS),
        "rhigh":  np.linspace(1.01, 1.06, _FAKE_DF_ROWS),
        "rlow":   np.linspace(0.98, 1.03, _FAKE_DF_ROWS),
    })
    return "2024-04-26", df

with (
    patch("techa.agents.ta.graph_nodes.load_analysis_data", side_effect=_fake_load_analysis_data),
    patch("techa.agents.ta.graph_nodes.bo_build_snapshot",  return_value={"bo": "snap"}),
    patch("techa.agents.ta.graph_nodes.ma_build_snapshot",  return_value={"ma": "snap"}),
    patch("techa.agents.ta.graph_nodes.ask_bo_trader",      return_value=_make_parsed_mock(TraderAnalysis,  agent_id="breakout")),
    patch("techa.agents.ta.graph_nodes.ask_ma_trader",      return_value=_make_parsed_mock(MATraderAnalysis, agent_id="ma")),
    patch("techa.agents.ta.graph_nodes._call_synthesis_llm", return_value="## Mocked TA Report"),
):
    g = create_manager("TEN.MI", data_source="parquet")
    state = g.invoke(g._initial_state)

check("ta graph: final_output present",          "final_output" in state)
check("ta graph: final_output is mocked report", state["final_output"] == "## Mocked TA Report")
check("ta graph: results list has 2 entries",    len(state.get("results", [])) == 2)
agent_ids = {r["agent_id"] for r in state["results"]}
check("ta graph: both workers ran",              agent_ids == {"breakout", "ma"})
check("ta graph: no worker errors",              all(r["error"] is None for r in state["results"]))
check("ta graph: payload is dict (not string)",  isinstance(state.get("payload"), dict))

# -- 2b. patterns graph (single worker, Send) ---------------------------------

from techa.agents.patterns.agent import create_pattern_agent
from techa.agents.patterns._tools.ask_pattern_trader import PatternScanAnalysis
import pandas as pd

_FAKE_HITS = pd.DataFrame(columns=["ticker", "date", "display_name", "signal"])

with (
    patch("techa.agents.patterns.graph_nodes.load_ohlcv_from_parquet",
          return_value=({"TEN.MI": pd.DataFrame(
              {"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.05]},
              index=pd.to_datetime(["2024-04-26"]),
          )}, "2024-04-26")),
    patch("techa.agents.patterns.graph_nodes.scan_last_bar",  return_value=_FAKE_HITS),
    patch("techa.agents.patterns.graph_nodes.scan_patterns",  return_value=_FAKE_HITS),
    patch("techa.agents.patterns.graph_nodes.ask_pattern_trader",
          return_value=_make_parsed_mock(PatternScanAnalysis, agent_id="pattern")),
):
    gp = create_pattern_agent(["TEN.MI"], data_source="parquet")
    state_p = gp.invoke(gp._initial_state)

check("patterns graph: final_output present",          "final_output" in state_p)
check("patterns graph: results list has 1 entry",      len(state_p.get("results", [])) == 1)
check("patterns graph: pattern worker ran",            state_p["results"][0]["agent_id"] == "pattern")
check("patterns graph: no worker errors",              state_p["results"][0]["error"] is None)
check("patterns graph: payload is dict (not string)",  isinstance(state_p.get("payload"), dict))

# -- 2c. add reducer accumulates, not overwrites -------------------------------

from operator import add as op_add
import typing, typing_extensions

# Simulate two parallel workers writing to results
r1 = [{"agent_id": "breakout", "data": {}, "error": None}]
r2 = [{"agent_id": "ma",       "data": {}, "error": None}]
merged = op_add(r1, r2)
check("add reducer merges two WorkerResult lists", len(merged) == 2)
check("add reducer preserves both agent_ids",
      {r["agent_id"] for r in merged} == {"breakout", "ma"})

# -- 2d. _dispatcher returns correct Send objects ------------------------------

from langgraph.types import Send
from techa.agents.ta.agent import _dispatcher as ta_dispatcher
from techa.agents.ta._subagents import WORKER_NAMES as ta_workers

fake_state: TechnicalAnalysisState = {"symbol": "TEN.MI", "data_source": "parquet"}
sends = ta_dispatcher(fake_state)
check("dispatcher returns list of Send",        all(isinstance(s, Send) for s in sends))
check("dispatcher length matches WORKER_NAMES", len(sends) == len(ta_workers))
check("dispatcher injects agent_id=breakout",   sends[0].arg.get("agent_id") == "breakout")
check("dispatcher injects agent_id=ma",         sends[1].arg.get("agent_id") == "ma")

# -- 2e. worker_node error path writes to results (never raises) ---------------

from techa.agents.ta.graph_nodes import worker_node

bad_state: TechnicalAnalysisState = {
    "agent_id": "breakout",
    "payload": {"symbol": "X", "breakout_snapshot": {}, "ma_snapshot": {}},
}
with patch("techa.agents.ta.graph_nodes.ask_bo_trader", side_effect=RuntimeError("boom")):
    out = worker_node(bad_state)
check("worker_node error path returns results list",  "results" in out)
check("worker_node error stored in error field",      out["results"][0]["error"] == "boom")
check("worker_node error path does not raise",        True)  # reaching here means no raise


# -----------------------------------------------------------------------------
# 3. LIVE SMOKE TEST (optional — requires OPENAI_API_KEY + data)
# -----------------------------------------------------------------------------

if "live" in sys.argv:
    import rich
    print("\n-- 3. Live smoke test -----------------------------------------------")

    print("\n  [patterns / live]")
    gp_live = create_pattern_agent(
        ["TEN.MI"],
        data_source="live",
        signal_filter="all",
        lookback_bars=40,
    )
    result_p = gp_live.invoke(gp_live._initial_state)
    check("live patterns: final_output non-empty", bool(result_p.get("final_output")))
    check("live patterns: results list populated", len(result_p.get("results", [])) >= 1)
    rich.print(result_p["final_output"])

    print("\n  [ta / live]")
    g_live = create_manager("TEN.MI", data_source="live")
    result_ta = g_live.invoke(g_live._initial_state)
    check("live ta: final_output non-empty", bool(result_ta.get("final_output")))
    check("live ta: both workers in results",
          {r["agent_id"] for r in result_ta.get("results", [])} == {"breakout", "ma"})
    rich.print(result_ta["final_output"])


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

total   = len(results)
passed  = sum(1 for _, ok, _ in results if ok)
failed  = total - passed

print(f"\n{'-'*60}")
print(f"  {PASS} {passed}/{total} passed" + (f"   {FAIL} {failed} failed" if failed else ""))
print(f"{'-'*60}\n")

sys.exit(0 if failed == 0 else 1)
