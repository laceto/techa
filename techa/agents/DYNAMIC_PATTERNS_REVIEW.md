# Dynamic Scalability Patterns — Review & Upgrade

**Date:** 2026-05-09
**Project:** techa
**Agents directory:** `techa/agents/`

---

## Compliance Checklist

| Principle | Before (session 1) | After (session 1) | Before (session 2) | After (session 2) |
|-----------|--------------------|-------------------|--------------------|-------------------|
| Send dispatch (P1)       | ❌ | ✅ | ✅ | ✅ |
| add accumulator (P2)     | ❌ | ✅ | ✅ | ✅ |
| Shared LLM helper (P3)   | ❌ | ✅ | ✅ | ✅ |
| Native state (P4)        | ❌ | ✅ | ✅ | ✅ |
| Shared schema module (P9) | — | — | ❌ | ✅ |
| Runtime config injection (P10) | — | — | ❌ | ✅ |

---

## Session 1 Findings (2026-05-08) — Principles 1–4

### Principle 1 — Dynamic Dispatch via `Send`

**Violation:** `ta/agent.py` and `patterns/agent.py` — static fan-out loops using `add_edge`.

**Before:**
```python
subgraphs = build_subgraphs()
for name, subgraph in subgraphs.items():
    builder.add_node(name, subgraph)
for name in WORKER_NAMES:
    builder.add_edge("prepare", name)
for name in WORKER_NAMES:
    builder.add_edge(name, "synthesise")
```

**After:**
```python
def _dispatcher(state) -> list[Send]:
    return [Send("worker_node", {"agent_id": name, **state}) for name in WORKER_NAMES]

builder.add_node("worker_node", worker_node)
builder.add_conditional_edges("prepare", _dispatcher)
builder.add_edge("worker_node", "synthesise")
```

**Impact:** Adding a new worker now requires only adding an entry to `WORKER_NAMES` in `_subagents.py`. Zero graph wiring changes.

---

### Principle 2 — Unified Result Accumulation via `add` reducer

**Violation:** `ta/graph_state.py` and `patterns/graph_state.py` — one state field per worker.

**Before:**
```python
breakout_result: Annotated[Optional[dict], _last]
ma_result:       Annotated[Optional[dict], _last]
# patterns:
pattern_result:  Annotated[Optional[dict], _last]
```

**After (both state TypeDicts):**
```python
results: Annotated[list[WorkerResult], add]
# WorkerResult = {"agent_id": str, "data": dict, "error": Optional[str]}
```

`WorkerResult` defined once in `_common.py` (now re-exported from `schema.py`) and imported by all state modules.
`synthesise_node` iterates `state["results"]` generically; it never references a worker by field name.

---

### Principle 3 — Centralized LLM Invocation

**Violation:** `ask_bo_trader.py`, `ask_ma_trader.py`, `ask_pattern_trader.py` — each instantiated `openai.OpenAI()`, defined `MODEL = "gpt-4.1-nano"`, and called `load_dotenv()` independently.

**Before (each file):**
```python
load_dotenv()
MODEL = "gpt-4.1-nano"
client = openai.OpenAI()
response = client.beta.chat.completions.parse(model=MODEL, ...)
return response.choices[0].message.parsed
```

**After:** `techa/agents/_llm.py` owns all of the above.

```python
# _llm.py
load_dotenv()
MODEL = "gpt-4.1-nano"
_client = openai.OpenAI()

def invoke_structured(schema, messages, max_tokens=1024):
    response = _client.beta.chat.completions.parse(...)
    return response.choices[0].message.parsed
```

Each `ask_*.py` file now imports `invoke_structured, MODEL` from `_llm`.

---

### Principle 4 — Native State (no JSON serialization)

**Violation:** `ta/graph_nodes.py` and `patterns/graph_nodes.py` — payload serialized to a JSON string for inter-node transport.

**Before:**
```python
# prepare_node
return {"payload_json": json.dumps(payload), ...}

# worker (inside create_subgraph)
payload = json.loads(state["payload_json"])
```

**After:**
```python
# prepare_node
return {"payload": payload, ...}   # native dict, no serialisation

# worker_node
payload = state["payload"]         # direct access, type-safe
```

---

## Session 2 Findings (2026-05-09) — Principles 9 and 10

### Principle 9 — Shared Schema Module

**Violation:** `WorkerResult` TypedDict was defined inline in `techa/agents/_common.py` (lines 20–24 before this session). There was no dedicated canonical schema module; `_common.py` mixed the type contract with constants and file I/O helpers.

**Before (`_common.py` lines 20–24):**
```python
class WorkerResult(TypedDict):
    """Standardized result envelope written by every worker_node."""
    agent_id: str
    data:     dict
    error:    Optional[str]
```

**After:** `WorkerResult` is defined in the new `techa/agents/schema.py`. `_common.py` re-exports it for backward compatibility so no import-site changes are needed across the codebase.

```python
# techa/agents/schema.py  (new file)
class WorkerResult(TypedDict):
    """Standardized result envelope written by every worker_node / runner_node."""
    agent_id: str
    data:     dict
    error:    Optional[str]

# techa/agents/_common.py  (updated)
from techa.agents.schema import WorkerResult  # noqa: F401
```

All four `graph_state.py` files continue to import `WorkerResult` from `techa.agents._common` and resolve correctly through the re-export chain.

---

### Principle 10 — Runtime Config Injection

**Violation:** `_DEFAULT_MODEL = "gpt-4o"` was hardcoded as a module-level constant in three separate files:
- `techa/agents/ta/graph_nodes.py` (line 31)
- `techa/agents/indicators/graph_nodes.py` (line 33)
- `techa/agents/orchestrator/graph_nodes.py` (line 32)

Changing the synthesis model required editing three files. This is the mirror of the per-file `MODEL = "gpt-4.1-nano"` duplication that P3 solved for worker tools.

**Before (three files, identical):**
```python
_DEFAULT_MODEL = "gpt-4o"
...
llm = ChatOpenAI(model=_DEFAULT_MODEL, temperature=0)
```

**After:** `SYNTHESIS_MODEL` is defined once in `techa/agents/_llm.py` alongside `MODEL`. All three synthesis nodes import it from there.

```python
# techa/agents/_llm.py  (updated)
MODEL = "gpt-4.1-nano"    # structured-output worker model (all ask_* tools)
SYNTHESIS_MODEL = "gpt-4o"   # synthesis LLM used in _call_synthesis_llm nodes

# ta/graph_nodes.py, indicators/graph_nodes.py, orchestrator/graph_nodes.py  (updated)
from techa.agents._llm import SYNTHESIS_MODEL
...
llm = ChatOpenAI(model=SYNTHESIS_MODEL, temperature=0)
```

Changing the synthesis model now requires editing exactly one line in `_llm.py`.

---

## Files Modified in Session 2

| File | Change |
|------|--------|
| `techa/agents/schema.py`                         | **Created** — canonical `WorkerResult` TypedDict definition |
| `techa/agents/_common.py`                        | Replaced inline `WorkerResult` definition with re-export from `schema.py`; removed unused `Optional` import |
| `techa/agents/_llm.py`                           | Added `SYNTHESIS_MODEL = "gpt-4o"` constant |
| `techa/agents/ta/graph_nodes.py`                 | Removed `_DEFAULT_MODEL`; import `SYNTHESIS_MODEL` from `_llm`; use it in `_call_synthesis_llm` |
| `techa/agents/indicators/graph_nodes.py`         | Same as above |
| `techa/agents/orchestrator/graph_nodes.py`       | Same as above |

---

## Files Modified in Session 1 (2026-05-08)

| File | Change |
|------|--------|
| `techa/agents/_llm.py`                          | **Created** — centralized client, MODEL, invoke_structured |
| `techa/agents/_common.py`                       | Added `WorkerResult` TypedDict, `get_result_by_id`, constants |
| `techa/agents/ta/graph_state.py`                | Replaced `payload_json`, `breakout_result`, `ma_result` with `payload`, `agent_id`, `results` |
| `techa/agents/patterns/graph_state.py`          | Replaced `payload_json`, `pattern_result` with `payload`, `agent_id`, `results` |
| `techa/agents/ta/graph_nodes.py`                | Removed `create_subgraph`; added `worker_node`; updated `prepare_node` and `synthesise_node` |
| `techa/agents/patterns/graph_nodes.py`          | Removed `create_subgraph`; added `worker_node`; updated `prepare_node` and `synthesise_node` |
| `techa/agents/ta/_subagents.py`                 | Removed `build_subgraphs`; kept `WORKER_NAMES` only |
| `techa/agents/patterns/_subagents.py`           | Removed `build_subgraphs`; kept `WORKER_NAMES` only |
| `techa/agents/ta/_tools/ask_bo_trader.py`       | Replaced `openai.OpenAI()` + `load_dotenv()` + `MODEL` with `invoke_structured` |
| `techa/agents/ta/_tools/ask_ma_trader.py`       | Same |
| `techa/agents/patterns/_tools/ask_pattern_trader.py` | Same; `max_tokens=2048` preserved |
| `techa/agents/ta/agent.py`                      | Replaced static fan-out with `Send`-based `_dispatcher` |
| `techa/agents/patterns/agent.py`                | Same |

---

## Public API — Unchanged Signatures

| Entry point | Signature | Status |
|-------------|-----------|--------|
| `techa.agents.ta.create_manager` | `(symbol, analysis_date, data_source, benchmark, fx, checkpointer)` | ✅ unchanged |
| `techa.agents.patterns.create_pattern_agent` | `(tickers, analysis_date, data_source, benchmark, fx, signal_filter, lookback_days, lookback_bars, checkpointer)` | ✅ unchanged |
| `techa.agents.indicators.create_indicator_agent` | `(symbol, analysis_date, data_source, lookback_days, checkpointer)` | ✅ unchanged |
| `techa.agents.orchestrator.create_orchestrator` | `(symbol, data_source, analysis_date, lookback_days, benchmark, fx, checkpointer)` | ✅ unchanged |
| `ask_bo_trader` | `(snapshot, ticker, question)` | ✅ unchanged |
| `ask_ma_trader` | `(snapshot, ticker, question)` | ✅ unchanged |
| `ask_pattern_trader` | `(payload, tickers, question)` | ✅ unchanged |
| `ask_trend_analyst` | `(payload, ticker, question)` | ✅ unchanged |
| `ask_momentum_analyst` | `(payload, ticker, question)` | ✅ unchanged |
| `ask_volatility_analyst` | `(payload, ticker, question)` | ✅ unchanged |
