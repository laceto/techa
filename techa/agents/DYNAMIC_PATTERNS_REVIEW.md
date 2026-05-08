# Dynamic Scalability Patterns — Review & Upgrade

**Date:** 2026-05-08
**Project:** techa
**Agents directory:** `techa/agents/`

---

## Compliance Checklist

| Principle | Before | After |
|-----------|--------|-------|
| Send dispatch      | ❌ | ✅ |
| add accumulator    | ❌ | ✅ |
| Shared LLM helper  | ❌ | ✅ |
| Native state       | ❌ | ✅ |

---

## Findings

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

`WorkerResult` defined once in `_common.py` and imported by both state modules.
`synthesise_node` now iterates `state["results"]` generically; it never references a worker by field name.

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

**After:** New file `techa/agents/_llm.py` owns all of the above.

```python
# _llm.py
load_dotenv()
MODEL = "gpt-4.1-nano"
_client = openai.OpenAI()

def invoke_structured(schema, messages, max_tokens=1024):
    response = _client.beta.chat.completions.parse(...)
    return response.choices[0].message.parsed
```

Each `ask_*.py` file now imports `invoke_structured, MODEL` from `_llm` and calls:
```python
return invoke_structured(TraderAnalysis, messages, max_tokens=1024)
```

**Note:** `_DEFAULT_MODEL = "gpt-4o"` in `ta/graph_nodes.py` is intentionally kept separate — it drives the LangChain `ChatOpenAI` synthesis call, a different model/client stack from the worker calls.

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

State field changed from `payload_json: Annotated[str, _last]` to `payload: Annotated[Optional[dict], _last]`.

---

## Files Modified

| File | Change |
|------|--------|
| `techa/agents/_llm.py`                          | **Created** — centralized client, MODEL, invoke_structured |
| `techa/agents/_common.py`                       | Added `WorkerResult` TypedDict |
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
| `ask_bo_trader` | `(snapshot, ticker, question)` | ✅ unchanged |
| `ask_ma_trader` | `(snapshot, ticker, question)` | ✅ unchanged |
| `ask_pattern_trader` | `(payload, tickers, question)` | ✅ unchanged |
