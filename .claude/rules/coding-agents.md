# Coding Rules — `techa.agents`

## Context
You are working in `techa/agents/`. This subpackage is a LangGraph multi-agent system for AI-powered technical analysis. It uses OpenAI structured output (`gpt-4.1-nano`) via `invoke_structured` from `techa.agents._llm`.

## Module layout

```
techa/agents/
├── _common.py                 WorkerResult TypedDict; RESULTS_PATH, HISTORY_BARS, _read_parquet_dated()
├── _llm.py                    Centralized OpenAI client: MODEL, _client, invoke_structured()
├── ta/                        Single-ticker TA agent (MA crossovers, breakouts)
│   ├── agent.py               create_manager() factory; _dispatcher() Send fan-out
│   ├── graph_state.py         TechnicalAnalysisState (TypedDict)
│   ├── graph_nodes.py         prepare_node, worker_node, synthesise_node
│   ├── _subagents.py          WORKER_NAMES only (no build_subgraphs)
│   └── _tools/
│       ├── prepare_tools.py   load_analysis_data(), load_live_data()
│       ├── ask_bo_trader.py
│       └── ask_ma_trader.py
├── patterns/                  Multi-ticker candlestick pattern scan agent
│   ├── agent.py               create_pattern_agent() factory; _dispatcher() Send fan-out
│   ├── graph_state.py         PatternScanState (TypedDict)
│   ├── graph_nodes.py         prepare_node, worker_node, synthesise_node
│   ├── _subagents.py          WORKER_NAMES only (no build_subgraphs)
│   └── _tools/
│       ├── prepare_tools.py   load_ohlcv_from_parquet(), download_ohlcv_live()
│       └── ask_pattern_trader.py
└── indicators/                Single-ticker indicator analysis agent (trend, momentum, volatility)
    ├── agent.py               create_indicator_agent() factory; _dispatcher() Send fan-out
    ├── graph_state.py         IndicatorAnalysisState (TypedDict)
    ├── graph_nodes.py         prepare_node, worker_node, synthesise_node
    ├── _subagents.py          WORKER_NAMES = ["trend", "momentum", "volatility"]
    └── _tools/
        ├── prepare_tools.py   load_ohlcv_from_parquet(), download_ohlcv_live()
        ├── ask_trend_analyst.py      TrendAnalysis schema + ask_trend_analyst()
        ├── ask_momentum_analyst.py   MomentumAnalysis schema + ask_momentum_analyst()
        └── ask_volatility_analyst.py VolatilityAnalysis schema + ask_volatility_analyst()
```

## Commands
```bash
# run an indicator analysis (live mode, default)
python -c "
from techa.agents.indicators import create_indicator_agent
g = create_indicator_agent('PST.MI')
r = g.invoke(g._initial_state)
print(r['final_output'])
"

# run a pattern scan (live mode)
python -c "
from techa.agents.patterns import create_pattern_agent
g = create_pattern_agent(['A2A.MI'], data_source='live')
r = g.invoke(g._initial_state)
print(r['final_output'])
"

# run a TA analysis (parquet mode)
python -c "
from techa.agents.ta import create_manager
g = create_manager('A2A.MI')
r = g.invoke(g._initial_state)
print(r['final_output'])
"
```

## Architecture invariants

### State
- Scalar/dict fields use `Annotated[T, _last]` (keep most recent write).
- The `results` field uses `Annotated[list[WorkerResult], add]` — the `add` (list concatenation) reducer lets parallel `worker_node` invocations each append one `WorkerResult` without overwriting each other. Do **not** use `add` on any other field.
- `payload` (native `dict`) is the **sole data channel** from `prepare_node` to `worker_node`. Workers never read from disk or network.

### Node responsibilities
- `prepare_node` — loads OHLCV/parquet, computes scans, stores result as native dict in `state["payload"]`.
- `worker_node` (single shared node) — receives `agent_id` injected by `Send`; reads `state["payload"]`; calls the appropriate `ask_*` function; appends one `WorkerResult` to `state["results"]`.
- `synthesise_node` — iterates `state["results"]` keyed by `agent_id`; formats to `final_output`. Never raises.

### Graph wiring (Send-based dynamic dispatch)
- Pattern: `START → prepare → (_dispatcher → Send) → worker_node → synthesise → END`
- TA: same shape; dispatcher emits two `Send` objects (breakout + ma) which run in parallel.

```python
def _dispatcher(state) -> list[Send]:
    return [Send("worker_node", {"agent_id": name, **state}) for name in WORKER_NAMES]

builder.add_conditional_edges("prepare", _dispatcher)
builder.add_edge("worker_node", "synthesise")
```

Adding a new worker requires only adding an entry to `WORKER_NAMES` — no edge wiring changes.

### Error handling
- `worker_node` catches all exceptions and returns `{"results": [{"agent_id": ..., "data": {}, "error": str(exc)}]}` so the graph always reaches `synthesise_node`.
- `synthesise_node` renders a graceful error message when `r["error"]` is set.

## Shared helpers

### `_common.py`
- `WorkerResult` — TypedDict `{"agent_id": str, "data": dict, "error": Optional[str]}`. Import this into every state module and `graph_nodes.py`.
- `RESULTS_PATH`: Path to `data/results/it/analysis_results.parquet`.
- `HISTORY_BARS = 300`: max rows loaded per ticker from parquet.
- `_read_parquet_dated(path, analysis_date)`: opens parquet, parses `date` column, applies `analysis_date` ceiling. Raises `FileNotFoundError` if path missing.

### `_llm.py`
- `MODEL = "gpt-4.1-nano"` — single source of truth for the model name.
- `_client` — the single `openai.OpenAI()` instance. Do not instantiate `OpenAI()` anywhere else.
- `invoke_structured(schema, messages, max_tokens=1024)` — calls `_client.beta.chat.completions.parse` and returns the parsed Pydantic object.

Do not duplicate any of the above. Always import from `techa.agents._common` and `techa.agents._llm`.

## Adding a new worker agent
1. Create `_tools/ask_{name}_trader.py` — Pydantic schema + `ask_{name}_trader(payload, ...)` function that calls `invoke_structured`.
2. Add `"{name}"` to `WORKER_NAMES` in `_subagents.py`. That is the **only** graph wiring change needed.
3. Add a branch for the new `agent_id` in `worker_node` in `graph_nodes.py`.
4. Read the new result from `results_by_id.get("{name}")` in `synthesise_node`.

Do **not** add a new state field for the result — all worker output goes through `state["results"]`.

## OpenAI structured output pattern
```python
from techa.agents._llm import invoke_structured, MODEL

return invoke_structured(
    MyPydanticModel,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ],
    max_tokens=1024,
)
```
`invoke_structured` calls `_client.beta.chat.completions.parse` and returns the parsed Pydantic object. Always use `MODEL = "gpt-4.1-nano"` — imported from `_llm`, not re-declared locally.

## Import path
Use `techa.` everywhere — `from techa.agents.patterns import create_pattern_agent`.

## Environment
`OPENAI_API_KEY` must be set. `load_dotenv()` is called once in `_llm.py` at import time — do **not** call it again in `ask_*.py` files.

## When done
→ STOP. Do not write tests here. Do not modify the router `CLAUDE.md`.
