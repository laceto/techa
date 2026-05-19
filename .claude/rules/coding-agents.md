# Coding Rules — `techa.agents`

## Context
You are working in `techa/agents/`. This subpackage is a LangGraph multi-agent system for AI-powered technical analysis and insurance risk assessment. It uses OpenAI structured output (`gpt-4.1-nano`) via `invoke_structured` from `techa.agents._llm`.

## Module layout

```
techa/agents/
├── schema.py                  WorkerResult TypedDict — canonical leaf import; no project imports
├── _common.py                 get_result_by_id(); RESULTS_PATH, HISTORY_BARS, _read_parquet_dated(); re-exports WorkerResult from schema.py
├── _llm.py                    Centralized OpenAI client: MODEL, SYNTHESIS_MODEL, _client, invoke_structured()
├── ta/                        Single-ticker TA agent (MA crossovers, breakouts)
│   ├── agent.py               create_manager() factory; _dispatcher() Send fan-out
│   ├── graph_state.py         TechnicalAnalysisState (TypedDict)
│   ├── graph_nodes.py         prepare_node, worker_node, synthesise_node
│   ├── _subagents.py          WORKER_NAMES, WORKER_REGISTRY, _run_breakout(), _run_ma()
│   └── _tools/
│       ├── prepare_tools.py   load_analysis_data(), load_live_data() [uses YFinanceDataHandler]
│       ├── ask_bo_trader.py
│       └── ask_ma_trader.py
├── patterns/                  Multi-ticker candlestick pattern scan agent
│   ├── agent.py               create_pattern_agent() factory; _dispatcher() Send fan-out
│   ├── graph_state.py         PatternScanState (TypedDict)
│   ├── graph_nodes.py         prepare_node, worker_node, synthesise_node
│   ├── _subagents.py          WORKER_NAMES, WORKER_REGISTRY, _run_pattern()
│   └── _tools/
│       ├── prepare_tools.py   load_ohlcv_from_parquet(), download_ohlcv_live() [uses YFinanceDataHandler]
│       └── ask_pattern_trader.py
├── indicators/                Single-ticker indicator analysis agent (trend, momentum, volatility)
│   ├── agent.py               create_indicator_agent() factory; _dispatcher() Send fan-out
│   ├── graph_state.py         IndicatorAnalysisState (TypedDict)
│   ├── graph_nodes.py         prepare_node, worker_node, synthesise_node
│   ├── _subagents.py          WORKER_NAMES = ["trend", "momentum", "volatility"]; WORKER_REGISTRY
│   └── _tools/
│       ├── prepare_tools.py   load_ohlcv_from_parquet(), download_ohlcv_live() [uses YFinanceDataHandler]
│       ├── ask_trend_analyst.py      TrendAnalysis schema + ask_trend_analyst()
│       ├── ask_momentum_analyst.py   MomentumAnalysis schema + ask_momentum_analyst()
│       └── ask_volatility_analyst.py VolatilityAnalysis schema + ask_volatility_analyst()
├── orchestrator/              Single-ticker orchestrator: shared OHLCV load + parallel fan-out
│   ├── agent.py               create_orchestrator() factory; _dispatcher() Send fan-out
│   ├── graph_state.py         OrchestratorState; raw_df channel + results accumulator
│   └── graph_nodes.py         prepare_node, runner_node, synthesise_node + _call_synthesis_llm
└── insurance/                 Insurance risk assessment agent (actuarial, accountant, medical, claims)
    ├── agent.py               create_insurance_agent() factory; _dispatcher() Send fan-out
    ├── graph_state.py         InsuranceAnalysisState (TypedDict)
    ├── graph_nodes.py         prepare_node, worker_node, synthesise_node
    ├── _subagents.py          WORKER_NAMES = ["actuarial", "accountant", "medical_underwriting", "claims_assessor"]; WORKER_REGISTRY
    └── _tools/
        ├── prepare_tools.py   build_payload() — builds 6 snapshots (kpi, medical, claims, ae, pricing, inforce)
        ├── ask_actuarial_analyst.py   ActuarialAnalysis schema + ask_actuarial_analyst()
        ├── ask_accountant.py          AccountingAnalysis schema + ask_accountant()
        ├── ask_medical_underwriter.py MedicalUnderwritingAnalysis schema + ask_medical_underwriter()
        └── ask_claims_assessor.py     ClaimsAssessment schema + ask_claims_assessor()
```

## Commands
```bash
# run the orchestrator (indicators + patterns in parallel, gpt-4o synthesis)
python -c "
from techa.agents.orchestrator import create_orchestrator
g = create_orchestrator('PST.MI')
r = g.invoke(g._initial_state)
print(r['final_output'])
"

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

# run an insurance risk assessment (demo profile)
python -c "
from techa.agents.insurance import create_insurance_agent
g = create_insurance_agent('POL-001')
r = g.invoke(g._initial_state)
print(r['final_output'])
"

# run with a custom risk profile
python -c "
from techa.agents.insurance import create_insurance_agent
profile = {'applicant': {'age': 55, 'gender': 'male', 'smoker': True, 'bmi': 31.0}}
g = create_insurance_agent('POL-002', risk_profile=profile)
r = g.invoke(g._initial_state)
print(r['final_output'])
"
```

## Architecture invariants

### State
- Scalar/dict fields use `Annotated[T, _last]` (keep most recent write).
- The `results` field uses `Annotated[list[WorkerResult], add]` — the `add` (list concatenation) reducer lets parallel `worker_node` invocations each append one `WorkerResult` without overwriting each other. Do **not** use `add` on any other field.
- `payload` (native `dict`) is the **sole data channel** from `prepare_node` to `worker_node` in the `ta`, `patterns`, `indicators`, and `insurance` agents. Workers never read from disk or network.
- In the `orchestrator`, the channel is `raw_df` — a serialised DataFrame (`df.reset_index().to_dict(orient="records")`) storing raw OHLCV with the DatetimeIndex preserved as a `"date"` column. `runner_node` reconstructs it via `pd.to_datetime` + `set_index("date")`.
- `relative` (`bool`, default `True`) is a caller-input field in both `TechnicalAnalysisState` and `OrchestratorState`. It is forwarded to `load_live_data(..., relative=relative)` which passes it to `generate_signals(...)`. In parquet mode the data is already relative, so `relative` is ignored.

### Node responsibilities
- `prepare_node` — loads OHLCV/parquet, computes scans (or stores raw OHLCV in `raw_df` for the orchestrator), stores result as native dict in `state["payload"]` (or `state["raw_df"]`).
- `worker_node` / `runner_node` (single shared node) — receives `agent_id` injected by `Send`; reads `state["payload"]` or reconstructs df from `state["raw_df"]`; dispatches to the appropriate function via `WORKER_REGISTRY[agent_id]`; appends one `WorkerResult` to `state["results"]`.
- `synthesise_node` — iterates `state["results"]` keyed by `agent_id`; formats to `final_output`. Never raises. The orchestrator's `synthesise_node` makes an additional LLM call via LangChain LCEL (`ChatPromptTemplate | ChatOpenAI`) using `gpt-4o` to produce the final structured markdown brief.

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

### `schema.py`
- `WorkerResult` — TypedDict `{"agent_id": str, "data": dict, "error": Optional[str]}`. This is the canonical definition. Import from here when you only need the type. `_common.py` re-exports it for backward compatibility.
- No project imports — `schema.py` is a leaf node in the import graph.

### `_common.py`
- Re-exports `WorkerResult` from `schema.py` (`from techa.agents.schema import WorkerResult`). Import `WorkerResult` from either location; `_common` is the legacy path.
- `get_result_by_id(results, agent_id)` — returns the first `WorkerResult` whose `agent_id` matches, or `None`. Use this in every `synthesise_node` instead of building a `results_by_id` dict manually.
- `RESULTS_PATH`: Path to `data/results/it/analysis_results.parquet`.
- `HISTORY_BARS = 300`: max rows loaded per ticker from parquet.
- `_read_parquet_dated(path, analysis_date)`: opens parquet, parses `date` column, applies `analysis_date` ceiling. Raises `FileNotFoundError` if path missing.

### `_llm.py`
- `MODEL = "gpt-4.1-nano"` — model for all `ask_*` structured-output worker calls.
- `SYNTHESIS_MODEL = "gpt-4o"` — model for `_call_synthesis_llm` in `ta`, `indicators`, and `orchestrator` nodes. Import this instead of declaring `_DEFAULT_MODEL` locally.
- `_client` — the single `openai.OpenAI()` instance. Do not instantiate `OpenAI()` anywhere else.
- `invoke_structured(schema, messages, max_tokens=1024)` — calls `_client.beta.chat.completions.parse` and returns the parsed Pydantic object.

Do not duplicate any of the above. Always import from `techa.agents._common` and `techa.agents._llm`.

## Insurance agent — snapshot pipeline

`build_payload()` in `insurance/_tools/prepare_tools.py` builds up to 6 pre-computed snapshots before workers run:

| Snapshot key | Source module | Input key in risk_profile |
|---|---|---|
| `kpi_snapshot` | `techa.insurance.build_kpi_snapshot` | `financial_history` (list of period dicts) |
| `medical_snapshot` | `techa.underwriting.build_medical_snapshot` | `applicant` dict |
| `claims_snapshot` | `techa.claims.build_claims_snapshot` | `claim_form` dict |
| `ae_snapshot` | `techa.actuarial.build_ae_snapshot` | `ae_data` (periods list with actual/expected) |
| `pricing_snapshot` | `techa.actuarial.build_pricing_snapshot` | `pricing_data` (cash_flows list) |
| `inforce_snapshot` | `techa.actuarial.build_inforce_snapshot` | `inforce_data` (periods list with PIF/GWP) |

Each snapshot builder follows the same pattern: `_adapter.py` validates/enriches, domain modules compute KPIs, `snapshot.py` orchestrates and returns a flat dict. All builders accept `nan_to_none=True` for JSON-safe output. Loading caps are applied only in `snapshot.py` (medical: 250%, claims: 100%).

Workers in `insurance/` receive `payload` (the full dict including all snapshots) and call their `ask_*` function directly — no additional snapshot building needed in `_subagents.py`.

## Adding a new worker agent
1. Create `_tools/ask_{name}_trader.py` — Pydantic schema + `ask_{name}_trader(payload, ...)` function that calls `invoke_structured`.
2. Add `"{name}"` to `WORKER_NAMES` and add a matching entry to `WORKER_REGISTRY` in `_subagents.py`. That is the **only** change needed in `_subagents.py` and `graph_nodes.py` requires **no edits** — the registry makes `worker_node` fully generic.
3. Read the new result with `get_result_by_id(state.get("results", []), "{name}")` in `synthesise_node`.

Do **not** add a new state field for the result — all worker output goes through `state["results"]`.

### Registry pattern for `_subagents.py`

For agents where all `ask_*` functions share the same signature `(payload, ticker=symbol)` (e.g. `indicators`):
```python
WORKER_REGISTRY: dict = {
    "trend":    ask_trend_analyst,
    "momentum": ask_momentum_analyst,
}
```

For agents where workers need snapshot building first (e.g. `ta`), use private wrappers:
```python
def _run_breakout(df: pd.DataFrame, symbol: str):
    from techa.breakout.bo_snapshot import build_snapshot as build
    return ask_bo_trader(build(df), ticker=symbol)

WORKER_REGISTRY: dict = {"breakout": _run_breakout}
```

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
