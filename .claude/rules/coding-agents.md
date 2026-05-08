# Coding Rules — `techa.agents`

## Context
You are working in `techa/agents/`. This subpackage is a LangGraph multi-agent system for AI-powered technical analysis. It uses OpenAI structured output (`gpt-4.1-nano`) via `client.beta.chat.completions.parse`.

## Module layout

```
techa/agents/
├── _common.py                 Shared constants: RESULTS_PATH, HISTORY_BARS, _read_parquet_dated()
├── ta/                        Single-ticker TA agent (MA crossovers, breakouts, indicators)
│   ├── agent.py               create_manager() factory
│   ├── graph_state.py         TechnicalAnalysisState (TypedDict)
│   ├── graph_nodes.py         prepare_node, synthesise_node
│   ├── _subagents.py          WORKER_NAMES, build_subgraphs()
│   └── _tools/
│       ├── prepare_tools.py   load_analysis_data(), load_live_data()
│       ├── ask_breakout_trader.py
│       └── ask_ma_trader.py
└── patterns/                  Multi-ticker candlestick pattern scan agent
    ├── agent.py               create_pattern_agent() factory
    ├── graph_state.py         PatternScanState (TypedDict)
    ├── graph_nodes.py         prepare_node, synthesise_node
    ├── _subagents.py          WORKER_NAMES, build_subgraphs()
    └── _tools/
        ├── prepare_tools.py   load_ohlcv_from_parquet(), download_ohlcv_live()
        └── ask_pattern_trader.py
```

## Commands
```bash
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
- All state fields use `Annotated[T, _last]` reducer. Required for parallel fan-out/fan-in (multiple workers can write to the same state without conflict).
- `payload_json` is the **sole data channel** from `prepare_node` to workers. Workers never read from disk or network.

### Node responsibilities
- `prepare_node` — loads OHLCV/parquet, computes scans, serialises result to `payload_json`.
- Worker subgraphs — receive `payload_json` only; call OpenAI; write structured result to `{worker}_result`.
- `synthesise_node` — reads `{worker}_result`; formats to `final_output`. Never raises.

### Graph wiring
- Pattern: `START → prepare → pattern_worker → synthesise → END`
- TA: `START → prepare → [breakout_worker, ma_worker] → synthesise → END` (parallel fan-out)

### Error handling
- Worker subgraphs catch all exceptions and return `{"error": str(exc)}` so the graph always reaches `synthesise_node`.
- `synthesise_node` renders a graceful error message when `{"error": ...}` is detected.

## Shared helpers (_common.py)
- `RESULTS_PATH`: Path to `data/results/it/analysis_results.parquet`.
- `HISTORY_BARS = 300`: max rows loaded per ticker from parquet.
- `_read_parquet_dated(path, analysis_date)`: opens parquet, parses `date` column, applies `analysis_date` ceiling. Raises `FileNotFoundError` if path missing.

Do not duplicate these in module-level code. Always import from `techa.agents._common`.

## Adding a new worker agent
1. Create `_tools/ask_{name}_trader.py` — Pydantic schema + `ask_{name}_trader(payload, tickers)` function.
2. Add `"{name}"` to `WORKER_NAMES` in `_subagents.py`.
3. Add `{name}_result: Annotated[Optional[dict], _last]` to the state TypedDict.
4. Add the result key to `synthesise_node`.

## OpenAI structured output pattern
```python
response = client.beta.chat.completions.parse(
    model=MODEL,
    max_tokens=2048,
    messages=[...],
    response_format=MyPydanticModel,
)
return response.choices[0].message.parsed
```
Always use `MODEL = "gpt-4.1-nano"`. Do not change this without explicit instruction.

## Import path
Use `techa.` everywhere — `from techa.agents.patterns import create_pattern_agent`.

## Environment
`OPENAI_API_KEY` must be set. Use `load_dotenv()` at the top of every `_tools/ask_*.py` file.

## When done
→ STOP. Do not write tests here. Do not modify the router `CLAUDE.md`.
