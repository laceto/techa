import rich

symbol    = "TGYM.MI"
benchmark = "FTSEMIB.MI"
date      = "2026-05-09"

# ---------------------------------------------------------------------------
# 1. TA agent — MA crossovers + range breakouts (relative-price, parquet or live)
# ---------------------------------------------------------------------------
# from techa.agents.ta import create_manager

# graph  = create_manager(symbol=symbol, data_source="live", benchmark=benchmark)
# result = graph.invoke(graph._initial_state)
# rich.print("[bold cyan]── TA agent ──[/bold cyan]")
# rich.print(result["final_output"])

# ---------------------------------------------------------------------------
# 2. Indicators agent — trend / momentum / volatility (raw OHLCV, live or parquet)
# ---------------------------------------------------------------------------
# from techa.agents.indicators import create_indicator_agent
#
# graph  = create_indicator_agent(symbol, data_source="live")
# result = graph.invoke(graph._initial_state)
# rich.print("[bold green]── Indicators agent ──[/bold green]")
# rich.print(result["final_output"])

# ---------------------------------------------------------------------------
# 3. Patterns agent — multi-ticker candlestick scan (live or parquet)
# ---------------------------------------------------------------------------
# from techa.agents.patterns import create_pattern_agent
#
# graph  = create_pattern_agent(
#     [symbol, "TEN.MI"],
#     data_source="live",
#     signal_filter="all",
#     lookback_days=365,
#     lookback_bars=40,
# )
# result = graph.invoke(graph._initial_state)
# rich.print("[bold yellow]── Patterns agent ──[/bold yellow]")
# rich.print(result["final_output"])

# ---------------------------------------------------------------------------
# 4. Orchestrator — loads OHLCV once, runs indicators + patterns in parallel,
#    then synthesises all four assessments into a structured brief via gpt-4o
# ---------------------------------------------------------------------------
from techa.agents.orchestrator import create_orchestrator

graph  = create_orchestrator(symbol, data_source="live")
result = graph.invoke(graph._initial_state)
rich.print("[bold magenta]── Orchestrator ──[/bold magenta]")
rich.print(result["final_output"])
