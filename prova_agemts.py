import rich

# from techa.agents.ta import create_manager
# data_source = "live"

# symbol = 'TEN.MI'
# benchmark = 'FTSEMIB.MI'
# date = '2026-05-08'
# fx = None

# graph = create_manager(
#     symbol=symbol,
#     analysis_date=date,
#     data_source=data_source,
#     benchmark=benchmark,
#     fx=fx,
# )

# result = graph.invoke({
#     "symbol":        symbol,
#     "analysis_date": date,
#     "data_source":   data_source,
#     "benchmark":     benchmark,
#     "fx":            fx,
# })

# rich.print(result)


from techa.agents.patterns import create_pattern_agent
graph = create_pattern_agent(
      ['TEN.MI'],
      analysis_date=None,
      data_source="live",   # default parquet, same as create_manager
      benchmark="FTSEMIB.MI",
      fx=None,
      signal_filter="all",
      lookback_days=365,
      lookback_bars=40,
      checkpointer=None,
  )
result = graph.invoke(graph._initial_state)
rich.print(result["final_output"])
# rich.print(result)