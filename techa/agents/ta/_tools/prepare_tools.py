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
    df_all = _read_parquet_dated(path, analysis_date)   # raises FileNotFoundError if missing

    # Filter to symbol; date resolution is scoped to this symbol's own range
    df = df_all[df_all["symbol"] == symbol].copy()

    if df.empty:
        raise ValueError(f"Symbol {symbol!r} not found in parquet.")

    dates = df["date"].dt.date   # already datetime from _read_parquet_dated
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
    relative: bool = False,
) -> tuple[str, pd.DataFrame]:
    """
    Download live OHLC for a single symbol, enrich with signals and stop-losses.

    Args:
        symbol:      Yahoo Finance ticker to analyse (e.g. "TCEHY").
        benchmark:   Benchmark ticker for calculate_relative_prices (default "FTSEMIB.MI").
        fx:          Optional FX ticker for currency conversion (e.g. "EURUSD=X").
                     Pass None when stock and benchmark share the same currency.
        config_path: Path to config.json for search-space parameters.
        relative:    If True, signals are computed on relative prices
                     (stock / benchmark). If False (default), absolute prices are
                     used. Matches the "relative: false" flags in config.json.

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

    # FX conversion: normalise stock into benchmark currency before computing relative prices
    if fx:
        df_fx = (
            handler.get_ohlc_data(fx)
            .reset_index()
            .rename(columns={"close": "fx_close"})[["date", "fx_close"]]
        )
        # get_ohlc_data returns a DatetimeIndex; reset so "date" becomes a merge key
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
    # stock_data first, benchmark second — matches pipeline.py and trend_scorer.py convention
    dfs = processor.calculate_relative_prices(df_stock, df_bm)
    print(dfs.tail())
    dfs, signal_columns = generate_signals(
        df=dfs,
        config_path=str(config_path),
        tt_search_space=tt_search_space,
        bo_search_space=bo_search_space,
        ma_search_space=ma_search_space,
        relative=relative,
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
