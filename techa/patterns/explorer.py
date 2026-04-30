"""
techa/patterns/explorer.py — Pattern visualization.

Public API
----------
plot_pattern(ohlcv, pattern, *, ...) -> None
    Overview price chart + up to 5 candlestick zoom windows for one pattern.

explore_patterns(ohlcv, patterns=None, signal_filter="all", *, ...) -> None
    Calls plot_pattern for every pattern that fired at least once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplfinance as mpf
import pandas as pd
from matplotlib.lines import Line2D

from techa.patterns._registry import PATTERN_DISPLAY, PATTERNS
from techa.patterns.scanner import scan_patterns

__all__ = ["plot_pattern", "explore_patterns"]

_MC = mpf.make_marketcolors(up="g", down="r", inherit=True)
_STYLE = mpf.make_mpf_style(marketcolors=_MC)


def _normalize(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    df.columns = df.columns.str.lower()
    return df


def _zoom_subset(df: pd.DataFrame, date, window_bars: int) -> tuple[pd.DataFrame, int]:
    """Return (subset, pattern_bar_position_in_subset)."""
    try:
        idx = df.index.get_loc(date)
    except KeyError:
        return df.iloc[0:0], 0
    start = max(0, idx - window_bars)
    end = min(len(df), idx + window_bars + 1)
    return df.iloc[start:end], idx - start


def _set_xticks(ax: plt.Axes, dates, pat_pos: int) -> None:
    """Show only first, pattern, and last date to avoid label clutter."""
    n = len(dates)
    if n == 0:
        return
    positions = sorted({0, pat_pos, n - 1})
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [dates[p].strftime("%b %d") for p in positions],
        fontsize=7,
        rotation=30,
        ha="right",
    )


def _vol_fmt(v: float, _) -> str:
    if v >= 1e9:
        return f"{v/1e9:.1f}B"
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    if v >= 1e3:
        return f"{v/1e3:.0f}K"
    return f"{v:.0f}"


def _draw_zoom(
    fig: plt.Figure,
    gs_spec,
    gs_vol_spec,
    subset: pd.DataFrame,
    pat_pos: int,
    label: str,
    color: str,
    has_volume: bool,
) -> None:
    ax_p = fig.add_subplot(gs_spec)

    subset_mpf = subset.rename(
        columns={
            c: c.title()
            for c in ["open", "high", "low", "close", "volume"]
            if c in subset.columns
        }
    )
    mpf.plot(
        subset_mpf,
        type="candle",
        ax=ax_p,
        volume=False,
        show_nontrading=False,
        style=_STYLE,
    )

    # Shade the pattern bar
    ax_p.axvspan(pat_pos - 0.45, pat_pos + 0.45, alpha=0.18, color=color, zorder=0)
    _set_xticks(ax_p, subset.index, pat_pos)
    ax_p.set_title(label, fontsize=7, color=color, pad=3)
    ax_p.tick_params(axis="y", labelsize=6)
    ax_p.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="both"))

    if has_volume and "volume" in subset.columns and gs_vol_spec is not None:
        ax_v = fig.add_subplot(gs_vol_spec)
        n = len(subset)
        bar_colors = [
            "g" if c >= o else "r"
            for o, c in zip(subset["open"].values, subset["close"].values)
        ]
        ax_v.bar(range(n), subset["volume"].values, color=bar_colors, alpha=0.55, width=0.65)
        ax_v.axvspan(pat_pos - 0.45, pat_pos + 0.45, alpha=0.18, color=color, zorder=0)
        ax_v.set_xlim(-0.5, n - 0.5)
        ax_v.tick_params(axis="x", labelbottom=False)
        ax_v.tick_params(axis="y", labelsize=6)
        ax_v.yaxis.set_major_formatter(mticker.FuncFormatter(_vol_fmt))
        ax_v.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))


def plot_pattern(
    ohlcv: pd.DataFrame,
    pattern: str,
    *,
    signal_filter: Literal["all", "bull", "bear"] = "all",
    window_bars: int = 5,
    max_occurrences: int = 5,
    show_volume: bool = True,
    output: Literal["show", "save"] = "show",
    output_dir: str | Path = ".",
    figsize: tuple[int, int] = (20, 10),
    symbol: str = "",
) -> None:
    """
    Plot a price overview with detected pattern signals and candlestick zoom windows.

    Layout
    ------
    Row 0 : Full-width price line. Thin dashed lines for every signal; thicker
            numbered markers for the n zoom occurrences shown below.
    Row 1 : Full-width volume (optional).
    Row 2 : Up to 5 candlestick zoom windows, one column each.
    Row 3 : Per-zoom volume bars (optional).

    Args:
        ohlcv:           DataFrame with open/high/low/close[/volume] (case-insensitive).
                         Index must be datetime, sorted ascending.
        pattern:         TA-Lib function name, e.g. "CDLENGULFING".
        signal_filter:   "all", "bull" (+100 only), or "bear" (−100 only).
        window_bars:     Trading bars shown on each side of each pattern occurrence.
        max_occurrences: Maximum zoom windows. Capped at 5.
        show_volume:     Add volume bars to overview and zoom windows.
                         Silently skipped when ohlcv has no volume column.
        output:          "show" opens an interactive window; "save" writes a PNG.
        output_dir:      Destination for saved PNGs (created if absent).
        figsize:         Figure size in inches (width, height).
        symbol:          Ticker label for chart titles (cosmetic only).
    """
    display_name = PATTERN_DISPLAY.get(pattern, pattern)
    scan_df = scan_patterns(ohlcv, patterns=[pattern], signal_filter=signal_filter)
    if scan_df.empty:
        return

    df = _normalize(ohlcv)
    has_volume = show_volume and "volume" in df.columns

    n_zooms = min(max_occurrences, 5, len(scan_df))
    zoom_rows = scan_df.tail(n_zooms).iloc[::-1].reset_index(drop=True)
    ncols = n_zooms

    # Layout: [main_price, main_vol?, zoom_price, zoom_vol?]
    if has_volume:
        height_ratios = [4, 1, 3, 1]
        nrows = 4
        zoom_price_row, zoom_vol_row = 2, 3
        main_vol_row = 1
    else:
        height_ratios = [3, 2]
        nrows = 2
        zoom_price_row, zoom_vol_row = 1, None
        main_vol_row = None

    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(nrows, ncols, height_ratios=height_ratios)

    # --- Main overview ---
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(df.index, df["close"], color="steelblue", lw=1, zorder=2)

    # All signals: thin background lines
    for _, row in scan_df.iterrows():
        c = "green" if row["signal"] > 0 else "red"
        ax_main.axvline(row["date"], color=c, ls="--", alpha=0.2, lw=0.7, zorder=1)

    # Zoomed signals: thick lines + numbered callouts
    for i, (_, row) in enumerate(zoom_rows.iterrows()):
        c = "green" if row["signal"] > 0 else "red"
        ax_main.axvline(row["date"], color=c, ls="--", alpha=0.85, lw=1.4, zorder=3)
        try:
            y = df["close"].loc[row["date"]]
        except KeyError:
            y = df["close"].iloc[-1]
        ax_main.annotate(
            f"#{i + 1}",
            xy=(row["date"], y),
            xytext=(4, 4),
            textcoords="offset points",
            color=c,
            fontsize=8,
            fontweight="bold",
            zorder=4,
        )

    n_bull = int((scan_df["signal"] > 0).sum())
    n_bear = int((scan_df["signal"] < 0).sum())
    prefix = f"{symbol} — " if symbol else ""
    ax_main.set_title(
        f"{prefix}{display_name}  ·  {n_bull} bullish  {n_bear} bearish  ({n_bull + n_bear} total)",
        fontsize=10,
    )
    ax_main.set_ylabel("Price")
    ax_main.grid(True, alpha=0.25)
    ax_main.legend(
        handles=[
            Line2D([0], [0], color="green", ls="--", lw=1.2, label="Bullish (+100)"),
            Line2D([0], [0], color="red",   ls="--", lw=1.2, label="Bearish (−100)"),
        ],
        loc="upper left",
        fontsize=8,
    )

    # --- Main volume ---
    if has_volume and main_vol_row is not None:
        ax_mv = fig.add_subplot(gs[main_vol_row, :], sharex=ax_main)
        ax_mv.bar(df.index, df["volume"], color="steelblue", alpha=0.3, width=1)
        ax_mv.yaxis.set_major_formatter(mticker.FuncFormatter(_vol_fmt))
        ax_mv.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
        ax_mv.tick_params(axis="y", labelsize=7)
        ax_mv.grid(True, alpha=0.2)
        ax_main.tick_params(axis="x", labelbottom=False)

    # --- Zoom windows ---
    for i, (_, row) in enumerate(zoom_rows.iterrows()):
        subset, pat_pos = _zoom_subset(df, row["date"], window_bars)
        if subset.empty:
            continue
        color = "green" if row["signal"] > 0 else "red"
        direction = "▲" if row["signal"] > 0 else "▼"
        label = f"{direction} #{i + 1}  {row['date'].strftime('%Y-%m-%d')}"

        vol_spec = gs[zoom_vol_row, i] if (has_volume and zoom_vol_row is not None) else None

        _draw_zoom(
            fig,
            gs[zoom_price_row, i],
            vol_spec,
            subset,
            pat_pos,
            label,
            color,
            has_volume,
        )

    if output == "save":
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{pattern}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def explore_patterns(
    ohlcv: pd.DataFrame,
    patterns: list[str] | None = None,
    signal_filter: Literal["all", "bull", "bear"] = "all",
    *,
    window_bars: int = 5,
    max_occurrences: int = 5,
    show_volume: bool = True,
    output: Literal["show", "save"] = "show",
    output_dir: str | Path = ".",
    figsize: tuple[int, int] = (20, 10),
    symbol: str = "",
) -> None:
    """
    Plot all patterns that fired at least once in the given OHLCV data.

    Calls plot_pattern for each pattern with at least one hit matching signal_filter.
    Patterns with zero matching signals are silently skipped.

    Args:
        ohlcv:           DataFrame with open/high/low/close[/volume] (case-insensitive).
        patterns:        TA-Lib function names to check. None checks all 61 patterns.
        signal_filter:   "all", "bull" (+100 only), or "bear" (−100 only).
        window_bars:     Trading bars shown on each side of each occurrence.
        max_occurrences: Maximum zoom windows per pattern. Capped at 5.
        show_volume:     Show volume bars. Silently skipped if ohlcv has no volume column.
        output:          "show" opens one interactive window per pattern that fired.
                         "save" writes one PNG per pattern to output_dir.
        output_dir:      Destination for saved PNGs.
        figsize:         Figure size in inches (width, height).
        symbol:          Ticker label for chart titles (cosmetic only).
    """
    names = [tn for _, tn in PATTERNS] if patterns is None else patterns
    for talib_name in names:
        plot_pattern(
            ohlcv,
            talib_name,
            signal_filter=signal_filter,
            window_bars=window_bars,
            max_occurrences=max_occurrences,
            show_volume=show_volume,
            output=output,
            output_dir=output_dir,
            figsize=figsize,
            symbol=symbol,
        )
