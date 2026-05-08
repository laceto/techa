"""
agents/patterns/_tools/ask_pattern_trader.py — Candlestick pattern scan AI assistant.

Usage (programmatic):
    from techa.agents.patterns._tools.ask_pattern_trader import ask_pattern_trader
    analysis = ask_pattern_trader(payload, tickers=["A2A.MI", "ENI.MI"])

What it does:
    Receives a JSON payload with the last-bar candlestick pattern hits from scan_last_bar
    and returns a structured multi-ticker analysis via OpenAI structured output.

Environment:
    OPENAI_API_KEY must be set.

Input payload shape (from prepare_node):
    {
      "tickers":       list[str],
      "scan_date":     "YYYY-MM-DD",
      "signal_filter": "all" | "bull" | "bear",
      "hits":          [{"ticker", "date", "display_name", "signal"}, ...],  # last-bar only
      "total_hits":    int,
      "recent_hits":   [{"ticker", "date", "display_name", "signal"}, ...],  # last lookback_bars bars
      "lookback_bars": int,
    }
    signal values: +100 = bullish, -100 = bearish.
    recent_hits overlaps with hits (last-bar patterns appear in both).
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Literal

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# Windows CLI only — Jupyter's OutStream has no reconfigure().
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a professional equity trader assistant. You receive a JSON payload with two
sections of candlestick pattern data across a set of tickers:

  hits         — patterns that fired on the most recent bar only (last-bar snapshot).
  recent_hits  — all patterns that fired in the last lookback_bars trading bars.
                 Overlaps with hits: today's patterns appear in both sections.
                 Use this to assess whether today's signal is isolated or part of
                 a developing sequence (clustering, recurrence, direction shift).

Signal convention (TA-Lib):
- signal = +100 → bullish pattern (potential reversal up or continuation of uptrend)
- signal = -100 → bearish pattern (potential reversal down or continuation of downtrend)

Pattern interpretation principles:
- A single candlestick pattern is a low-conviction signal on its own. Confluence of
  2+ same-direction patterns on the same ticker and date raises conviction to medium or high.
- Reversal patterns (e.g. Hammer, Engulfing, Morning Star, Evening Star, Doji) have
  higher significance than continuation patterns when they appear at a key structure.
- Doji patterns alone are ambiguous — they flag indecision, not direction.
- Multiple bearish patterns + 0 bullish patterns on a ticker = bearish bias, and vice versa.
- Mixed (bullish + bearish on same bar) = conflicting signals — flag as mixed, low conviction.
- Recent activity trend:
    increasing  — more pattern hits in the second half of the lookback window than the first.
                  Signals are building — the setup is developing conviction.
    decreasing  — fewer hits in the second half. Pattern activity is fading.
    stable      — roughly equal distribution. Background noise or a steady theme.

Output format:
- description: 3-5 sentence overall narrative. Which tickers fired today, net
  bullish/bearish balance, standout confluences, and whether recent history
  supports or contradicts today's signal. No filler.
- total_hits, bullish_count, bearish_count: from the input payload (last-bar counts).
- recent_activity: one entry per ticker that appears in recent_hits. For each ticker
  summarise the pattern activity over the lookback window: total hits, bullish/bearish
  split, distinct pattern names, and whether activity is increasing/stable/decreasing.
  commentary: one sentence — is the recent pattern flow building conviction or fading?
- ticker_summaries: one entry per ticker with a last-bar hit. Use recent_activity to
  add context: note if today's signal is the first in a while (isolated) or the latest
  in a sequence (reinforcing). net_bias and conviction from last-bar confluence logic.
  verdict: one actionable sentence — pattern names, direction, recency context, and
  what to watch.
- top_actionable: tickers with medium or high conviction on the last bar AND supporting
  recent activity (not purely isolated single-bar signals).
- watchlist: single-bar hits or mixed signals, or tickers with declining recent activity.
- summary: single sentence. Net market message — overall bias and most notable pattern
  development across the scan.

Be concise. No padding. Use exact pattern names and signal directions from the data.\
"""


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class RecentPatternActivity(BaseModel):
    ticker:            str                                                   = Field(description="Ticker symbol.")
    total_recent_hits: int                                                   = Field(description="Total pattern hits in the last lookback_bars bars.")
    bullish_recent:    int                                                   = Field(description="Bullish (+100) hits in the recent window.")
    bearish_recent:    int                                                   = Field(description="Bearish (-100) hits in the recent window.")
    pattern_names:     list[str]                                             = Field(description="Distinct pattern display names that fired in the recent window.")
    activity_trend:    Literal["increasing", "stable", "decreasing"]        = Field(
        description=(
            "increasing: more hits in the second half of the window than the first half. "
            "decreasing: fewer hits in the second half. stable: roughly equal."
        )
    )
    commentary:        str                                                   = Field(
        description="One sentence: is pattern activity building conviction, fading, or holding steady over the recent window?"
    )


class TickerPatternSummary(BaseModel):
    ticker:           str                                              = Field(description="Ticker symbol.")
    date:             str                                              = Field(description="ISO date of the last bar that fired.")
    bullish_patterns: list[str]                                        = Field(description="Display names of bullish (+100) patterns that fired.")
    bearish_patterns: list[str]                                        = Field(description="Display names of bearish (-100) patterns that fired.")
    net_bias:         Literal["bullish", "bearish", "mixed", "none"]   = Field(
        description=(
            "bullish: only bullish patterns. bearish: only bearish patterns. "
            "mixed: both directions. none: no patterns (should not appear)."
        )
    )
    conviction:       Literal["high", "medium", "low"]                 = Field(
        description=(
            "high: 3+ same-direction patterns. medium: 2 same-direction patterns. "
            "low: 1 pattern or mixed signals."
        )
    )
    verdict:          str                                              = Field(
        description="One actionable sentence for a professional trader. Pattern names, direction, and what to watch."
    )


class PatternScanAnalysis(BaseModel):
    description:      str                       = Field(
        description=(
            "3-5 sentence narrative summary of the full scan. Which tickers fired, "
            "net bullish/bearish balance, notable confluences. Exact pattern names. No filler."
        )
    )
    total_hits:       int                          = Field(description="Total pattern hits on the last bar across all tickers.")
    bullish_count:    int                          = Field(description="Number of bullish (+100) last-bar hits.")
    bearish_count:    int                          = Field(description="Number of bearish (-100) last-bar hits.")
    recent_activity:  list[RecentPatternActivity]  = Field(
        description=(
            "One entry per ticker that appears in recent_hits. "
            "Empty list if no patterns fired in the recent window for any ticker."
        )
    )
    ticker_summaries: list[TickerPatternSummary]   = Field(description="One entry per ticker that had at least one last-bar hit.")
    top_actionable:   list[str]                    = Field(
        description="Tickers with medium or high conviction (2+ same-direction patterns). Prioritised for action."
    )
    watchlist:        list[str]                  = Field(
        description="Tickers with low conviction (single hit or mixed). Worth monitoring."
    )
    summary:          str                        = Field(
        description="One sentence. Net market message from this scan — overall bias and most notable finding."
    )


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

MODEL = "gpt-4.1-nano"


def ask_pattern_trader(
    payload: dict,
    tickers: list[str],
    question: str | None = None,
) -> PatternScanAnalysis:
    """
    Send the scan_last_bar payload to an OpenAI model and return a structured analysis.

    Uses OpenAI structured output (beta.chat.completions.parse) to guarantee
    the response conforms to PatternScanAnalysis schema — no post-processing needed.

    Args:
        payload:  Dict from prepare_node — the scan_last_bar results payload.
        tickers:  List of ticker symbols that were scanned (for context in the message).
        question: Optional follow-up question. Defaults to None.

    Returns:
        PatternScanAnalysis Pydantic model parsed directly from the model response.
    """
    client = openai.OpenAI()

    user_content = (
        f"Tickers scanned: {', '.join(tickers)}\n\n"
        f"Scan payload:\n{json.dumps(payload, indent=2)}"
    )
    if question:
        user_content += f"\n\nQuestion: {question}"

    log.info(
        "Sending pattern scan payload to %s — %d hits across %d tickers",
        MODEL,
        payload.get("total_hits", 0),
        len(tickers),
    )

    response = client.beta.chat.completions.parse(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        response_format=PatternScanAnalysis,
    )

    return response.choices[0].message.parsed
