"""
agents/indicators/_tools/ask_momentum_analyst.py — Momentum AI analyst.

Analyses MACD, Stochastic, and ROC from the build_snapshot output.
Returns a structured MomentumAnalysis.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Literal

from pydantic import BaseModel, Field

from techa.agents._llm import invoke_structured, MODEL

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a professional equity analyst specialising in momentum and oscillator analysis.
You receive a JSON snapshot of a single ticker's last trading bar and return a
structured momentum assessment.

Relevant snapshot fields:
- macd:           MACD line (EMA12 - EMA26). Positive = bullish; negative = bearish.
- macd_signal:    9-bar EMA of the MACD line (signal line).
- macd_hist:      MACD histogram = macd - macd_signal.
                  Positive and rising = accelerating upward momentum.
                  Negative and falling = accelerating downward momentum.
                  Histogram crossing zero = momentum reversal signal.
- stoch_k:        Slow stochastic %K (14,3,3). Range 0–100.
                  > 80 = overbought territory. < 20 = oversold territory.
- stoch_d:        Slow stochastic %D (3-bar SMA of %K). Signal line for stoch_k.
- stoch_fast_k:   Fast stochastic %K (5,3). Reacts faster, more noise.
- stoch_fast_d:   Fast stochastic %D.
- roc_10d:        Rate of change over 10 bars (%). Positive = price rising.
- roc_20d:        Rate of change over 20 bars (%). Medium-term momentum.
- chg_1d:         1-day price change (%).
- chg_5d:         5-day price change (%).

Interpretation guidelines:
- MACD bias: macd_hist > 0 and rising = bullish. macd_hist < 0 and falling = bearish.
  macd crossing above signal = fresh bullish signal. Below = bearish.
- Stochastic zone:
    stoch_k > 80 = overbought (caution for longs, opportunity for shorts).
    stoch_k < 20 = oversold (caution for shorts, opportunity for longs).
    20–80 = neutral.
- Momentum direction:
    roc_20d > 0 + macd_hist > 0 + chg_5d > 0 = accelerating upward.
    All negative = accelerating downward.
    Mixed signs = decelerating or rotating.
- Stochastic cross: stoch_k crossing stoch_d in overbought/oversold zones
  is a higher-conviction signal than a mid-range cross.

Output:
- macd_bias: "bullish" (macd_hist > 0), "bearish" (< 0), "neutral" (near 0).
- stoch_condition: "overbought" (stoch_k > 75), "oversold" (< 25), "neutral".
- momentum_direction: "accelerating_up", "accelerating_down", "decelerating", "neutral".
- conviction: "high" = aligned MACD + stoch + ROC; "medium" = partial; "low" = mixed.\
"""


class MomentumAnalysis(BaseModel):
    description:          str = Field(
        description=(
            "2-3 sentence summary of the momentum picture. "
            "Cover: MACD position and histogram direction, stochastic zone, "
            "ROC trend, and short-term change. Use exact values."
        )
    )
    macd_bias:            Literal["bullish", "bearish", "neutral"] = Field(
        description="bullish: macd_hist > 0. bearish: macd_hist < 0. neutral: near zero or diverging."
    )
    macd_hist:            float = Field(description="MACD histogram value from snapshot.")
    stoch_condition:      Literal["overbought", "neutral", "oversold"] = Field(
        description="overbought: stoch_k > 75. oversold: stoch_k < 25. neutral: otherwise."
    )
    stoch_k:              float = Field(description="Slow stochastic %K value from snapshot.")
    momentum_direction:   Literal["accelerating_up", "accelerating_down", "decelerating", "neutral"] = Field(
        description=(
            "accelerating_up: roc_20d > 0 and macd_hist > 0 and chg_5d > 0. "
            "accelerating_down: all negative. decelerating: mixed signs. neutral: near zero."
        )
    )
    roc_20d:              float = Field(description="20-bar rate of change (%) from snapshot.")
    chg_5d:               float = Field(description="5-day price change (%) from snapshot.")
    conviction:           Literal["high", "medium", "low"] = Field(
        description="high: MACD + stoch + ROC all aligned. medium: 2 of 3. low: mixed or conflicting."
    )
    verdict:              str   = Field(
        description="One actionable sentence. Include macd_hist, stoch_k, and roc_20d values."
    )


def ask_momentum_analyst(payload: dict, ticker: str, question: str | None = None) -> MomentumAnalysis:
    """
    Send the indicator snapshot to the model for momentum analysis.

    Args:
        payload: Dict from prepare_node — contains "symbol", "date", "snapshot".
        ticker:  Ticker symbol string.
        question: Optional follow-up question.

    Returns:
        MomentumAnalysis Pydantic model.
    """
    snapshot = payload.get("snapshot", payload)
    momentum_keys = [
        "macd", "macd_signal", "macd_hist",
        "stoch_k", "stoch_d", "stoch_fast_k", "stoch_fast_d",
        "roc_10d", "roc_20d", "chg_1d", "chg_5d",
    ]
    momentum_snap = {k: snapshot[k] for k in momentum_keys if k in snapshot}

    user_content = (
        f"Ticker: {ticker}  Date: {payload.get('date', 'unknown')}\n\n"
        f"Momentum snapshot:\n{json.dumps(momentum_snap, indent=2)}"
    )
    if question:
        user_content += f"\n\nQuestion: {question}"

    log.info("Sending momentum snapshot for %s to %s", ticker, MODEL)

    return invoke_structured(
        MomentumAnalysis,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=1024,
    )
