"""
agents/indicators/_tools/ask_trend_analyst.py — MA trend AI analyst.

Analyses the SMA/EMA alignment, slope quality, and golden cross from the
build_snapshot output. Returns a structured TrendAnalysis.
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
You are a professional equity analyst specialising in moving-average trend analysis.
You receive a JSON snapshot of a single ticker's last trading bar and return a
structured trend assessment.

Relevant snapshot fields:
- price:             Last close price (absolute or relative depending on data mode).
- sma20/50/200:      Simple moving averages. Price above all three = strong uptrend.
- ema20/50:          Exponential moving averages. EMA reacts faster than SMA.
- dist_sma20_pct:    % distance of price from SMA20. Negative = price below SMA20.
- dist_sma50_pct:    % distance of price from SMA50.
- dist_sma200_pct:   % distance of price from SMA200.
- slope_sma20:       OLS slope of SMA20 over the last 10 bars, in %/bar.
                     Positive = SMA20 is rising; negative = falling.
- slope_sma20_r2:    R² of the slope fit. < 0.3 = noisy (slope unreliable);
                     0.3–0.7 = moderate; ≥ 0.7 = strong (slope is directional).
- golden_cross:      True when SMA50 > SMA200 (long-term uptrend confirmation).
                     False = death cross (long-term downtrend).

Interpretation guidelines:
- SMA alignment (bullish): price > SMA20 > SMA50 > SMA200 — full stack bullish.
- SMA alignment (bearish): price < SMA20 < SMA50 < SMA200 — full stack bearish.
- slope_sma20 near 0 + slope_sma20_r2 < 0.3 = sideways; no trend conviction.
- Golden cross + dist_sma200_pct > 0 + rising slope = high-conviction uptrend.
- dist_sma20_pct extremes (> +5% or < -5%) may indicate short-term extension.

Output:
- sma_alignment: "bullish" if price above SMA50 and golden_cross=True;
                 "bearish" if price below SMA50 and golden_cross=False;
                 "mixed" otherwise.
- slope_direction: "up" if slope_sma20 > 0.05, "down" if < -0.05, else "flat".
- slope_quality: "strong" (r2 >= 0.7), "moderate" (0.3–0.7), "weak" (< 0.3).
- conviction: "high" = aligned SMA stack + strong directional slope;
              "medium" = partial alignment or moderate slope;
              "low" = mixed or sideways.
- verdict: one actionable sentence for a professional trader.\
"""


class TrendAnalysis(BaseModel):
    description:     str = Field(
        description=(
            "2-3 sentence summary of the MA trend picture. "
            "Cover: SMA stack alignment, slope direction and quality, golden/death cross status. "
            "Use exact values from the snapshot."
        )
    )
    sma_alignment:   Literal["bullish", "bearish", "mixed"] = Field(
        description="bullish: price above SMA50 and golden_cross=True. bearish: price below SMA50 and golden_cross=False. mixed: neither."
    )
    slope_direction: Literal["up", "flat", "down"] = Field(
        description="up: slope_sma20 > 0.05. down: slope_sma20 < -0.05. flat: otherwise."
    )
    slope_quality:   Literal["strong", "moderate", "weak"] = Field(
        description="strong: r2 >= 0.7. moderate: 0.3–0.7. weak: < 0.3."
    )
    golden_cross:    bool  = Field(description="True when SMA50 > SMA200.")
    dist_sma20_pct:  float = Field(description="% distance of price from SMA20 (from snapshot).")
    dist_sma50_pct:  float = Field(description="% distance of price from SMA50 (from snapshot).")
    conviction:      Literal["high", "medium", "low"] = Field(
        description="high: full SMA alignment + strong slope. medium: partial. low: mixed or sideways."
    )
    verdict:         str   = Field(
        description="One actionable sentence for a professional trader. Include slope value and R²."
    )


def ask_trend_analyst(payload: dict, ticker: str, question: str | None = None) -> TrendAnalysis:
    """
    Send the indicator snapshot to the model for trend analysis.

    Args:
        payload: Dict from prepare_node — contains "symbol", "date", "snapshot".
        ticker:  Ticker symbol string (for context in the user message).
        question: Optional follow-up question.

    Returns:
        TrendAnalysis Pydantic model.
    """
    snapshot = payload.get("snapshot", payload)
    trend_keys = [
        "price", "sma20", "sma50", "sma200", "ema20", "ema50",
        "dist_sma20_pct", "dist_sma50_pct", "dist_sma200_pct",
        "slope_sma20", "slope_sma20_r2", "golden_cross",
    ]
    trend_snap = {k: snapshot[k] for k in trend_keys if k in snapshot}

    user_content = (
        f"Ticker: {ticker}  Date: {payload.get('date', 'unknown')}\n\n"
        f"Trend snapshot:\n{json.dumps(trend_snap, indent=2)}"
    )
    if question:
        user_content += f"\n\nQuestion: {question}"

    log.info("Sending trend snapshot for %s to %s", ticker, MODEL)

    return invoke_structured(
        TrendAnalysis,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=1024,
    )
