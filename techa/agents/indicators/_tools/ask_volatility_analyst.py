"""
agents/indicators/_tools/ask_volatility_analyst.py — Volatility & volume flow AI analyst.

Analyses ATR, Bollinger Bands, historical volatility, and volume flow indicators
from the build_snapshot output. Returns a structured VolatilityAnalysis.
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
You are a professional equity analyst specialising in volatility and volume flow analysis.
You receive a JSON snapshot of a single ticker's last trading bar and return a
structured volatility and volume assessment.

Relevant snapshot fields:
- atr:           Average True Range (14-bar). Absolute price units; use for stop sizing.
- atr_pct:       NATR — normalised ATR = ATR / close * 100. Comparable across tickers.
                 < 1.5% = low volatility. 1.5–3% = normal. > 3% = high volatility.
- bb_upper/mid/lower: Bollinger Bands (20-bar SMA ± 2 std dev).
- bb_width:      (upper - lower) / mid. Low = tight bands = potential energy building.
                 High = wide bands = volatility expansion in progress.
- bb_pct_b:      Position within bands: 0 = at lower band, 1 = at upper band, 0.5 = midline.
                 > 1 = above upper band (overbought extension).
                 < 0 = below lower band (oversold extension).
- hist_vol_20d:  Annualised historical volatility over last 20 bars (%).
                 Compare to atr_pct to assess whether current ATR is consistent with
                 recent volatility. Large divergence may signal a recent spike or collapse.
- volume:        Last-bar absolute volume.
- obv:           On-Balance Volume (cumulative). Rising OBV = buying pressure.
                 Falling OBV = selling pressure. Only meaningful as a trend, not a level.
- ad:            Chaikin Accumulation/Distribution line. Measures money flow.
                 Divergence from price = potential reversal signal.
- adosc:         Chaikin A/D Oscillator (EMA3 - EMA10 of AD). Crosses zero signal
                 shifts in accumulation vs. distribution. Positive = accumulation.

Interpretation guidelines:
- Volatility regime: atr_pct < 1.5 = low; 1.5–3 = normal; > 3 = high.
- BB position:
    bb_pct_b > 0.8 = upper band (near overbought).
    bb_pct_b < 0.2 = lower band (near oversold).
    bb_pct_b 0.4–0.6 = middle (trend continuation zone when trending, mean reversion when ranging).
- BB squeeze: bb_width is contextual but a narrow width (< 0.05) signals potential breakout.
- Volume flow: adosc > 0 = accumulation; < 0 = distribution.
  Trend + adosc > 0 = buying conviction. Trend + adosc < 0 = potential distribution / churn.

Output:
- volatility_regime: "low" (atr_pct < 1.5), "normal" (1.5–3), "high" (> 3).
- bb_position: "above_upper" (bb_pct_b > 1), "upper_half" (0.5–1), "middle" (0.4–0.6),
  "lower_half" (0–0.5), "below_lower" (< 0).
- bb_squeeze: True when bb_width < 0.06 (tight bands, potential energy).
- volume_flow: "accumulation" (adosc > 0), "distribution" (adosc < 0), "neutral" (near 0).
- conviction: "high" = clear regime + directional flow; "medium" = one ambiguous signal;
  "low" = conflicting signals or missing data.\
"""


class VolatilityAnalysis(BaseModel):
    description:       str = Field(
        description=(
            "2-3 sentence summary of the volatility and volume flow picture. "
            "Cover: current ATR regime, Bollinger Band position, whether bands are squeezing, "
            "and volume flow direction. Use exact values from the snapshot."
        )
    )
    volatility_regime: Literal["high", "normal", "low"] = Field(
        description="low: atr_pct < 1.5. normal: 1.5–3. high: > 3."
    )
    atr_pct:           float = Field(description="NATR value from snapshot (normalised ATR %).")
    hist_vol_20d:      float = Field(description="Annualised 20-bar historical volatility (%) from snapshot.")
    bb_position:       Literal["above_upper", "upper_half", "middle", "lower_half", "below_lower"] = Field(
        description="Position within Bollinger Bands based on bb_pct_b."
    )
    bb_pct_b:          float = Field(description="Bollinger %B from snapshot (0 = lower band, 1 = upper band).")
    bb_width:          float = Field(description="Bollinger Band width from snapshot.")
    bb_squeeze:        bool  = Field(description="True when bb_width < 0.06 — tight bands signal potential breakout.")
    volume_flow:       Literal["accumulation", "distribution", "neutral"] = Field(
        description="Based on adosc: > 0 = accumulation, < 0 = distribution, near 0 = neutral."
    )
    conviction:        Literal["high", "medium", "low"] = Field(
        description="high: clear volatility regime + directional volume flow. medium: one ambiguous. low: conflicting."
    )
    verdict:           str   = Field(
        description="One actionable sentence. Include atr_pct, bb_pct_b, and volume flow direction."
    )


def ask_volatility_analyst(payload: dict, ticker: str, question: str | None = None) -> VolatilityAnalysis:
    """
    Send the indicator snapshot to the model for volatility and volume flow analysis.

    Args:
        payload: Dict from prepare_node — contains "symbol", "date", "snapshot".
        ticker:  Ticker symbol string.
        question: Optional follow-up question.

    Returns:
        VolatilityAnalysis Pydantic model.
    """
    snapshot = payload.get("snapshot", payload)
    vol_keys = [
        "atr", "atr_pct",
        "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pct_b",
        "hist_vol_20d",
        "volume", "obv", "ad", "adosc",
    ]
    vol_snap = {k: snapshot[k] for k in vol_keys if k in snapshot}

    user_content = (
        f"Ticker: {ticker}  Date: {payload.get('date', 'unknown')}\n\n"
        f"Volatility & volume snapshot:\n{json.dumps(vol_snap, indent=2)}"
    )
    if question:
        user_content += f"\n\nQuestion: {question}"

    log.info("Sending volatility snapshot for %s to %s", ticker, MODEL)

    return invoke_structured(
        VolatilityAnalysis,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=1024,
    )
