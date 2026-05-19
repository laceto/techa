# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is your task?

**Adding or modifying indicators in `techa.indicators`** (ta-lib backed, raw OHLCV)
→ READ: `.claude/rules/coding-indicators.md`

**Adding or modifying candlestick pattern scanning or visualization in `techa.patterns`** (ta-lib backed, mplfinance)
→ READ: `.claude/rules/coding-patterns.md`

**Adding or modifying analytics in `techa.ma` or `techa.breakout`** (relative-price, custom Wilder)
→ READ: `.claude/rules/coding-ma-breakout.md`

**Writing tests**
→ READ: `.claude/rules/test-rules.md`

**Tests are failing — fixing them**
→ READ: `.claude/rules/test-failing-rules.md`
→ Overrides `test-rules.md`

**Adding or modifying agents in `techa.agents`** (LangGraph, OpenAI structured output)
— includes `techa.agents.insurance`, `techa.agents.indicators`, `techa.agents.patterns`,
  `techa.agents.ta`, `techa.agents.orchestrator`
→ READ: `.claude/rules/coding-agents.md`

**Adding or modifying insurance KPI analytics in `techa.insurance`**
(pure-Python; loss ratio, combined ratio, reserve adequacy, growth metrics)
→ READ: `.claude/rules/coding-agents.md` for agent wiring; no separate rule file — follow the
  same thin-orchestrator pattern as `techa.underwriting` and `techa.claims`.

**Adding or modifying medical underwriting analytics in `techa.underwriting`**
(pure-Python; BMI, BP, cholesterol, metabolic, lifestyle, conditions, family-risk loadings)
→ Follow the existing pattern in `techa/underwriting/`: `_adapter.py` validates input,
  domain modules compute per-group KPIs, `snapshot.py` orchestrates and caps total loading.

**Adding or modifying claims assessment analytics in `techa.claims`**
(pure-Python; timeline, severity, medical coherence, documentation, fraud indicators)
→ Follow the existing pattern in `techa/claims/`: `_adapter.py` validates and derives dates,
  domain modules compute per-group KPIs, `snapshot.py` orchestrates.

**Adding or modifying actuarial analytics in `techa.actuarial`**
(pure-Python; A/E monitoring, reinsurance pricing, in-force portfolio health)
→ Follow the existing pattern in `techa/actuarial/ae/`, `techa/actuarial/pricing/`,
  `techa/actuarial/inforce/`: each sub-package has `_adapter.py`, `analysis.py`, `snapshot.py`.

**Debugging NaN propagation, Wilder smoothing, or signal validation issues**
→ READ: `.claude/rules/debugging-rules.md`

---

Load your file now. Do not load the others.
