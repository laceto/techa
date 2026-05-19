# techa

Technical analysis and insurance risk primitives for trader and actuary assistants.

## Subpackages

| Package | Backed by | Input | Purpose |
|---|---|---|---|
| `techa.indicators` | TA-Lib | Raw OHLCV DataFrame | Last-bar indicator snapshot (trend, momentum, volatility, volume) |
| `techa.patterns` | TA-Lib + mplfinance | Raw OHLCV DataFrame | Candlestick pattern scanner and visualizer (61 patterns) |
| `techa.ma` | Manual Wilder | Relative-price parquet | Moving-average crossover analytics |
| `techa.breakout` | Manual | Relative-price parquet | Range breakout analytics |
| `techa.insurance` | Pure Python | Periodic financial DataFrame | Insurance financial KPI snapshot (loss/expense/combined ratios, reserves, growth) |
| `techa.underwriting` | Pure Python | Applicant questionnaire dict | Medical underwriting KPI snapshot (biometrics, CV, metabolic, lifestyle, conditions) |
| `techa.claims` | Pure Python | Claim form dict | Claims assessment KPI snapshot (timeline, severity, medical, documentation, fraud) |
| `techa.actuarial` | Pure Python | Experience / pricing / in-force dicts | A/E monitoring, reinsurance deal pricing, and in-force portfolio health snapshots |
| `techa.agents` | LangGraph + OpenAI | OHLCV / parquet | AI-powered multi-agent technical analysis and pattern reports |
| `techa.agents.orchestrator` | LangGraph + OpenAI + LangChain | OHLCV / parquet | Single-ticker orchestrator: loads OHLCV once and fans out to `indicators`, `patterns`, and `ta` agents |
| `techa.agents.insurance` | LangGraph + OpenAI | Risk profile dict | Insurance risk assessment: actuarial, financial, medical underwriting, and claims workers |

---

## Trading Domain

### `techa.indicators`

Last-bar technical indicator snapshot from raw OHLCV history.

```python
from techa.indicators import build_snapshot, build_snapshot_from_parquet

snap = build_snapshot(ohlcv_df)                    # dict of floats
snap = build_snapshot(ohlcv_df, nan_to_none=True)  # JSON-safe (NaN → None)

snap = build_snapshot_from_parquet(
    ticker="SIE.DE",
    data_path="data/ohlcv.parquet",
    ticker_col="symbol",
    date_col="date",
)
```

**Input:** DataFrame with columns `open/high/low/close/volume` (case-insensitive), sorted ascending, minimum 30 bars.

**Selected output keys:**

| Key | Notes |
|---|---|
| `price` | Last close |
| `golden_cross` | SMA50 > SMA200 |
| `dist_sma20/50/200_pct` | % distance from each MA |
| `slope_sma20`, `slope_sma20_r2` | OLS slope (%/bar) and R²; R² < 0.3 = noise |
| `macd_hist` | Positive = bullish |
| `stoch_k/d` | Slow stochastic |
| `roc_10d/20d` | Rate of change (%) |
| `atr_pct` | Normalised ATR (%) |
| `bb_upper/mid/lower`, `bb_pct_b` | Bollinger bands; position within bands [0, 1] |
| `hist_vol_20d` | Annualised historical volatility (%) |
| `obv`, `ad`, `adosc` | On-balance volume, Chaikin A/D line, Chaikin A/D oscillator |

Group snapshots across many tickers:

```python
from techa.indicators import build_group_snapshot, build_ticker_snapshot

gs = build_group_snapshot(ohlcv_dict)   # GroupSnapshot(tickers=df, groups=df)
df = build_ticker_snapshot(ohlcv_dict)  # per-ticker last-bar table
```

---

### `techa.patterns`

Detect and visualise all 61 TA-Lib candlestick patterns.

```python
from techa.patterns import scan_patterns, scan_last_bar, plot_pattern, explore_patterns

hits = scan_patterns(ohlcv_df)
hits = scan_patterns(ohlcv_df, patterns=["CDLENGULFING", "CDLDOJI"])
hits = scan_patterns(ohlcv_df, signal_filter="bull")  # +100 only
# columns: date | talib_name | display_name | signal (+100 or -100)

report = scan_last_bar({"STMMI.MI": ohlcv_stmmi, "PRY.MI": ohlcv_pry})
# columns: ticker | date | display_name | signal

plot_pattern(ohlcv_df, "CDLENGULFING", symbol="SIE.DE")
explore_patterns(ohlcv_df, symbol="SIE.DE", output="save", output_dir="charts/")
```

**CLI:**

```bash
python -m techa.patterns SIE.DE
python -m techa.patterns SIE.DE 2020-01-01 2024-01-01
python -m techa.patterns SIE.DE 2020-01-01 2024-01-01 save
```

---

### `techa.ma` and `techa.breakout`

Analytics on relative-price data (ticker / benchmark) loaded from a pre-computed parquet. All smoothing uses manual Wilder implementations; no TA-Lib dependency.

```python
from techa.ma.trend_quality import assess_ma_trend
from techa.breakout.range_quality import assess_range
```

Slopes are OLS-based and always returned with R² (R² < 0.3 = noise-dominated).

---

## Insurance Domain

### `techa.insurance`

Financial KPI snapshot for an insurance entity. Input is a periodic DataFrame (quarterly or annual rows).

```python
from techa.insurance import build_kpi_snapshot

snap = build_kpi_snapshot(df)                  # dict of floats
snap = build_kpi_snapshot(df, nan_to_none=True, periods_per_year=4, trend_lookback=8)
```

**Required columns:** `gwp`, `claims_incurred`, `expenses`.

**Optional columns:** `net_written_premium`, `reinsurance_ceded`, `claims_outstanding`, `claims_settled`, `reserve_held`, `reserve_required`, `policies_in_force`, `new_business`.

Raises `ValueError` if `len(df) < 4`.

**Output groups (34 keys):**

| Group | Key examples |
|---|---|
| Profitability | `loss_ratio`, `expense_ratio`, `combined_ratio`, `underwriting_margin_pct`, `reinsurance_cession_pct`, `net_claims_ratio`, `loss_ratio_trend`, `loss_ratio_trend_r2` |
| Reserves | `reserve_adequacy_ratio`, `reserve_adequacy_pct`, `reserve_surplus`, `reserve_to_gwp_pct`, `claims_settlement_ratio`, `claims_outstanding_ratio`, `reserve_adequacy_trend`, `reserve_adequacy_trend_r2` |
| Growth | `gwp_latest`, `nwp_latest`, `premium_growth_pp`, `premium_growth_yoy`, `claims_growth_pp`, `claims_growth_yoy`, `gwp_cagr`, `gwp_trend`, `gwp_trend_r2`, `avg_premium`, `lapse_rate`, `new_business_ratio` |

---

### `techa.underwriting`

Medical underwriting KPI snapshot from an applicant questionnaire dict.

```python
from techa.underwriting import build_medical_snapshot

snap = build_medical_snapshot(questionnaire)
snap = build_medical_snapshot(questionnaire, nan_to_none=True)
```

**Required fields:** `age`, `gender`.

**Optional fields:** `height_cm`, `weight_kg` (or `bmi`), `systolic_bp`, `diastolic_bp`, `total_cholesterol`, `hdl_cholesterol`, `fasting_glucose`, `hba1c`, `smoker`, `smoking_status`, `cigarettes_per_day`, `years_smoked`, `years_quit`, `alcohol_units_per_week`, `medical_history` (list), `medications` (list), `family_history` (list), `family_history_age_at_onset` (dict), `occupation_class` (1–4).

**Output groups:**

| Group | Key examples |
|---|---|
| Biometrics | `bmi`, `bmi_category`, `bmi_loading_pct` |
| Cardiovascular | `bp_systolic`, `bp_diastolic`, `pulse_pressure`, `bp_category`, `bp_loading_pct`, `cholesterol_ratio`, `cholesterol_risk`, `cholesterol_loading_pct`, `cv_risk_score` |
| Metabolic | `diabetes_status`, `hba1c`, `hba1c_category`, `fasting_glucose`, `glucose_category`, `metabolic_loading_pct` |
| Lifestyle | `smoking_status`, `pack_years`, `cigarettes_per_day`, `years_quit`, `smoking_loading_pct`, `alcohol_units_per_week`, `alcohol_risk`, `alcohol_loading_pct` |
| Conditions | `condition_count`, `conditions_loading_pct`, `critical_condition_flag` |
| Family history | `family_risk_factor_count`, `family_history_loading_pct`, `hereditary_cancer_risk` |
| Aggregate | `occupation_class`, `occupation_loading_pct`, `total_medical_loading_pct` (capped at 250%), `risk_score` (0–100) |

---

### `techa.claims`

Claims assessment KPI snapshot from a structured claim form dict.

```python
from techa.claims import build_claims_snapshot

snap = build_claims_snapshot(claim_form)
snap = build_claims_snapshot(claim_form, nan_to_none=True)
```

**Required fields:** `claim_type` — one of `"death"`, `"critical_illness"`, `"total_permanent_disability"`, `"income_protection"`, `"medical_expense"`, `"hospital_cash"`, `"accident"`.

**Optional fields:** `date_of_event`, `date_of_submission`, `policy_inception_date`, `claim_amount_requested`, `sum_assured`, `premium_annual`, `diagnosis` (list), `icd_codes` (list), `admission_date`, `discharge_date`, `prognosis`, `treatment_summary`, `treating_physician`, `hospital_name`, `pre_existing_conditions_declared` (list), `medical_history_consistent`, `nondisclosure_flag`, `documents_submitted` (list).

**Output groups:**

| Group | Key examples |
|---|---|
| Timeline | `policy_age_days`, `policy_age_months`, `early_claim_flag`, `very_early_claim_flag`, `submission_delay_days`, `submission_delay_risk`, `inpatient_duration_days`, `timeline_loading_pct` |
| Severity | `claim_amount_requested`, `sum_assured`, `claim_to_sa_ratio`, `claim_to_premium_ratio`, `severity_category`, `severity_loading_pct` |
| Medical | `diagnosis_categories`, `icd_chapters`, `primary_diagnosis_category`, `claim_type_match`, `prognosis_risk`, `coherence_loading_pct` |
| Documentation | `documents_submitted_count`, `required_documents`, `missing_documents`, `documentation_completeness_pct`, `documentation_status`, `documentation_loading_pct` |
| Fraud | `fraud_flags`, `fraud_flag_descriptions`, `fraud_indicator_count`, `fraud_risk_level`, `fraud_loading_pct` |
| Aggregate | `claim_type`, `claims_loading_pct` (capped at 100%), `claims_risk_level` |

---

### `techa.actuarial`

Three independent actuarial snapshot builders.

#### Actual vs Expected (A/E) monitoring

```python
from techa.actuarial import build_ae_snapshot

snap = build_ae_snapshot(ae_data)
# ae_data = {"periods": [{"actual_claims": ..., "expected_claims": ...}, ...]}
```

**Output:** `period_count`, `actual_total`, `expected_total`, `aggregate_ae_ratio`, `aggregate_ae_pct`, `ae_alert_level` (`green`/`amber`/`red`), `credibility_weight` (Bühlmann Z), `credibility_weighted_ae`, `z_score`, `z_score_significant`, `cumulative_deviation`, `max/min_ae_ratio`, `periods_above_expected`, `ae_trend_slope`, `ae_trend_r2`, `ae_trend_direction` (`improving`/`stable`/`deteriorating`), `ae_volatility`.

#### Reinsurance deal pricing

```python
from techa.actuarial import build_pricing_snapshot

snap = build_pricing_snapshot(pricing_data)
# pricing_data = {
#     "cash_flows": [{"ceded_premium": ..., "ceded_claims": ..., "expenses": ..., ...}, ...],
#     "treaty_type": "quota_share",
#     "discount_rate": 0.05,
# }
```

**Output:** `term_years`, `discount_rate`, `total_ceded_premium/claims/expenses/commission/profit`, `npv`, `payback_period_years`, `loss_ratio`, `expense_ratio`, `commission_ratio`, `profit_margin`, `break_even_loss_ratio`, `stress_ae25/50_loss_ratio`, `stress_ae25/50_profit_margin`, `pricing_adequacy` (`adequate`/`marginal`/`inadequate`), `cumulative_cash_flows`.

#### In-force portfolio health

```python
from techa.actuarial import build_inforce_snapshot

snap = build_inforce_snapshot(inforce_data)
# inforce_data = {
#     "periods": [{"policies_in_force": ..., "gross_premium_income": ..., ...}, ...],
#     "periods_per_year": 4,
# }
```

**Output:** `period_count`, `pif_latest`, `gwp_latest`, `pif_cagr`, `gwp_cagr`, `avg_lapse_rate`, `persistency_rate`, `lapse_trend_slope/r2/direction`, `avg_mortality_rate_ppm`, `new_business_ratio`, `bel_latest`, `risk_margin_latest`, `scr_latest`, `solvency_coverage_ratio`, `solvency_status` (`adequate`/`watch`/`breach`), `bel_to_annualised_gwp`, `risk_margin_ratio`, `bel_trend_slope/r2`.

---

## Agents

All agents use LangGraph Send-based fan-out and OpenAI structured output. Set `OPENAI_API_KEY` before use. Worker calls use `gpt-4.1-nano`; synthesis uses `gpt-4o`.

### `techa.agents.indicators` — Indicator Analysis Agent

Single-ticker analysis: trend (MA alignment), momentum (MACD / stochastic / ROC), and volatility & volume flow (ATR / BB / Chaikin). Three workers in parallel; synthesis via `gpt-4o`.

```python
from techa.agents.indicators import create_indicator_agent

graph = create_indicator_agent("PST.MI")                                              # live
graph = create_indicator_agent("ENI.MI", data_source="parquet", analysis_date="2024-06-30")
result = graph.invoke(graph._initial_state)
print(result["final_output"])
```

| Argument | Default | Description |
|---|---|---|
| `symbol` | required | Ticker to analyse |
| `data_source` | `"live"` | `"live"` (yfinance) or `"parquet"` |
| `analysis_date` | `None` | ISO date ceiling; `None` → latest bar (parquet mode only) |
| `lookback_days` | `365` | History to download (live mode only) |
| `checkpointer` | `None` | LangGraph checkpointer |

---

### `techa.agents.ta` — Technical Analysis Agent

Single-ticker MA crossovers and breakout analysis.

```python
from techa.agents.ta import create_manager

graph = create_manager("A2A.MI", analysis_date="2024-06-30")           # parquet (default)
graph = create_manager("ENI.MI", data_source="live", benchmark="FTSEMIB.MI")
result = graph.invoke(graph._initial_state)
print(result["final_output"])
```

| Argument | Default | Description |
|---|---|---|
| `symbol` | required | Ticker to analyse |
| `data_source` | `"parquet"` | `"parquet"` or `"live"` |
| `analysis_date` | `None` | ISO date ceiling |
| `benchmark` | `"FTSEMIB.MI"` | Benchmark for relative-price computation (live mode) |
| `fx` | `None` | Optional FX ticker for currency conversion |
| `relative` | `False` | Use relative prices (stock / benchmark) in live mode |
| `checkpointer` | `None` | LangGraph checkpointer |

---

### `techa.agents.patterns` — Candlestick Pattern Scan Agent

Multi-ticker last-bar pattern scan with structured per-ticker analysis.

```python
from techa.agents.patterns import create_pattern_agent

graph = create_pattern_agent(["A2A.MI", "ENI.MI"], analysis_date="2024-06-30")
graph = create_pattern_agent(["A2A.MI"], data_source="live", signal_filter="bear")
result = graph.invoke(graph._initial_state)
print(result["final_output"])
```

| Argument | Default | Description |
|---|---|---|
| `tickers` | required | List of tickers to scan |
| `data_source` | `"parquet"` | `"parquet"` or `"live"` |
| `analysis_date` | `None` | ISO date ceiling (parquet mode only) |
| `signal_filter` | `"all"` | `"all"`, `"bull"`, or `"bear"` |
| `lookback_days` | `365` | History to download (live mode only) |
| `lookback_bars` | `20` | Recent pattern history bars sent to the model |
| `checkpointer` | `None` | LangGraph checkpointer |

---

### `techa.agents.orchestrator` — Orchestrator Agent

Single-ticker orchestrator: loads OHLCV once and fans out in parallel to `indicators`, `patterns`, and `ta` agents; final synthesis via `gpt-4o`.

```python
from techa.agents.orchestrator import create_orchestrator

graph = create_orchestrator("PST.MI")                                           # live (default)
graph = create_orchestrator("A2A.MI", data_source="parquet", analysis_date="2024-06-30")
result = graph.invoke(graph._initial_state)
print(result["final_output"])
```

| Argument | Default | Description |
|---|---|---|
| `symbol` | required | Ticker to analyse |
| `data_source` | `"live"` | `"live"` or `"parquet"` |
| `analysis_date` | `None` | ISO date ceiling (parquet mode only) |
| `lookback_days` | `365` | History to download (live mode only) |
| `benchmark` | `"FTSEMIB.MI"` | Benchmark ticker (ta runner) |
| `fx` | `None` | Optional FX ticker |
| `relative` | `False` | Use relative prices in live mode (ta runner) |
| `checkpointer` | `None` | LangGraph checkpointer |

**`final_output` sections:** Position Recommendation, Signal Confluence Scorecard, Indicators Deep-Dive, Candlestick Patterns, TA Deep-Dive, Entry & Exit Plan, Bottom Line.

---

### `techa.agents.insurance` — Insurance Risk Assessment Agent

Single-policy risk assessment: four workers run in parallel via LangGraph Send dispatch; a Life Head of Business (`gpt-4o`) synthesises all assessments.

```python
from techa.agents.insurance import create_insurance_agent

graph = create_insurance_agent("POL-00123", risk_profile=risk_profile)
result = graph.invoke(graph._initial_state)
print(result["final_output"])
```

**`risk_profile` input sections:**

| Section | Used by | Converted to |
|---|---|---|
| `applicant` | All workers | Applicant demographics and coverage context |
| `coverage` | All workers | Policy terms |
| `claims_history` | `actuarial`, `claims_assessor` | Historical claims context |
| `financial_metrics` | `accountant` | Pre-computed financial KPIs |
| `claim_form` | `claims_assessor` | `build_claims_snapshot()` |
| `financial_history` | `accountant` | `build_kpi_snapshot()` |
| `ae_data` | `actuarial` | `build_ae_snapshot()` |
| `pricing_data` | `actuarial` | `build_pricing_snapshot()` |
| `inforce_data` | `actuarial` | `build_inforce_snapshot()` |
| `questionnaire` | `medical_underwriting` | `build_medical_snapshot()` |

**Workers and structured output:**

| Worker | Pydantic model fields |
|---|---|
| `actuarial` | `mortality_percentile`, `expected_loss_ratio`, `mortality_loading_pct`, `risk_classification` (`standard`/`substandard`/`postpone`/`decline`) |
| `accountant` | `financial_health`, `combined_ratio`, `loss_ratio`, `reserve_status`, `profitability_outlook`, `financial_loading_pct` |
| `medical_underwriting` | `underwriting_decision`, `medical_loading_pct`, BMI / smoker / occupation loadings, `exclusions`, `additional_requirements` |
| `claims_assessor` | `claims_risk_level`, `frequency_risk`, `severity_risk`, `fraud_flags`, `claims_loading_pct` |

---

## Development

```bash
python -m pytest techa/indicators/tests/   # indicator tests
python -m pytest techa/ma/tests/           # MA tests
python -m pytest techa/breakout/tests/     # breakout tests
python -m pytest techa/                    # all tests
```
