# Insurance Agent — Enhancement Roadmap

## Current state

Four parallel workers (actuarial, accountant, medical_underwriting, claims_assessor) each consume
pre-built snapshots produced by `build_payload()` and write one `WorkerResult` to `state["results"]`.
`synthesise_node` formats all four results into a single narrative `final_output` string.

---

## Phase 1 — Structured decision output  *(highest leverage, lowest effort)*

**Goal:** replace the narrative-only `final_output` with a machine-readable `DecisionRecord` alongside it,
so downstream policy-admin systems can consume the agent's recommendation without parsing free text.

### Changes

#### `techa/agents/insurance/_tools/decision_record.py` (new)
```python
class DecisionRecord(BaseModel):
    decision:            Literal["accept", "refer", "decline"]
    recommended_loading_pct: float          # blended loading from all workers
    conditions:          list[str]          # e.g. ["exclusion: pre-existing cardiac", "annual review"]
    review_date:         str | None         # ISO date — None if accept with no review required
    primary_driver:      str                # one sentence
    confidence:          Literal["high", "medium", "low"]
```

#### `graph_state.py`
Add `decision: DecisionRecord | None` field with `_last` reducer.

#### `synthesise_node`
- Derive `DecisionRecord` deterministically from the four `WorkerResult` dicts (no extra LLM call needed):
  - `decision = "decline"` if any worker returns `risk_classification == "decline"` or `fraud_risk_level == "very_high"`
  - `decision = "refer"` if any returns `postpone` or `fraud_risk_level == "high"`
  - `decision = "accept"` otherwise
  - `recommended_loading_pct` = weighted mean of per-worker loadings (actarial 40%, medical 35%, claims 25%)
- Keep existing narrative `final_output` unchanged.

### Acceptance criteria
- `state["decision"]` is a populated `DecisionRecord` on every run.
- `"decline"` or `"refer"` decisions are never overridden by a single low-conviction worker.

---

## Phase 2 — Two-wave conditional dispatch  *(fraud triage gate)*

**Goal:** run a lightweight fraud triage pass first; if `fraud_risk_level` is `high` or `very_high`,
deepen the medical underwriting and activate the legal compliance worker before synthesis.

### Graph topology change

```
START → prepare → fraud_triage → _dispatcher(state) → worker_node → synthesise → END
                       ↓
              if high/very_high fraud:
                  → deep_medical_node
                  → legal_compliance_node
```

#### `fraud_triage` node
- Calls `build_claims_snapshot` (already computed in `payload["claims_snapshot"]`) — no new LLM call.
- Writes `state["fraud_risk_level"]` (`Annotated[str, _last]`).
- Returns `Send` objects: always emit actuarial + accountant; conditionally add deep_medical + legal.

#### `_tools/ask_legal_compliance.py` (new)
```python
class LegalComplianceAnalysis(BaseModel):
    policy_age_at_event_days: int
    exclusions_triggered:     list[str]
    nondisclosure_materiality: Literal["none", "minor", "material", "voiding"]
    eligibility_verdict:      Literal["eligible", "refer", "ineligible"]
    legal_notes:              list[str]
    verdict:                  str
```
System prompt covers: policy inception rules, standard exclusion clauses,
non-disclosure materiality test (IDD / ICOBS 8 principles).

#### `ask_medical_underwriter.py` — deep mode
Add `depth: Literal["standard", "deep"] = "standard"` param.
`deep` mode appends extra prompt section requesting specialist opinion triggers,
impairment-specific loading tables, and 12-month forward morbidity projection.

### Acceptance criteria
- Standard path (no fraud flags): same 4-worker parallel run as today.
- High-fraud path: 6 workers run; `synthesise_node` renders a dedicated fraud section.

---

## Phase 3 — New analytical snapshots

### 3a. Epidemiological enrichment snapshot  (`techa/actuarial/geo/snapshot.py`)

Attach regional risk indices to the applicant before actuarial analysis.

**Input** (`geo_data` key in `risk_profile`):
```python
{
    "postcode_area": "SW1",
    "imd_decile": 4,              # Index of Multiple Deprivation (1=most deprived)
    "regional_ae_index": 1.08,    # local mortality vs national (from ONS / CMI)
    "hospital_quality_score": 72  # CQC / ACSA rating 0–100
}
```

**Output keys** (`geo_snapshot`):
- `imd_loading_pct` — deprivation loading: decile 1–3 → +15%, 4–6 → +5%, 7–10 → 0%
- `regional_ae_adjustment` — multiplicative factor applied to ae_snapshot expected counts
- `hospital_access_loading_pct` — score < 50 → +5%
- `geo_risk_level` — low / elevated / high

**Wiring:** `build_payload()` builds `geo_snapshot` when `geo_data` key present;
`ask_actuarial_analyst` SYSTEM_PROMPT updated with Tool 4 section.

---

### 3b. Portfolio concentration snapshot  (`techa/actuarial/concentration/snapshot.py`)

Flags when a single policy is an outsized share of total in-force exposure.

**Input** (`portfolio_context` key in `risk_profile`):
```python
{
    "portfolio_total_sa":    850_000_000,
    "portfolio_gwp":         12_500_000,
    "portfolio_policy_count": 10_500,
    "policy_sum_assured":    500_000,
}
```

**Output keys** (`concentration_snapshot`):
- `sa_concentration_pct` — this policy / total SA
- `concentration_flag` — True if > 0.5% of portfolio SA
- `net_retention_recommendation` — suggested retention limit (£)
- `reinsurance_trigger` — True if sum_assured exceeds retention recommendation

**Wiring:** fed into `ask_actuarial_analyst` alongside `pricing_snapshot`.

---

### 3c. Applicant health trend snapshot  (`techa/underwriting/trend/snapshot.py`)

Replaces the current single-point-in-time applicant dict with longitudinal data.

**Input** (`health_history` key in `risk_profile`):
```python
[
    {"year": 2021, "bmi": 29.1, "systolic_bp": 128, "total_cholesterol": 5.2},
    {"year": 2022, "bmi": 29.8, "systolic_bp": 131, "total_cholesterol": 5.5},
    {"year": 2023, "bmi": 30.6, "systolic_bp": 135, "total_cholesterol": 5.9},
]
```

**Output keys** (`health_trend_snapshot`):
- `bmi_trend_slope`, `bmi_trend_direction` — improving / stable / deteriorating
- `bp_trend_slope`,  `bp_trend_direction`
- `cholesterol_trend_slope`, `cholesterol_trend_direction`
- `trajectory_loading_pct` — extra loading for deteriorating trend: 0–40%
- `trend_conviction` — high (≥3 years) / medium (2 years) / low (1 year)

**Wiring:** `build_payload()` builds when `health_history` key present;
`ask_medical_underwriter` SYSTEM_PROMPT updated with new snapshot section.

---

## Phase 4 — Monte Carlo stress testing  (`techa/actuarial/stress/snapshot.py`)

Replaces the two-point A/E stress in `pricing_snapshot` with a full loss distribution.

**Input:** reuses `pricing_data` + new `stress_config`:
```python
{
    "n_simulations": 10_000,
    "ae_volatility":  0.12,     # std dev of annual A/E ratio
    "correlation":    0.30,     # between mortality and lapse shocks
}
```

**Output keys** (`stress_snapshot`):
- `mean_loss_ratio`, `p75_loss_ratio`, `p95_loss_ratio`, `p99_loss_ratio`
- `var_95_profit_margin`, `tvar_95_profit_margin`
- `ruin_probability` — P(cumulative loss > total premium)
- `adequate_at_p75` — bool: profit margin positive at 75th percentile
- `stress_verdict` — adequate / marginal / inadequate

**Implementation:** pure NumPy — `np.random.default_rng().lognormal(...)` for A/E draws;
vectorised NPV via broadcast; no scipy dependency.

---

## Phase 5 — Human-in-the-loop interrupt

**Goal:** pause the graph before synthesis for high-risk decisions and require underwriter sign-off.

### Trigger condition
`fraud_risk_level in ("high", "very_high")` OR `risk_classification in ("postpone", "decline")`.

### Mechanism
LangGraph `interrupt()` API — insert an `interrupt_node` between `worker_node` and `synthesise`:

```python
from langgraph.types import interrupt

def interrupt_node(state):
    if _requires_review(state):
        decision = interrupt({
            "reason":        "High-risk case requires underwriter review",
            "worker_results": state["results"],
            "suggested_action": _draft_decision(state),
        })
        return {"human_override": decision}
    return {}
```

`synthesise_node` checks `state.get("human_override")` and incorporates the underwriter's
input into `final_output` and `DecisionRecord`.

### Deployment requirement
Graph must be compiled with a `checkpointer` (e.g. `MemorySaver` for local,
`AsyncPostgresSaver` for production) to persist state across the interrupt boundary.

---

## Phase 6 — Batch mode & audit trail

### 6a. Batch entry point  (`techa/agents/insurance/batch.py`)

```python
async def run_batch(
    cases: list[tuple[str, dict | None]],
    max_concurrency: int = 10,
) -> list[dict]:
    """
    Run the insurance graph concurrently over a list of (policy_id, risk_profile) pairs.
    Returns list of final states (includes decision + final_output).
    """
```

Uses `asyncio.gather` + `graph.ainvoke` with a semaphore to cap concurrency.

### 6b. Audit trail serialisation  (`techa/agents/insurance/_audit.py`)

After every `synthesise_node` run, append one row to
`data/results/insurance/decisions.parquet`:

| Column | Type | Description |
|---|---|---|
| `run_id` | str | `uuid4()` |
| `policy_id` | str | |
| `assessment_date` | date | |
| `decision` | str | accept / refer / decline |
| `recommended_loading_pct` | float | |
| `worker_results_json` | str | JSON of all 4–6 WorkerResults |
| `final_output` | str | narrative text |
| `model` | str | MODEL constant at run time |

Enables regulatory audit, A/B model comparison, and back-testing loading recommendations
against actual claims experience.

---

## Priority summary

| Phase | Effort | Impact | Dependency |
|---|---|---|---|
| 1 — DecisionRecord output | Low | High | None |
| 2 — Two-wave fraud triage | Medium | High | Phase 1 |
| 3a — Geo enrichment | Low | Medium | None |
| 3b — Concentration snapshot | Low | Medium | None |
| 3c — Health trend snapshot | Medium | Medium | None |
| 4 — Monte Carlo stress | Medium | Medium | None |
| 5 — Human-in-the-loop | High | High | Phase 2 |
| 6 — Batch + audit trail | Medium | High | Phase 1 |

**Recommended sequence:** 1 → 3a/3b (parallel) → 2 → 6 → 3c → 4 → 5
