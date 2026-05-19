"""
agents/insurance/graph_nodes.py — Node implementations for the InsuranceAnalysis graph.

Nodes:
  prepare_node        — validates and enriches the risk profile; stores payload.
  fraud_triage_node   — reads claims_snapshot["fraud_risk_level"] from the pre-built payload;
                        writes state["fraud_risk_level"] so the dispatcher can activate
                        the legal_compliance worker for high/very_high fraud cases.
  worker_node         — single shared node dispatched by Send with agent_id injected;
                        calls the appropriate ask_* function and appends a WorkerResult.
  _call_synthesis_llm — compiles specialist reports into a final underwriting decision brief.
  synthesise_node     — reads state["results"], derives DecisionRecord, delegates to LLM
                        acting as Life Head of Business.

Invariant: state["payload"] is the sole data channel from prepare_node to worker_node.
           No worker reads from disk or network.
"""

from __future__ import annotations

import json
import logging

from techa.agents._common import get_result_by_id
from techa.agents._llm import SYNTHESIS_MODEL
from techa.agents.insurance.graph_state import InsuranceAnalysisState
from techa.agents.insurance._tools.prepare_tools import build_payload
from techa.agents.insurance._tools.decision_record import derive_decision
from techa.agents.insurance._subagents import WORKER_REGISTRY

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 3 — Underwriting decision compilation prompt (Life Head of Business)
# ---------------------------------------------------------------------------

_REPORT_SYSTEM = """\
You are the Life Head of Business at a major insurance company with 20+ years of underwriting
experience. You have received four independent assessments for policy application {policy_id}:
an actuarial analysis, an accounting analysis, a medical underwriting assessment, and a
claims history assessment.

Your audience is the Chief Underwriter and Risk Committee. Be direct, precise, and accountable.
Every loading and exclusion must be traceable to a specific finding in the assessments.
Do not fabricate data. If a figure is unavailable, say so explicitly.

---

## Underwriting Decision

State clearly:
  ACCEPT — STANDARD TERMS
  ACCEPT — WITH LOADING (state total loading %)
  REFER — TO CHIEF UNDERWRITER (state the reason)
  DECLINE (state the reason)

Two to three paragraphs covering:
- The primary risk driver that determined the decision.
- The specific finding that would change the decision (e.g., a satisfactory GP report,
  a 12-month deferral, or a reduced sum assured).
- Any exclusion clauses or special conditions attached to the acceptance.

---

## Risk Assessment Scorecard

Summary table:
| Specialist          | Finding                  | Loading Recommended | Conviction |
|---------------------|--------------------------|---------------------|------------|
| Actuarial           | …                        | X%                  | High/Med/Low |
| Accountant          | …                        | X%                  | High/Med/Low |
| Medical Underwriting| …                        | X%                  | High/Med/Low |
| Claims Assessor     | …                        | X%                  | High/Med/Low |
| **TOTAL LOADING**   | (after diversification)  | **X%**              |            |

Diversification note: where two loadings address the same underlying risk (e.g., BMI
loading from both actuarial and medical), apply the higher of the two, not both.

---

## Premium Calculation

- Base annual premium (from application): £X
- Actuarial loading (X%): +£X
- Medical loading (X%): +£X
- Claims loading (X%): +£X
- Financial loading (X%): +£X
- **Total adjusted annual premium**: **£X**
- Note any rounding or minimum premium thresholds.

---

## Actuarial Assessment Deep-Dive

1. **Risk classification** — standard / substandard / postpone / decline and why.
2. **Mortality percentile** — position within the standard insured population.
3. **Expected loss ratio** — at the recommended loading.
4. **Key mortality/morbidity drivers** — top 3 factors with loadings.

---

## Financial Viability Assessment

1. **Combined ratio** — current value and trend implication.
2. **Reserve adequacy** — status and regulatory position.
3. **Profitability outlook** — impact of this policy on the book.
4. **Repricing trigger** — at what loss ratio would a rate review be required?

---

## Medical Underwriting Assessment

1. **BMI & lifestyle** — category, loading, and commentary.
2. **Blood pressure** — category and loading.
3. **Medical conditions** — each condition, loading applied, and any exclusion.
4. **Occupation risk** — class and loading.
5. **Required evidence** — list any medical evidence needed before policy issuance.
6. **Exclusion clauses** — full text of each exclusion to be endorsed on the policy.

---

## Claims Risk Assessment

1. **Frequency** — claims count, risk level, and pattern.
2. **Severity** — largest claim relative to premium, severity risk level.
3. **Loss ratio** — assessment and trend.
4. **Fraud / suspicious patterns** — any flags and recommended action.

---

## Special Conditions & Exclusions

Full legal text of all exclusion clauses to be endorsed on the policy document.
Additional requirements before policy is issued (e.g., medical examinations, blood tests).
Review date — when the policy should be re-underwritten.

---

## Bottom Line

One paragraph: net underwriting conviction, the single most material risk factor,
and the specific event or evidence that would change the decision.
"""

_REPORT_HUMAN = """\
Policy application:       {policy_id}
Product:                  {product_type}
Assessment date:          {assessment_date}

Actuarial analysis:       {actuarial_analysis}

Accounting analysis:      {accounting_analysis}

Medical underwriting:     {medical_analysis}

Claims assessment:        {claims_analysis}{legal_addendum}
"""


# ---------------------------------------------------------------------------
# Fraud section for synthesis prompt (appended when legal_compliance ran)
# ---------------------------------------------------------------------------

_FRAUD_SECTION_TEMPLATE = """

---

## Legal Compliance & Fraud Assessment

Legal compliance assessment: {legal_analysis}

Summarise:
1. **Eligibility verdict** — eligible / refer / ineligible and the primary legal ground.
2. **Non-disclosure materiality** — none / minor / material / voiding.
3. **Exclusion clauses triggered** — list each applicable clause.
4. **Recommended action** — payment / refer to SIU / deny with legal notice.
"""


# ---------------------------------------------------------------------------
# prepare_node
# ---------------------------------------------------------------------------


def prepare_node(state: InsuranceAnalysisState) -> dict:
    """
    Validate and enrich the risk profile; store as the canonical payload dict.

    If state["risk_profile"] is None, the built-in demo profile is used.

    Args:
        state: Must contain "policy_id". Optional: "risk_profile".

    Returns:
        Dict updating "payload".
    """
    policy_id    = state["policy_id"]
    risk_profile = state.get("risk_profile")

    payload = build_payload(policy_id, risk_profile)

    log.info(
        "[prepare] policy=%s product=%s age=%s",
        policy_id,
        payload.get("product_type"),
        payload.get("applicant", {}).get("age"),
    )

    return {"payload": payload}


# ---------------------------------------------------------------------------
# fraud_triage_node — reads pre-built claims_snapshot; no LLM call
# ---------------------------------------------------------------------------


def fraud_triage_node(state: InsuranceAnalysisState) -> dict:
    """
    Read fraud_risk_level from the already-built claims_snapshot.

    This node adds zero latency — it only reads payload["claims_snapshot"] which
    was computed synchronously by build_payload() in prepare_node. The result is
    written to state["fraud_risk_level"] so the Send dispatcher can decide whether
    to activate the legal_compliance worker.

    Args:
        state: Must contain "payload" set by prepare_node.

    Returns:
        {"fraud_risk_level": str}  — one of: low / medium / high / very_high.
    """
    payload          = state.get("payload", {})
    claims_snapshot  = payload.get("claims_snapshot") or {}
    fraud_risk_level = claims_snapshot.get("fraud_risk_level", "low")

    log.info(
        "[fraud_triage] policy=%s fraud_risk_level=%s fraud_flags=%s",
        payload.get("policy_id"),
        fraud_risk_level,
        claims_snapshot.get("fraud_flags", []),
    )
    return {"fraud_risk_level": fraud_risk_level}


# ---------------------------------------------------------------------------
# worker_node — single shared node dispatched by Send
# ---------------------------------------------------------------------------


def worker_node(state: InsuranceAnalysisState) -> dict:
    """
    Call the appropriate specialist based on agent_id injected by the Send dispatcher.

    Never raises — exceptions are caught and stored as a WorkerResult with error set,
    so the other dispatched specialists can continue and synthesise_node always runs.

    Args:
        state: Must contain "agent_id" (injected by Send) and "payload" (set by prepare_node).

    Returns:
        {"results": [WorkerResult]} — appended to state via the add reducer.
    """
    agent_id  = state["agent_id"]
    payload   = state["payload"]
    policy_id = payload["policy_id"]

    try:
        worker_func = WORKER_REGISTRY.get(agent_id)
        if worker_func is None:
            raise ValueError(f"Unknown agent_id: {agent_id!r}")
        result = worker_func(payload, policy_id=policy_id)

        log.info("[worker] %s assessment complete for %s", agent_id, policy_id)
        return {"results": [{"agent_id": agent_id, "data": result.model_dump(), "error": None}]}

    except Exception as exc:
        log.error("[worker] %s failed: %s", agent_id, exc, exc_info=True)
        return {"results": [{"agent_id": agent_id, "data": {}, "error": str(exc)}]}


# ---------------------------------------------------------------------------
# synthesise_node — Life Head of Business
# ---------------------------------------------------------------------------


def _call_synthesis_llm(
    policy_id: str,
    product_type: str,
    assessment_date: str,
    actuarial_analysis: str,
    accounting_analysis: str,
    medical_analysis: str,
    claims_analysis: str,
    legal_addendum: str = "",
) -> str:
    """
    Call the LLM once as Life Head of Business to compile the final underwriting decision.

    Args:
        policy_id:           Application reference.
        product_type:        Insurance product type.
        assessment_date:     ISO date of assessment.
        actuarial_analysis:  JSON string of the actuarial worker result (or "unavailable").
        accounting_analysis: JSON string of the accountant worker result (or "unavailable").
        medical_analysis:    JSON string of the medical underwriting result (or "unavailable").
        claims_analysis:     JSON string of the claims assessor result (or "unavailable").
        legal_addendum:      Pre-formatted legal fraud section string (empty when not activated).

    Returns:
        Markdown string containing the underwriting decision brief.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages([
        ("system", _REPORT_SYSTEM),
        ("human",  _REPORT_HUMAN),
    ])
    llm = ChatOpenAI(model=SYNTHESIS_MODEL, temperature=0)

    response = (prompt | llm).invoke({
        "policy_id":           policy_id,
        "product_type":        product_type,
        "assessment_date":     assessment_date,
        "actuarial_analysis":  actuarial_analysis,
        "accounting_analysis": accounting_analysis,
        "medical_analysis":    medical_analysis,
        "claims_analysis":     claims_analysis,
        "legal_addendum":      legal_addendum,
    })
    return response.content if hasattr(response, "content") else str(response)


def synthesise_node(state: InsuranceAnalysisState) -> dict:
    """
    Compile all four specialist reports into a final underwriting decision brief.

    Acts as the Life Head of Business. Never raises — missing or errored specialist
    results are passed as "unavailable" so the LLM can still produce a partial report.
    """
    payload         = state.get("payload", {})
    policy_id       = payload.get("policy_id", state.get("policy_id", "unknown"))
    product_type    = payload.get("product_type", "unknown")
    assessment_date = payload.get("assessment_date", "unknown")

    def _fmt(agent_id: str) -> str:
        r = get_result_by_id(state.get("results", []), agent_id)
        if not r:
            return "unavailable"
        if r.get("error"):
            return f"unavailable — {r['error']}"
        return json.dumps(r["data"], indent=2)

    actuarial_str  = _fmt("actuarial")
    accounting_str = _fmt("accountant")
    medical_str    = _fmt("medical_underwriting")
    claims_str     = _fmt("claims_assessor")
    legal_str      = _fmt("legal_compliance")  # "unavailable" when legal worker not activated

    # Derive structured decision record deterministically — no LLM call needed
    results = state.get("results", [])
    try:
        decision_record = derive_decision(results, assessment_date)
        decision_dict   = decision_record.model_dump()
        log.info(
            "[synthesise] decision=%s loading=%.1f%% confidence=%s",
            decision_dict["decision"],
            decision_dict["recommended_loading_pct"],
            decision_dict["confidence"],
        )
    except Exception as exc:
        log.error("[synthesise] derive_decision failed: %s", exc, exc_info=True)
        decision_dict = None

    # Append legal fraud section to the human prompt when legal worker ran
    legal_addendum = (
        _FRAUD_SECTION_TEMPLATE.format(legal_analysis=legal_str)
        if legal_str != "unavailable"
        else ""
    )

    log.info(
        "[synthesise] generating underwriting decision for %s (legal=%s)",
        policy_id,
        "yes" if legal_addendum else "no",
    )
    try:
        brief = _call_synthesis_llm(
            policy_id,
            product_type,
            assessment_date,
            actuarial_str,
            accounting_str,
            medical_str,
            claims_str,
            legal_addendum=legal_addendum,
        )
    except Exception as exc:
        log.error("[synthesise] LLM call failed: %s", exc, exc_info=True)
        raw = json.dumps(
            {r["agent_id"]: r for r in results},
            indent=2,
            default=str,
        )
        brief = f"Synthesis failed: {exc}\n\nRaw specialist results:\n{raw}"

    log.info("[synthesise] underwriting brief generated (%d chars)", len(brief))
    return {"final_output": brief, "decision": decision_dict}
