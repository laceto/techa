"""
agents/insurance/agent.py — create_insurance_agent() graph factory.

Builds the LangGraph StateGraph:
  START → prepare_node → (Send dispatcher) → worker_node → synthesise_node → END

The dispatcher fans out to worker_node four times — once each for:
  actuarial, accountant, medical_underwriting, claims_assessor.

synthesise_node acts as Life Head of Business and makes the final underwriting decision.
Adding a new specialist requires only updating WORKER_NAMES in _subagents.py.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from techa.agents.insurance.graph_state import InsuranceAnalysisState
from techa.agents.insurance.graph_nodes import prepare_node, synthesise_node, worker_node
from techa.agents.insurance._subagents import WORKER_NAMES


def _dispatcher(state: InsuranceAnalysisState) -> list[Send]:
    """Fan out to worker_node once per registered specialist, injecting agent_id via Send."""
    return [Send("worker_node", {"agent_id": name, **state}) for name in WORKER_NAMES]


def create_insurance_agent(
    policy_id: str,
    risk_profile: dict | None = None,
    checkpointer=None,
):
    """
    Build and compile the InsuranceAnalysis LangGraph for a single policy application.

    Four specialists run in parallel (actuarial, accountant, medical_underwriting,
    claims_assessor), then synthesise_node acts as Life Head of Business and
    produces the final underwriting decision with loading factors and terms.

    Args:
        policy_id:    Unique policy or application reference (required).
        risk_profile: Structured insurance risk data dict — see graph_state.py for the
                      expected shape. If None, prepare_node uses a built-in demo profile
                      so the graph can run end-to-end without real data.
        checkpointer: Optional LangGraph checkpointer for persistence / resumption.

    Returns:
        CompiledStateGraph ready for .invoke() / .stream(). Exposes ._initial_state
        so callers can do: graph.invoke(graph._initial_state).

    Example:
        from techa.agents.insurance import create_insurance_agent

        profile = {
            "product_type": "term_life",
            "assessment_date": "2026-05-19",
            "applicant": {
                "age": 45, "gender": "male", "smoker": False, "bmi": 27.5,
                "occupation_class": 1, "systolic_bp": 130,
                "medical_history": ["hypertension"],
                "family_history": ["cardiovascular_disease"],
            },
            "coverage": {"sum_assured": 500000, "premium_annual": 3200, "term_years": 20},
            "claims_history": {"total_claims_count": 1, "total_claims_paid": 3200,
                               "largest_single_claim": 3200, "years_since_last_claim": 3},
            "financial_metrics": {"loss_ratio": 0.56, "expense_ratio": 0.28,
                                   "combined_ratio": 0.84, "reserve_adequacy_pct": 105.6},
        }
        g = create_insurance_agent("APP-2026-001", risk_profile=profile)
        r = g.invoke(g._initial_state)
        print(r["final_output"])
    """
    builder = StateGraph(InsuranceAnalysisState)

    builder.add_node("prepare",     prepare_node)
    builder.add_node("worker_node", worker_node)
    builder.add_node("synthesise",  synthesise_node)

    builder.add_edge(START, "prepare")
    builder.add_conditional_edges("prepare", _dispatcher)
    builder.add_edge("worker_node", "synthesise")
    builder.add_edge("synthesise", END)

    initial_state: InsuranceAnalysisState = {
        "policy_id":    policy_id,
        "risk_profile": risk_profile,
        "results":      [],
    }

    graph = builder.compile(checkpointer=checkpointer)
    graph._initial_state = initial_state
    return graph
