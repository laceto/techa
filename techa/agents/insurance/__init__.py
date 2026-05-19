"""agents/insurance — LangGraph insurance risk assessment agent.

Two-wave dispatch: four standard specialists (actuarial, accountant, medical_underwriting,
claims_assessor) always run; legal_compliance activates when fraud_risk_level is high/very_high.
synthesise_node acts as Life Head of Business and produces a narrative brief plus a
machine-readable DecisionRecord. Every run appends an audit row to decisions.parquet.

Public API:
    create_insurance_agent(policy_id, risk_profile, checkpointer) -> CompiledStateGraph
    run_batch(cases, max_concurrency) -> list[dict]  (async)
"""

from techa.agents.insurance.agent import create_insurance_agent
from techa.agents.insurance.batch import run_batch

__all__ = ["create_insurance_agent", "run_batch"]
