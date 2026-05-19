"""agents/insurance — LangGraph insurance risk assessment agent.

Four parallel specialists (actuarial, accountant, medical_underwriting, claims_assessor)
feed into a Life Head of Business synthesis that makes the final underwriting decision.
"""

from techa.agents.insurance.agent import create_insurance_agent

__all__ = ["create_insurance_agent"]
