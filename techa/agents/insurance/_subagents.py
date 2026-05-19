"""
agents/insurance/_subagents.py — Worker registry for the InsuranceAnalysis agent.

WORKER_NAMES is the single source of truth for which specialist dimensions run.
Adding a new specialist requires only adding one entry here — the dispatcher
in agent.py fans out dynamically via Send; no graph wiring changes needed.
"""

from __future__ import annotations

from techa.agents.insurance._tools.ask_actuarial_analyst   import ask_actuarial_analyst
from techa.agents.insurance._tools.ask_accountant          import ask_accountant
from techa.agents.insurance._tools.ask_medical_underwriter import ask_medical_underwriter
from techa.agents.insurance._tools.ask_claims_assessor     import ask_claims_assessor

WORKER_NAMES: list[str] = [
    "actuarial",
    "accountant",
    "medical_underwriting",
    "claims_assessor",
]

WORKER_REGISTRY: dict = {
    "actuarial":            ask_actuarial_analyst,
    "accountant":           ask_accountant,
    "medical_underwriting": ask_medical_underwriter,
    "claims_assessor":      ask_claims_assessor,
}
