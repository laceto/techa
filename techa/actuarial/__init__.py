"""
techa/actuarial — Actuarial KPI snapshot builders.

Five independent snapshot tools:

1. build_ae_snapshot(ae_data, *, nan_to_none) -> dict
   Actual vs Expected (A/E) experience monitoring.
   Computes aggregate A/E ratio, credibility weighting, Poisson Z-score,
   alert level (green/amber/red), A/E trend direction, and volatility.
   Required: periods list with actual_claims and expected_claims per period.

2. build_pricing_snapshot(pricing_data, *, nan_to_none) -> dict
   New reinsurance deal pricing from projected cash flows.
   Computes loss/expense/commission/profit ratios, NPV, payback period,
   break-even loss ratio, and A/E +25%/+50% stress scenarios.
   Required: cash_flows list with ceded_premium and ceded_claims per year.

3. build_inforce_snapshot(inforce_data, *, nan_to_none) -> dict
   In-force portfolio health assessment.
   Computes lapse rate, persistency, mortality rate, portfolio growth CAGR,
   new business ratio, Solvency II coverage ratio, and BEL trend.
   Required: periods list with policies_in_force and gross_premium_income.

4. build_geo_snapshot(geo_data, *, nan_to_none) -> dict
   Geospatial / epidemiological risk enrichment.
   Computes IMD deprivation loading, regional A/E adjustment, and hospital
   access loading. All fields optional — safe defaults applied when absent.

5. build_concentration_snapshot(portfolio_context, *, nan_to_none) -> dict
   Portfolio concentration risk assessment.
   Flags when a single policy is an outsized share of in-force exposure and
   recommends a net retention limit for reinsurance structuring.
   Required: portfolio_total_sa, portfolio_policy_count, policy_sum_assured.
"""

from techa.actuarial.ae.snapshot            import build_ae_snapshot
from techa.actuarial.pricing.snapshot       import build_pricing_snapshot
from techa.actuarial.inforce.snapshot       import build_inforce_snapshot
from techa.actuarial.geo.snapshot           import build_geo_snapshot
from techa.actuarial.concentration.snapshot import build_concentration_snapshot

__all__ = [
    "build_ae_snapshot",
    "build_pricing_snapshot",
    "build_inforce_snapshot",
    "build_geo_snapshot",
    "build_concentration_snapshot",
]
