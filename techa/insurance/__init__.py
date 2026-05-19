"""
techa/insurance — Insurance accountant KPI snapshot builder.

Primary entry point
-------------------
build_kpi_snapshot(df, *, nan_to_none, periods_per_year, trend_lookback) -> dict
    Last-period accountant KPI snapshot from an insurance financials time series.

    Required input columns:  gwp, claims_incurred, expenses.
    Optional input columns:  nwp, claims_paid, reserve_held, reserve_required,
                             policies_in_force, new_policies, lapsed_policies.

    Output groups:
        Profitability — loss_ratio, expense_ratio, combined_ratio, underwriting_margin_pct,
                        underwriting_profit, reinsurance_cession_pct, net_claims_ratio,
                        *_trend and *_trend_r2 for each ratio.
        Reserves      — reserve_adequacy_ratio, reserve_adequacy_pct, reserve_surplus,
                        reserve_to_gwp_pct, claims_settlement_ratio,
                        claims_outstanding, claims_outstanding_ratio,
                        reserve_adequacy_trend, reserve_adequacy_trend_r2.
        Growth        — gwp_latest, nwp_latest, premium_growth_pp, claims_growth_pp,
                        premium_growth_yoy, claims_growth_yoy, gwp_cagr,
                        gwp_trend, gwp_trend_r2, avg_premium, lapse_rate, new_business_ratio.
"""

from techa.insurance.snapshot import build_kpi_snapshot

__all__ = ["build_kpi_snapshot"]
