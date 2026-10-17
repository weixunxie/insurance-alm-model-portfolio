"""Shared model assembly used by scripts/run_all.py and the dashboard.

build_base: curve, key tenors, policy, universe (with capital factors), liabilities and
            expected cash flows per USD 1 of market value.
build_data: LP data for a universe, plus the policy's scenario / downgrade-buffer rows.
portfolio_analysis: Module A (cash flow gaps), Module B (KRD exposure, surplus scenarios) and
            capital for one set of holdings, without writing files.
"""

from dataclasses import dataclass

import pandas as pd

from alm.capital import add_capital_factors, capital_report
from alm.cashflow_matching import asset_cashflows, forward_rates, gap_summary, gap_table, per_dollar_cashflows
from alm.config import load_yaml
from alm.curve import SpotCurve, load_curve, load_key_tenors
from alm.krd import key_rate_dv01
from alm.krd_exposure import bucket_krdv01, duration_gap, krd_exposure_table, surplus_scenarios
from alm.liabilities import liability_price_fn, project_liabilities
from alm.optimizer import ProblemData, build_problem_data, krd_tolerances
from alm.overlay import add_overlay
from alm.scenario_constraints import apply_policy_constraints
from alm.universe import build_universe, cashflow_matrix, cashflow_table


@dataclass
class Base:
    curve: SpotCurve
    keys: list
    policy: dict
    universe: pd.DataFrame          # with cap_factor for the policy's CLO regime
    liab: pd.DataFrame
    per_dollar: pd.DataFrame        # (t, id) expected cash flow per USD 1 MV


def build_base(regime: str | None = None) -> Base:
    """Everything the LP needs that does not depend on the policy limits."""
    curve, keys, policy = load_curve(), load_key_tenors(), load_yaml("policy.yaml")
    universe = add_capital_factors(build_universe(curve, keys), regime or policy["capital"]["clo_regime"])
    per_dollar = per_dollar_cashflows(cashflow_matrix(cashflow_table(universe, curve), "expected"),
                                      universe.set_index("id")["price"])
    return Base(curve, keys, policy, universe, project_liabilities(curve), per_dollar)


def build_data(u, per_dollar, liab, curve, keys, policy, regime="new_2026", overlay=False,
               scenario_on=None, buffer_on=None, hy_after_on=None) -> ProblemData:
    """LP data for universe u (with cap factors), plus the policy's scenario / downgrade-buffer rows."""
    lk = key_rate_dv01(liability_price_fn(liab), curve, keys)
    d = build_problem_data(u, per_dollar, liab.total.to_numpy(), lk, forward_rates(curve, len(liab)), keys)
    if overlay:
        d = add_overlay(d, curve, load_yaml("overlay.yaml"))
    return apply_policy_constraints(d, u, liab, curve, policy, load_yaml("scenarios.yaml"), regime,
                                    scenario_on, buffer_on, hy_after_on)


def portfolio_analysis(holdings, curve, keys, universe, liab, per_dollar, policy) -> dict:
    """Module A + Module B + capital for one portfolio (USD millions; KRD tables in USD millions per bp)."""
    funding, rr = policy["cash_flow_funding"], policy["rate_risk"]
    gaps = gap_table(asset_cashflows(holdings, per_dollar).to_numpy(), liab.total.to_numpy(), curve,
                     funding["borrow_spread"], funding["beginning_cash"])
    lpf = liability_price_fn(liab)
    exposure = krd_exposure_table(holdings, universe, lpf, curve, keys)
    exposure["tolerance"] = krd_tolerances(exposure.liability_krdv01 * 1000, rr["krd_floor"],
                                           rr["krd_pct_of_liability"]) / 1000
    scen = surplus_scenarios(holdings, universe, lpf, curve, keys, load_yaml("scenarios.yaml")["rate_scenarios"],
                             exposure)
    cap_rows, cap = capital_report(holdings, universe, policy["capital"]["budget"])
    return {"holdings": holdings, "gaps": gaps, "gap_stats": gap_summary(gaps, funding["check_years"]),
            "exposure": exposure, "buckets": bucket_krdv01(holdings, universe, keys),
            "dgap": duration_gap(exposure, holdings.sum(), lpf(curve)), "scen": scen,
            "cap_rows": cap_rows, "cap": cap}
