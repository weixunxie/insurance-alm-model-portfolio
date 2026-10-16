import cvxpy as cp
import numpy as np
import pytest

from alm.capital import add_capital_factors
from alm.cashflow_matching import forward_rates, per_dollar_cashflows
from alm.config import load_yaml
from alm.curve import load_curve, load_key_tenors
from alm.krd import key_rate_dv01
from alm.liabilities import liability_price_fn, project_liabilities
from alm.optimizer import budget_shadow_price, build_problem_data, build_rows, solve
from alm.overlay import add_overlay
from alm.scenario_constraints import apply_policy_constraints
from alm.stress import run_scenario
from alm.universe import build_universe, cashflow_matrix, cashflow_table

CURVE, KEYS = load_curve(), load_key_tenors()
POLICY = load_yaml("policy.yaml")
SCEN = load_yaml("scenarios.yaml")


@pytest.fixture(scope="module")
def env():
    u = add_capital_factors(build_universe(CURVE, KEYS))
    per_dollar = per_dollar_cashflows(cashflow_matrix(cashflow_table(u, CURVE)), u.set_index("id").price)
    liab = project_liabilities(CURVE)
    lk = key_rate_dv01(liability_price_fn(liab), CURVE, KEYS)
    d0 = build_problem_data(u, per_dollar, liab.total.to_numpy(), lk, forward_rates(CURVE, 30), KEYS)
    d = apply_policy_constraints(d0, u, liab, CURVE, POLICY, SCEN)
    return u, liab, d0, d


def test_scenario_rows_match_full_repricing(env):
    u, liab, _, d = env
    res = solve(d, POLICY, 1.2)
    x = res["x"][res["x"] > 1e-9]
    for name, dp, dl in zip(d.extra["scenario_names"], d.extra["scenario_dp"], d.extra["scenario_dl"]):
        r = run_scenario(x, u, liab, CURVE, KEYS, {"curve": SCEN["rate_scenarios"][name]}, POLICY)
        lp_ds = float(dp @ res["x"].to_numpy()) - dl
        assert lp_ds == pytest.approx(r["dS"], abs=1e-9), name                 # linear row == stress engine
        assert lp_ds >= -POLICY["scenario_constraints"]["max_loss"] - 1e-6


def test_downgrade_buffer_matches_stress_engine(env):
    u, liab, _, d = env
    res = solve(d, POLICY, 1.2)
    x = res["x"][res["x"] > 1e-9]
    r = run_scenario(x, u, liab, CURVE, KEYS, {"downgrade": POLICY["downgrade_buffer"]["rule"]}, POLICY)
    assert float(d.extra["cap_factor_downgraded"] @ res["x"].to_numpy()) == pytest.approx(r["capital_charge"], abs=1e-9)
    assert r["capital_charge"] <= POLICY["downgrade_buffer"]["multiple"] * 1.2 + 1e-6


def test_all_rows_satisfied_and_constraints_cost_income(env):
    _, _, d0, d = env
    r0, r1 = solve(d0, POLICY, 1.2), solve(d, POLICY, 1.2)
    assert r1["status"] == cp.OPTIMAL
    assert r1["objective"] <= r0["objective"] + 1e-9
    x = r1["x"].to_numpy()
    for row in build_rows(d, POLICY, 1.2):
        assert row.a @ x <= row.b + 1e-6 if row.sense == "<=" else row.a @ x == pytest.approx(row.b, abs=1e-6)


def test_budget_shadow_price_with_buffer_matches_finite_difference(env):
    _, _, _, d = env
    res = solve(d, POLICY, 1.2)
    h = 0.01
    fd = (solve(d, POLICY, 1.2 + h)["objective"] - solve(d, POLICY, 1.2 - h)["objective"]) / (2 * h)
    assert budget_shadow_price(d, res) == pytest.approx(fd, rel=0.02)


def test_overlay_gets_scenario_rows_for_swaps(env):
    u, liab, d0, _ = env
    d = apply_policy_constraints(add_overlay(d0, CURVE, load_yaml("overlay.yaml")), u, liab, CURVE, POLICY, SCEN)
    i = d.ids.index("SWAP_RECV_30Y")
    k = d.extra["scenario_names"].index("parallel_down_100")
    assert d.extra["scenario_dp"][k][i] > 0                     # receiver gains when rates fall
    assert d.extra["cap_factor_downgraded"][i] == 0.0


def test_optional_hy_after_downgrade_limit(env):
    from alm.migration import downgrade_rule, migrate
    from alm.optimizer import is_hy
    u, liab, d0, _ = env
    d = apply_policy_constraints(d0, u, liab, CURVE, POLICY, SCEN, hy_after_on=True)
    res = solve(d, POLICY, 1.2)
    x = res["x"][res["x"] > 1e-9]
    sub = u.set_index("id", drop=False).loc[x.index]
    migrated, _ = migrate(sub, downgrade_rule(sub, POLICY["downgrade_buffer"]["rule"]))
    hy_after = float(x[is_hy(migrated.naic_designation)].sum())
    assert hy_after <= POLICY["limits"]["hy_max"] + 1e-6
    assert res["objective"] <= solve(apply_policy_constraints(d0, u, liab, CURVE, POLICY, SCEN), POLICY, 1.2)["objective"]
