import copy

import cvxpy as cp
import numpy as np
import pytest

from alm.capital import add_capital_factors
from alm.cashflow_matching import forward_rates, funding_balance, per_dollar_cashflows
from alm.config import load_yaml
from alm.curve import load_curve, load_key_tenors
from alm.krd import key_rate_dv01
from alm.liabilities import liability_price_fn, project_liabilities
from alm.optimizer import build_problem_data, build_rows, is_hy, solve, solve_or_diagnose
from alm.universe import build_universe, cashflow_matrix, cashflow_table

CURVE, KEYS = load_curve(), load_key_tenors()
POLICY = load_yaml("policy.yaml")


@pytest.fixture(scope="module")
def data():
    u = add_capital_factors(build_universe(CURVE, KEYS))
    per_dollar = per_dollar_cashflows(cashflow_matrix(cashflow_table(u, CURVE)), u.set_index("id").price)
    liab = project_liabilities(CURVE)
    lk = key_rate_dv01(liability_price_fn(liab), CURVE, KEYS)
    return build_problem_data(u, per_dollar, liab.total.to_numpy(), lk, forward_rates(CURVE, 30), KEYS)


@pytest.fixture(scope="module")
def base(data):
    return solve(data, POLICY, POLICY["capital"]["budget"])


def test_solution_satisfies_every_constraint(data, base):
    assert base["status"] == cp.OPTIMAL
    x = base["x"].to_numpy()
    assert x.sum() == pytest.approx(100.0, abs=1e-6)
    assert np.all(x >= -1e-9)
    for r in build_rows(data, POLICY, POLICY["capital"]["budget"]):
        if r.sense == "==":
            assert r.a @ x == pytest.approx(r.b, abs=1e-6)
        else:
            assert r.a @ x <= r.b + 1e-6, r.name


def test_independent_policy_checks(data, base):
    x = base["x"].to_numpy()
    assert data.cap_factor @ x <= 1.2 + 1e-6
    assert x[is_hy(data.designation)].sum() <= 10 + 1e-6
    assert x[np.isin(data.bucket, ["RMBS", "CMBS", "ABS", "CLO"])].sum() <= 30 + 1e-6
    assert x[data.bucket == "AGENCY_MBS"].sum() <= 20 + 1e-6
    gaps = data.cf @ x - data.liab_cf
    assert np.all(funding_balance(gaps, data.fwd)[:10] >= -1e-6)    # recursion, independent of the matrix
    net = data.krd.T @ x - data.liab_krd
    assert abs(net.sum()) <= 5 + 1e-6


def test_capital_shadow_price_matches_finite_difference(data, base):
    dual = base["constraints"].set_index("constraint").loc["capital", "dual"]
    h = 0.01
    up = solve(data, POLICY, 1.2 + h)["objective"]
    dn = solve(data, POLICY, 1.2 - h)["objective"]
    assert dual == pytest.approx((up - dn) / (2 * h), rel=0.02)


def test_infeasible_policy_is_diagnosed(data):
    bad = copy.deepcopy(POLICY)
    bad["limits"]["liquidity_min"] = 101.0                 # impossible with a $100M budget
    res = solve_or_diagnose(data, bad, 1.2)
    assert res["hard_status"] in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE)
    v = res["violations"].set_index("constraint")
    assert "liquidity_min" in v.index
    assert v.loc["liquidity_min", "slack_used"] == pytest.approx(1.0, abs=1e-6)


def test_shadow_cost_breakdown_reduced_costs(data, base):
    from alm.optimizer import shadow_cost_breakdown
    held = base["x"][base["x"] > 1e-6].index
    # trading between two held positions: reduced cost 0 (complementary slackness)
    t = shadow_cost_breakdown(data, POLICY, base, {"HY_BB+_5Y": 1.0, "IG_BBB_10Y": -1.0})
    assert "HY_BB+_5Y" in held and "IG_BBB_10Y" in held
    assert t.attrs["reduced_cost"] == pytest.approx(0.0, abs=1e-9)
    assert "hy_max" in set(t.constraint)
    # adding a position the LP chose not to hold: reduced cost <= 0
    for sec in set(data.ids) - set(held):
        t = shadow_cost_breakdown(data, POLICY, base, {sec: 1.0})
        assert t.attrs["reduced_cost"] <= 1e-9, sec
