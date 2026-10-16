import numpy as np
import pandas as pd
import pytest

from alm.bonds import effective_convexity, effective_duration
from alm.capital import add_capital_factors
from alm.cashflow_matching import forward_rates, per_dollar_cashflows
from alm.config import load_yaml
from alm.curve import load_curve, load_key_tenors
from alm.krd import key_rate_dv01
from alm.liabilities import liability_price_fn, project_liabilities
from alm.migration import migrate, notch_down
from alm.optimizer import build_problem_data, build_rows, solve
from alm.overlay import SwapOverlay, add_overlay, swap_value_per_100
from alm.portfolios import load_rule_based
from alm.stress import run_scenario
from alm.structured import rate_dependent_cpr
from alm.universe import build_universe, cashflow_matrix, cashflow_table, load_assumptions, price_fn, solve_coupon

CURVE, KEYS = load_curve(), load_key_tenors()
POLICY = load_yaml("policy.yaml")


@pytest.fixture(scope="module")
def env():
    u = add_capital_factors(build_universe(CURVE, KEYS))
    per_dollar = per_dollar_cashflows(cashflow_matrix(cashflow_table(u, CURVE)), u.set_index("id").price)
    liab = project_liabilities(CURVE)
    lk = key_rate_dv01(liability_price_fn(liab), CURVE, KEYS)
    d = build_problem_data(u, per_dollar, liab.total.to_numpy(), lk, forward_rates(CURVE, 30), KEYS)
    return u, liab, d


def test_s_curve_is_centered_monotone_and_bounded():
    cfg = load_yaml("prepayment.yaml")
    f = lambda dy: rate_dependent_cpr(0.08, dy, cfg["a"], cfg["b"], cfg["cpr_min"], cfg["cpr_max"])
    assert f(0.0) == pytest.approx(0.08)
    assert f(-0.01) > f(0.0) > f(0.01)                         # rates down -> faster prepayment
    assert cfg["cpr_min"] <= f(0.10) and f(-0.10) <= cfg["cpr_max"]


def test_rate_dependent_prepayment_creates_negative_convexity():
    for model, sign in (("constant", 1), ("rate_dependent", -1)):
        a = load_assumptions(prepay_model=model)
        sec = next(a[a.id == "FNMA_30Y"].itertuples(index=False))
        pf = price_fn(sec, solve_coupon(sec, CURVE), CURVE)
        assert np.sign(effective_convexity(pf, CURVE)) == sign, model
    # base-case price is identical under both models (S-curve centered at zero rate change)
    prices = []
    for model in ("constant", "rate_dependent"):
        sec = next(load_assumptions(prepay_model=model).query("id == 'FNMA_30Y'").itertuples(index=False))
        prices.append(price_fn(sec, solve_coupon(sec, CURVE), CURVE)(CURVE))
    assert prices[0] == pytest.approx(prices[1], abs=1e-12)


def test_notch_and_migration_rules(env):
    u = env[0]
    assert notch_down("2.C", 1) == "3.A" and notch_down("5.C", 3) == "6" and notch_down("EXEMPT", 2) == "EXEMPT"
    sub = u.set_index("id", drop=False).loc[["IG_BBB-_30Y", "CLO_BBB"]]
    new, widening = migrate(sub, pd.Series({"IG_BBB-_30Y": "3.A", "CLO_BBB": "2.C"}))
    assert new.loc["IG_BBB-_30Y", "cap_factor"] == pytest.approx(0.03151)        # corporate bond 3.A
    assert new.loc["CLO_BBB", "cap_factor"] == pytest.approx(0.03280)            # CLO table 2.C, thick tranche
    assert np.all(widening > 0) and np.all(new.pd > sub.pd)


def test_downgrade_and_lapse_scenarios(env):
    u, liab, _ = env
    h = load_rule_based()["A_conservative_ig"]
    base = run_scenario(h, u, liab, CURVE, KEYS, {}, POLICY)
    down = run_scenario(h, u, liab, CURVE, KEYS, {"downgrade": {"bbb_family": "3.A"}}, POLICY)
    assert down["capital_charge"] > base["capital_charge"]
    assert down["hy_after"] == pytest.approx(h[[i for i in h.index if "BBB" in i]].sum())
    assert down["hy_limit_breached"]
    lapse = run_scenario(h, u, liab, CURVE, KEYS, {"lapse": {"mult": 1.5, "years": [1, 5]}}, POLICY)
    assert abs(lapse["dL"]) < 0.5                       # surrender at reserve value: PV nearly unchanged
    assert lapse["min_funding_y1_to_check"] < base["min_funding_y1_to_check"]   # but cash leaves earlier


def test_swap_overlay(env):
    _, _, d = env
    cfg = load_yaml("overlay.yaml")
    for t in cfg["tenors"]:
        assert swap_value_per_100(t, CURVE, CURVE) == pytest.approx(0.0, abs=1e-10)
    d2 = add_overlay(d, CURVE, cfg)
    r0, r1 = solve(d, POLICY, 1.2), solve(d2, POLICY, 1.2)
    assert r1["objective"] >= r0["objective"] - 1e-9                 # superset of choices
    x = r1["x"].to_numpy()
    swap = d2.extra["is_overlay"]
    assert x[~swap].sum() == pytest.approx(100.0, abs=1e-6)            # swaps use no budget
    assert x[swap].sum() <= cfg["max_total_notional"] + 1e-6
    assert d2.cap_factor[swap].sum() == 0.0
    for r in build_rows(d2, POLICY, 1.2):
        assert r.a @ x <= r.b + 1e-6 if r.sense == "<=" else r.a @ x == pytest.approx(r.b, abs=1e-6)
    # SwapOverlay valuation is consistent with the LP's KRD columns
    notional = {int(i.split("_")[-1][:-1]): v for i, v in r1["x"].items() if i.startswith("SWAP_") and v > 0}
    ov = SwapOverlay(notional, CURVE)
    assert ov.value(CURVE) == pytest.approx(0.0, abs=1e-9)
    assert ov.krdv01(CURVE, KEYS) * 1000 == pytest.approx(d2.krd[swap].T @ x[swap], abs=1e-6)


def test_tighter_capital_lowers_income(env):
    _, _, d = env
    assert solve(d, POLICY, 0.9)["objective"] < solve(d, POLICY, 1.2)["objective"]
