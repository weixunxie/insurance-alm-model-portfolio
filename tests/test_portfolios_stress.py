import numpy as np
import pandas as pd
import pytest

from alm.capital import add_capital_factors
from alm.cashflow_matching import forward_rates, per_dollar_cashflows
from alm.config import load_yaml
from alm.curve import load_curve, load_key_tenors
from alm.krd import key_rate_dv01
from alm.krd_exposure import krd_exposure_table, surplus_scenarios
from alm.liabilities import liability_price_fn, project_liabilities
from alm.optimizer import build_problem_data, is_hy, solve
from alm.portfolios import load_rule_based, policy_checks, portfolio_summary
from alm.robustness import perturbed_net_spread, run_robustness, summarize
from alm.stress import run_scenario
from alm.universe import build_universe, cashflow_matrix, cashflow_table

CURVE, KEYS = load_curve(), load_key_tenors()
POLICY = load_yaml("policy.yaml")


@pytest.fixture(scope="module")
def env():
    u = add_capital_factors(build_universe(CURVE, KEYS))
    per_dollar = per_dollar_cashflows(cashflow_matrix(cashflow_table(u, CURVE)), u.set_index("id").price)
    liab = project_liabilities(CURVE)
    lk = key_rate_dv01(liability_price_fn(liab), CURVE, KEYS)
    d = build_problem_data(u, per_dollar, liab.total.to_numpy(), lk, forward_rates(CURVE, 30), KEYS)
    return u, per_dollar, liab, d


def test_rule_based_portfolios_sum_to_100_and_respect_bucket_limits(env):
    u = env[0]
    for name, h in load_rule_based().items():
        assert h.sum() == pytest.approx(100.0), name
        assert set(h.index) <= set(u.id), name
        assert policy_checks(h, u, POLICY)["bucket_limits_ok"], name
    b = policy_checks(load_rule_based()["B_yield_seeking"], u, POLICY)
    assert b["hy"] == pytest.approx(10.0)                     # CLO_BB (3.B) counts toward HY
    assert b["structured_ex_agency"] == pytest.approx(30.0)


def test_summary_income_identities(env):
    u, per_dollar, liab, _ = env
    h = load_rule_based()["A_conservative_ig"]
    s = portfolio_summary(h, u, liab, CURVE, KEYS, per_dollar, POLICY)
    ui = u.set_index("id").loc[h.index]
    assert s["net_spread_income"] == pytest.approx(float(h @ (ui.spread - ui.el_annual)))
    assert s["expected_income"] == pytest.approx(float(h @ ui.ytm) - s["expected_credit_loss"])


def test_stress_engine_matches_module_b_and_isolates_spread_shocks(env):
    u, _, liab, _ = env
    h = load_rule_based()["B_yield_seeking"]
    lpf = liability_price_fn(liab)
    exp = krd_exposure_table(h, u, lpf, CURVE, KEYS)
    spec = {"type": "parallel", "bp": 100}
    ref = surplus_scenarios(h, u, lpf, CURVE, KEYS, {"up": spec}, exp).iloc[0]
    r = run_scenario(h, u, liab, CURVE, KEYS, {"curve": spec}, POLICY)
    assert r["dS"] == pytest.approx(ref.dS_full, abs=1e-10)

    hy = run_scenario(h, u, liab, CURVE, KEYS, {"spread_shocks": [{"select": {"naic": "hy"}, "bp": 200}]}, POLICY)
    assert hy["dL"] == 0.0
    ui = u.set_index("id").loc[h.index]
    naic_hy = is_hy(ui.naic_designation)                  # same rule as the HY limit: includes CLO_BB (3.B)
    assert "CLO_BB" in ui.index[naic_hy]
    # first-order check: dA ~ -sum MV * spread duration * 200bp over NAIC 3+ holdings only
    approx = -float((h[naic_hy] * ui.spread_dur[naic_hy]).sum()) * 0.02
    assert hy["dA"] == pytest.approx(approx, rel=0.1)
    # book-value proxy: a pure spread move does not change the capital charge
    assert hy["capital_charge"] == pytest.approx(float(h @ ui.cap_factor))
    assert hy["capital_charge_mtm"] < hy["capital_charge"]

    zero = run_scenario(h, u, liab, CURVE, KEYS, {}, POLICY)
    assert zero["dA"] == 0.0 and zero["dS"] == 0.0


def test_robustness_is_reproducible_and_feasible(env):
    u, _, _, d = env
    cfg = {"n_draws": 20, "seed": 42, "spread_sd": 0.10, "pd_sigma": 0.25}
    base = solve(d, POLICY, POLICY["capital"]["budget"])["x"]
    a = run_robustness(d, u, POLICY, cfg, base)
    b = run_robustness(d, u, POLICY, cfg, base)
    pd.testing.assert_frame_equal(a, b)
    assert (a.status == "optimal").all()
    assert np.allclose(a[d.ids].sum(axis=1), 100.0, atol=1e-6)
    assert (a.capital_used <= POLICY["capital"]["budget"] + 1e-6).all()
    # fixed C is feasible in every draw, so the re-optimized in-sample income can never be lower
    assert (a.net_income >= a.income_fixed_C - 1e-9).all()
    s = summarize(a, d, u, POLICY, base)
    assert s["curse"]["optimizers_curse_gap"] >= 0
    assert s["positions"].pct_draws_held.between(0, 1).all()
    assert np.isnan(s["positions"].loc["UST_10Y", "pct_draws_at_issuer_cap"])   # exempt issuer


def test_lognormal_pd_multiplier_has_mean_one():
    rng = np.random.default_rng(0)
    n = 200_000
    out = perturbed_net_spread(np.zeros(n), np.ones(n), np.zeros(n), rng, 0.10, 0.25)
    assert -out.mean() == pytest.approx(1.0, abs=0.01)                 # net spread = -PD when s = 0, R = 0
