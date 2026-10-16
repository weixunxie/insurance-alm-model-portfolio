import numpy as np
import pytest

from alm.bonds import curve_modified_duration, macaulay_duration, ytm
from alm.curve import load_curve, load_key_tenors
from alm.liabilities import liability_price_fn, project_liabilities
from alm.universe import build_universe, cashflow_matrix, cashflow_table, schedule

CURVE, KEYS = load_curve(), load_key_tenors()


@pytest.fixture(scope="module")
def universe():
    return build_universe(CURVE, KEYS)


def test_universe_size_and_prices(universe):
    assert 25 <= len(universe) <= 35
    coupon_bonds = universe[universe.cf_model != "zero"]
    assert np.all(np.abs(coupon_bonds.price - 100) < 1.0)          # coupons solved near par
    assert np.allclose(universe.krd_1y + universe.krd_2y + universe.krd_3y + universe.krd_5y
                       + universe.krd_7y + universe.krd_10y + universe.krd_20y + universe.krd_30y,
                       universe.effective_dur, rtol=0.01)


def test_floater_rate_vs_spread_duration(universe):
    clo = universe[universe.cf_model == "floater"]
    assert np.all(clo.effective_dur < 1.0)                          # rate duration ~ time to reset
    for sec in clo.itertuples(index=False):
        # spread bump moves discounting only, so spread duration = modified duration of the
        # projected cash flows under z(t) + s
        s = schedule(sec, sec.coupon, CURVE, CURVE)
        assert sec.spread_dur == pytest.approx(
            curve_modified_duration(s["t"], s["total"], CURVE, sec.spread), abs=1e-6)
    # below WAL: coupons arrive before principal and every cash flow is discounted
    assert np.all(clo.spread_dur < clo.wal)


def test_fixed_bond_spread_duration_matches_effective(universe):
    fixed = universe[universe.cf_model.isin(["bullet", "zero"])]
    assert np.allclose(fixed.spread_dur, fixed.effective_dur, rtol=0.02)


def test_expected_cashflows_below_promised_in_total(universe):
    cf = cashflow_table(universe, CURVE)
    df = CURVE.discount_factor(cf.t.to_numpy(float))
    by_sec = cf.assign(pv_exp=cf.expected * df, pv_prom=cf.promised * df).groupby("id")
    assert np.all(by_sec.expected.sum() <= by_sec.promised.sum() + 1e-9)
    assert np.all(by_sec.pv_exp.sum() <= by_sec.pv_prom.sum() + 1e-9)
    m = cashflow_matrix(cf)
    assert m.shape == (30, len(universe))


def test_liabilities_pv_and_duration():
    liab = project_liabilities(CURVE)
    assert liability_price_fn(liab)(CURVE) == pytest.approx(90.0, abs=1e-9)
    t, cf = liab.t.to_numpy(float), liab.total.to_numpy()
    mac = macaulay_duration(t, cf, ytm(t, cf, 90.0))
    assert 9.0 <= mac <= 11.0
    assert np.allclose(liab.benefits + liab.surrenders + liab.expenses, liab.total)
