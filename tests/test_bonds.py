"""Textbook checks (tolerance 1e-3)."""

import pytest

from alm.bonds import (annuity_cashflows, convexity, dv01, dv01_fd, effective_convexity,
                       effective_duration, fixed_cashflows, fixed_price_fn, macaulay_duration,
                       modified_duration, price_at_yield, pv_on_curve, ytm, zero_cashflows)
from alm.curve import flat_curve

TOL = 1e-3
FLAT = flat_curve(0.04)
BOND_3Y = fixed_cashflows(face=100, coupon=0.05, maturity=3)


def test_3y_bond_price_and_durations():
    t, cf = BOND_3Y
    assert pv_on_curve(t, cf, FLAT) == pytest.approx(102.775, abs=TOL)
    assert macaulay_duration(t, cf, 0.04) == pytest.approx(2.861, abs=TOL)
    assert modified_duration(t, cf, 0.04) == pytest.approx(2.751, abs=TOL)
    assert convexity(t, cf, 0.04) == pytest.approx(10.41, abs=0.005)
    assert dv01(t, cf, 0.04) == pytest.approx(0.02828, abs=1e-5)


def test_analytic_dv01_equals_central_difference():
    t, cf = BOND_3Y
    assert dv01_fd(fixed_price_fn(t, cf), FLAT) == pytest.approx(dv01(t, cf, 0.04), rel=1e-6)


def test_effective_measures_match_analytic_on_flat_curve():
    t, cf = BOND_3Y
    pf = fixed_price_fn(t, cf)
    assert effective_duration(pf, FLAT) == pytest.approx(modified_duration(t, cf, 0.04), abs=TOL)
    assert effective_convexity(pf, FLAT) == pytest.approx(convexity(t, cf, 0.04), abs=0.01)


def test_par_bond():
    t, cf = BOND_3Y
    assert price_at_yield(t, cf, 0.05) == pytest.approx(100.000, abs=TOL)


def test_annuity():
    t, cf = annuity_cashflows(payment=10, n_years=5)
    assert price_at_yield(t, cf, 0.04) == pytest.approx(44.518, abs=TOL)
    assert macaulay_duration(t, cf, 0.04) == pytest.approx(2.922, abs=TOL)
    assert modified_duration(t, cf, 0.04) == pytest.approx(2.809, abs=TOL)


def test_ytm_recovers_yield():
    t, cf = BOND_3Y
    assert ytm(t, cf, 102.775) == pytest.approx(0.04, abs=1e-5)
    assert ytm(t, cf, price_at_yield(t, cf, 0.0437)) == pytest.approx(0.0437, abs=1e-10)


@pytest.mark.parametrize("maturity", [2, 10, 30])
def test_zero_coupon_modified_duration(maturity):
    t, cf = zero_cashflows(100, maturity)
    assert modified_duration(t, cf, 0.04) == pytest.approx(maturity / 1.04, abs=1e-12)
