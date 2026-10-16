import numpy as np
import pandas as pd
import pytest

from alm.cashflow_matching import (first_shortfall_year, forward_rates, funding_balance,
                                   funding_matrix, per_dollar_cashflows)
from alm.curve import load_curve, load_key_tenors
from alm.universe import build_universe, cashflow_matrix, cashflow_table

CURVE = load_curve()


def test_toy_example_section_6():
    assets, liabilities = np.array([6.0, 3.0]), np.array([4.0, 5.0])
    gaps = assets - liabilities
    assert np.allclose(gaps, [2.0, -2.0])
    assert np.allclose(funding_balance(gaps), [2.0, 0.0])            # before reinvestment interest
    # with reinvestment at 4%: 2 grows to 2.08, minus 2 = 0.08
    assert np.allclose(funding_balance(gaps, fwd=[0.04, 0.04]), [2.0, 0.08])


def test_borrowing_uses_forward_plus_spread():
    bal = funding_balance([-1.0, 0.0], fwd=[0.04, 0.04], borrow_spread=0.01)
    assert bal[1] == pytest.approx(-1.05)
    assert first_shortfall_year(bal) == 1
    assert first_shortfall_year([1.0, 2.0]) is None


def test_funding_matrix_matches_recursion_when_balance_stays_positive():
    fwd = forward_rates(CURVE, 10)
    gaps = np.array([3.0, -1.0, 2.0, -2.5, 1.0, 0.5, -1.0, 2.0, -0.5, 1.0])
    rec = funding_balance(gaps, fwd)
    assert np.all(rec >= 0)
    assert np.allclose(funding_matrix(fwd) @ gaps, rec)


def test_per_dollar_cashflows_reprice_to_one_dollar():
    keys = load_key_tenors()
    u = build_universe(CURVE, keys)
    cf = cashflow_table(u, CURVE)
    promised = per_dollar_cashflows(cashflow_matrix(cf, "promised"), u.set_index("id").price)
    expected = per_dollar_cashflows(cashflow_matrix(cf, "expected"), u.set_index("id").price)
    t = np.arange(1, 31, dtype=float)
    for sec in u.itertuples(index=False):
        df = CURVE.discount_factor(t, sec.spread)
        assert promised[sec.id] @ df == pytest.approx(1.0, abs=1e-9)   # $1 of MV
        assert expected[sec.id] @ df <= 1.0 + 1e-9
