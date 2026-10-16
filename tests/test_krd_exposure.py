import numpy as np
import pandas as pd
import pytest

from alm.bonds import dv01_fd
from alm.config import load_yaml
from alm.curve import linear_twist, load_curve, load_key_tenors
from alm.krd_exposure import (duration_gap, krd_exposure_table, portfolio_value,
                              scenario_key_shocks_bp, surplus_scenarios)
from alm.liabilities import liability_price_fn, project_liabilities
from alm.universe import build_universe

CURVE, KEYS = load_curve(), load_key_tenors()


@pytest.fixture(scope="module")
def setup():
    u = build_universe(CURVE, KEYS)
    h = pd.Series(load_yaml("portfolios.yaml")["A_conservative_ig"]["holdings"], dtype=float)
    lpf = liability_price_fn(project_liabilities(CURVE))
    return u, h, lpf, krd_exposure_table(h, u, lpf, CURVE, KEYS)


def test_portfolio_reprices_to_market_value(setup):
    u, h, _, _ = setup
    assert h.sum() == pytest.approx(100.0)
    assert portfolio_value(h, u, CURVE, CURVE) == pytest.approx(100.0, abs=1e-9)


def test_summed_key_rate_dv01_equals_parallel_dv01(setup):
    u, h, lpf, exp = setup
    asset_dv01 = dv01_fd(lambda c: portfolio_value(h, u, c, CURVE), CURVE)
    assert exp.asset_krdv01.sum() == pytest.approx(asset_dv01, rel=0.01)
    assert exp.liability_krdv01.sum() == pytest.approx(dv01_fd(lpf, CURVE), rel=0.01)


def test_twist_decomposes_exactly_into_key_rate_shocks():
    shocked = linear_twist(CURVE, -0.005, 0.005)
    shocks = scenario_key_shocks_bp(CURVE, shocked, KEYS)
    assert shocks[0] == pytest.approx(-50) and shocks[-1] == pytest.approx(50)
    assert np.allclose(shocks, -50 + 100 * (KEYS - 1) / 29)


def test_krd_approximation_converges_for_small_shocks(setup):
    u, h, lpf, exp = setup
    small = {"tiny_twist": {"type": "twist", "short_bp": -1, "long_bp": 1, "t_short": 1, "t_long": 30},
             "tiny_parallel": {"type": "parallel", "bp": 1}}
    res = surplus_scenarios(h, u, lpf, CURVE, KEYS, small, exp)
    assert np.allclose(res.dS_krd, res.dS_full, rtol=0.01)       # remaining gap is second order (convexity)


def test_duration_gap_toy():
    exp = pd.DataFrame({"asset_krdv01": [0.05], "liability_krdv01": [0.045]})
    g = duration_gap(exp, assets=100.0, liabilities=90.0)
    assert g["asset_duration"] == pytest.approx(5.0)
    assert g["liability_duration"] == pytest.approx(5.0)
    assert g["duration_gap"] == pytest.approx(5.0 - 0.9 * 5.0)
    assert g["surplus_dv01"] == pytest.approx(0.005)
