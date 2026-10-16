import numpy as np
import pytest

from alm.bonds import BP, effective_duration, fixed_cashflows, fixed_price_fn, zero_cashflows
from alm.curve import flat_curve, key_rate_shocks, load_curve, load_key_tenors, parallel_shift
from alm.krd import key_rate_durations, key_rate_dv01, krd_pnl

KEYS = load_key_tenors()                       # 1, 2, 3, 5, 7, 10, 20, 30
FLAT = flat_curve(0.04)


def _zero_position(pv, maturity, curve):
    """Zero-coupon position with present value `pv` on `curve`."""
    t, cf = zero_cashflows(1.0, maturity)
    return fixed_price_fn(t, cf * pv / curve.discount_factor(maturity))


def _barbell_and_liability():
    a2 = _zero_position(62.5, 2, FLAT)
    a10 = _zero_position(37.5, 10, FLAT)
    assets = lambda c: a2(c) + a10(c)
    liability = _zero_position(100.0, 5, FLAT)
    return assets, liability


def test_barbell_vs_bullet_krd_gap():
    assets, liability = _barbell_and_liability()
    assert assets(FLAT) == pytest.approx(100.0)
    assert liability(FLAT) == pytest.approx(100.0)

    krd_a = key_rate_durations(assets, FLAT, KEYS)
    krd_l = key_rate_durations(liability, FLAT, KEYS)
    assert krd_a.sum() == pytest.approx(4.808, abs=1e-3)
    assert krd_l.sum() == pytest.approx(4.808, abs=1e-3)

    gap = dict(zip(KEYS, krd_a - krd_l))      # both PVs are 100, so durations are comparable
    assert gap[2] == pytest.approx(1.202, abs=1e-3)
    assert gap[5] == pytest.approx(-4.808, abs=1e-3)
    assert gap[10] == pytest.approx(3.606, abs=1e-3)
    assert all(abs(gap[k]) < 1e-9 for k in KEYS if k not in (2, 5, 10))


def test_barbell_surplus_under_parallel_and_steepener():
    assets, liability = _barbell_and_liability()
    net = key_rate_dv01(assets, FLAT, KEYS) - key_rate_dv01(liability, FLAT, KEYS)

    # Parallel: KRD approximation is exactly flat; full repricing differs only by convexity
    assert krd_pnl(net, np.full(len(KEYS), 100.0)) == pytest.approx(0.0, abs=1e-6)
    for dz in (0.01, -0.01):
        shocked = parallel_shift(FLAT, dz)
        ds = (assets(shocked) - assets(FLAT)) - (liability(shocked) - liability(FLAT))
        assert abs(ds) < 0.1

    # Steepener: 2Y -50bp, 10Y +50bp, 5Y unchanged (triangular key rate shocks)
    shocks_bp = np.where(KEYS == 2, -50.0, np.where(KEYS == 10, 50.0, 0.0))
    assert krd_pnl(net, shocks_bp) == pytest.approx(-1.202, abs=1e-3)
    shocked = key_rate_shocks(FLAT, KEYS, shocks_bp * BP)
    ds_full = (assets(shocked) - 100.0) - (liability(shocked) - 100.0)
    assert ds_full == pytest.approx(-1.20, abs=0.1)   # second-order (convexity) difference


@pytest.mark.parametrize("coupon,maturity,spread", [(0.05, 3, 0.0), (0.045, 10, 0.012), (0.05, 30, 0.02)])
def test_sum_of_krds_equals_effective_duration(coupon, maturity, spread):
    curve = load_curve()
    t, cf = fixed_cashflows(100, coupon, maturity)
    pf = fixed_price_fn(t, cf, spread)
    krd = key_rate_durations(pf, curve, KEYS)
    assert krd.sum() == pytest.approx(effective_duration(pf, curve), rel=0.01)
    assert np.all(krd >= -1e-12)
