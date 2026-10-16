import numpy as np
import pytest

from alm.curve import (flat_curve, key_rate_shape, key_rate_shocks, linear_twist,
                       load_curve, load_key_tenors, parallel_shift)


def test_base_curve_nodes_and_interpolation():
    c = load_curve()
    assert c.zero(10) == pytest.approx(0.0425)
    assert c.zero(4) == pytest.approx((0.0385 + 0.0390) / 2)      # linear between 3Y and 5Y
    assert c.zero(0.5) == pytest.approx(0.0400)                    # flat before 1Y
    assert c.zero(40) == pytest.approx(0.0455)                     # flat after 30Y


def test_discount_factor_and_forward():
    c = flat_curve(0.04)
    assert c.discount_factor(0.0) == pytest.approx(1.0)
    assert c.discount_factor(3) == pytest.approx(1.04 ** -3)
    assert c.discount_factor(3, spread=0.01) == pytest.approx(1.05 ** -3)
    assert c.forward_rate(2, 3) == pytest.approx(0.04)


def test_triangular_weights_sum_to_one():
    keys = load_key_tenors()
    t = np.linspace(0, 40, 401)
    total = sum(key_rate_shape(t, keys, k) for k in range(len(keys)))
    assert np.allclose(total, 1.0)
    # peak at its own tenor, zero at neighbours
    assert key_rate_shape(5.0, keys, 3) == pytest.approx(1.0)
    assert key_rate_shape(4.0, keys, 3) == pytest.approx(0.5)
    assert key_rate_shape(7.0, keys, 3) == pytest.approx(0.0)


def test_shifts():
    c = load_curve()
    keys = load_key_tenors()
    assert np.allclose(parallel_shift(c, 0.01).zero(keys) - c.zero(keys), 0.01)
    steep = linear_twist(c, -0.005, 0.005)
    assert steep.zero(1) - c.zero(1) == pytest.approx(-0.005)
    assert steep.zero(30) - c.zero(30) == pytest.approx(0.005)
    assert steep.zero(15.5) - c.zero(15.5) == pytest.approx(0.0)
    shocked = key_rate_shocks(c, keys, [0, -0.005, 0, 0, 0, 0.005, 0, 0])
    assert shocked.zero(5) == pytest.approx(c.zero(5))
    assert shocked.zero(2) - c.zero(2) == pytest.approx(-0.005)
