import numpy as np
import pytest

from alm.config import load_yaml
from alm.structured import (amortizing_schedule, bullet_schedule, expected_cashflows,
                            sequential_waterfall, wal)


def test_level_pay_amortization_without_prepayment():
    s = amortizing_schedule(100, 0.05, amort_term=10, maturity=10, cpr=0.0)
    principal = s["scheduled_principal"] + s["prepayment"]
    assert principal.sum() == pytest.approx(100)
    assert np.allclose(s["total"], s["total"][0])              # level payment
    assert s["total"][0] == pytest.approx(100 * 0.05 / (1 - 1.05 ** -10))


def test_prepayment_shortens_wal_and_balloon_pays_off():
    slow = amortizing_schedule(100, 0.05, 30, 30, cpr=0.0)
    fast = amortizing_schedule(100, 0.05, 30, 30, cpr=0.10)
    assert wal(fast) < wal(slow)
    assert (fast["scheduled_principal"] + fast["prepayment"]).sum() == pytest.approx(100)
    balloon = amortizing_schedule(100, 0.05, amort_term=30, maturity=10)
    assert balloon["scheduled_principal"][-1] > 80                # most principal at the balloon
    assert (balloon["scheduled_principal"]).sum() == pytest.approx(100)


def test_expected_cashflows():
    s = bullet_schedule(100, 0.05, 5)
    riskless = expected_cashflows(s, pd=0.0, recovery=0.4)
    assert np.allclose(riskless["expected"], s["total"])
    risky = expected_cashflows(s, pd=0.02, recovery=0.4)
    # a default brings recovery forward, so a single year can exceed promised; totals cannot
    assert risky["expected"].sum() < s["total"].sum()
    # year 1: survivors get the 5 coupon, defaulters recover 40% of 100 face
    assert risky["expected"][0] == pytest.approx(0.98 * 5 + 0.02 * 0.4 * 100)


def test_waterfall_subordination():
    w = load_yaml("waterfall_demo.yaml")
    args = (w["collateral"], w["collateral_coupon"])
    clean = sequential_waterfall(*args, 0.0, w["recovery"], w["repay_rate"], w["maturity"], w["tranches"])
    assert np.allclose(clean["loss_pct"], 0.0)
    for cdr in (0.05, 0.10, 0.20):
        r = sequential_waterfall(*args, cdr, w["recovery"], w["repay_rate"], w["maturity"], w["tranches"])
        assert r["loss_pct"][0] <= r["loss_pct"][1] <= r["loss_pct"][2]   # junior loses first
        assert r["principal_paid"].sum() + r["written_down"].sum() == pytest.approx(w["collateral"])
