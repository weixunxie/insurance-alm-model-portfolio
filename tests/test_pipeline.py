"""alm.pipeline: the shared model assembly reproduces the base-case LP in outputs/metrics.json."""

import json

import cvxpy as cp
import pytest

from alm.config import OUTPUT_DIR
from alm.optimizer import budget_shadow_price, solve_or_diagnose
from alm.pipeline import build_base, build_data


@pytest.fixture(scope="module")
def base():
    return build_base()


def test_build_base_shapes(base):
    assert len(base.universe) == base.per_dollar.shape[1]
    assert len(base.liab) == base.per_dollar.shape[0] == 30
    assert "cap_factor" in base.universe


def test_build_data_reproduces_base_case(base):
    head = json.loads((OUTPUT_DIR / "metrics.json").read_text())["headline"]
    d = build_data(base.universe, base.per_dollar, base.liab, base.curve, base.keys, base.policy)
    res = solve_or_diagnose(d, base.policy, base.policy["capital"]["budget"])
    assert res["status"] == cp.OPTIMAL
    assert res["objective"] == pytest.approx(head["income_C"], abs=1e-6)
    assert float(d.cap_factor @ res["x"].to_numpy()) == pytest.approx(head["capital_used_C"], abs=1e-6)
    assert budget_shadow_price(d, res) == pytest.approx(head["shadow_price_capital_budget"], abs=1e-6)
