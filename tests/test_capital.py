import numpy as np
import pandas as pd
import pytest

from alm.capital import (add_capital_factors, capital_charge, capital_factor, capital_report,
                         clo_regime_impact)
from alm.curve import load_curve, load_key_tenors
from alm.universe import build_universe


@pytest.fixture(scope="module")
def universe():
    return build_universe(load_curve(), load_key_tenors())


def test_spec_capital_examples(universe):
    u = add_capital_factors(universe).set_index("id")
    assert capital_charge(10.0, u.loc["IG_BBB_10Y", "cap_factor"]) == pytest.approx(0.1523, abs=1e-12)
    assert capital_charge(10.0, u.loc["UST_10Y", "cap_factor"]) == 0.0
    assert u.loc["GNMA_30Y", "cap_factor"] == 0.0
    assert u.loc["FNMA_30Y", "cap_factor"] == pytest.approx(0.00158)


def test_clo_uses_new_2026_factors_not_bond_factors(universe):
    new = add_capital_factors(universe, "new_2026").set_index("id")
    old = add_capital_factors(universe, "old_bond").set_index("id")
    assert new.loc["CLO_AAA", "cap_factor"] == pytest.approx(0.00040)
    assert old.loc["CLO_AAA", "cap_factor"] == pytest.approx(0.00158)
    assert new.loc["CLO_BB", "cap_factor"] == pytest.approx(0.25150)          # 3.B, thickness 5% > 4%: no surcharge
    assert new.loc["IG_BBB_10Y", "cap_factor"] == old.loc["IG_BBB_10Y", "cap_factor"]


def test_thin_tranche_surcharge_rule():
    """No universe tranche is <= 4% thick, so test the rule on synthetic tranches."""
    row = lambda desig, thick, coll: pd.Series({"bucket": "CLO", "naic_designation": desig,
                                                "tranche_thickness": thick, "clo_collateral": coll})
    assert capital_factor(row("3.B", 0.03, "BSL")) == pytest.approx(0.25150 + 0.1177)     # thin BSL: add-on
    assert capital_factor(row("2.C", 0.04, "BSL")) == pytest.approx(0.03280 + 0.1177)     # boundary: <= 4%, 2.C
    assert capital_factor(row("3.B", 0.03, "MM")) == pytest.approx(0.25150)               # middle market: none
    assert capital_factor(row("3.B", 0.05, "BSL")) == pytest.approx(0.25150)              # thick: none
    assert capital_factor(row("2.B", 0.03, "BSL")) == pytest.approx(0.03240)              # above 2.C: none


def test_capital_report_and_regime_impact(universe):
    u = add_capital_factors(universe)
    h = pd.Series({"IG_BBB_10Y": 10.0, "UST_10Y": 80.0, "CLO_AAA": 10.0})
    rows, summary = capital_report(h, u, budget=1.2)
    assert summary["capital_charge"] == pytest.approx(0.1523 + 10 * 0.00040)
    assert summary["utilization"] == pytest.approx(summary["capital_charge"] / 1.2)
    assert rows.contribution.sum() == pytest.approx(1.0)
    imp = clo_regime_impact(h, universe).set_index("regime")
    assert imp.loc["old_bond", "clo_charge"] == pytest.approx(10 * 0.00158)
    assert imp.loc["new_2026", "clo_charge"] == pytest.approx(10 * 0.00040)
