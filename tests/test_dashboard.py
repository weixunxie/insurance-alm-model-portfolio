"""Dashboard: app imports, /health, and every page renders from model outputs."""

import pytest

pytest.importorskip("dash")

from dashboard import app as app_module  # noqa: E402
from dashboard.data import m  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return app_module.server.test_client()


def test_server_exists():
    assert app_module.server is not None


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.data == b"ok"


def test_index_serves(client):
    assert client.get("/").status_code == 200


@pytest.mark.parametrize("path", [p for p, _, _ in app_module.PAGES])
def test_every_route_renders(path):
    out = app_module.route(path)
    assert len(out) == 1 + len(app_module.PAGES)
    assert sum(c == "nav-link active" for c in out[1:]) == 1


def test_overview_numbers_come_from_metrics():
    from dashboard.components import fmt
    from dashboard.pages import overview
    text = str(overview.layout())
    assert fmt(m("phase5.portfolios.C.net_spread_income")) in text
    assert fmt(m("phase5.portfolios.A.net_spread_income")) in text


# ---------- Optimizer page: live re-solve ----------

from dataclasses import replace  # noqa: E402
import time  # noqa: E402

from dashboard import data as D  # noqa: E402
from dashboard import solve as S  # noqa: E402
from dashboard.pages import optimizer  # noqa: E402


def test_base_controls_reproduce_metrics():
    res = S.solve_controls(S.base_controls())
    head = m("headline")
    assert res["feasible"]
    assert res["net_income"] == pytest.approx(head["income_C"], abs=1e-6)
    assert res["capital_used"] == pytest.approx(head["capital_used_C"], abs=1e-6)
    assert res["shadow_price"] == pytest.approx(head["shadow_price_capital_budget"], abs=1e-6)


CONTROL_CHANGES = [
    {"budget_k": 0.40}, {"budget_k": 3.00}, {"scenario_on": False}, {"buffer_on": False},
    {"overlay_on": True}, {"max_loss_down": 0.0}, {"regime": "old_bond"},
    {"budget_k": 0.40, "overlay_on": True, "max_loss_down": 0.0, "regime": "old_bond"},
]


@pytest.mark.parametrize("change", CONTROL_CHANGES)
def test_each_control_solves_or_diagnoses(change):
    c = replace(S.base_controls(), **change)
    res = S.solve_controls(c)
    if res["feasible"]:
        assert res["status"] == "optimal"
        assert res["allocation"].sum() == pytest.approx(100.0, abs=1e-6)
        assert res["capital_used"] <= c.budget_k + 1e-6
    else:
        assert res["violations"] is not None and len(res["violations"]) > 0
    optimizer.results(c)          # renders without raising


def test_infeasible_controls_are_diagnosed():
    """A required surplus GAIN of $2M at -100bp is infeasible; the soft solve names the conflicting row."""
    c = replace(S.base_controls(), max_loss_down=-2.0)
    res = S.solve_controls(c)
    assert not res["feasible"] and res["hard_status"] == "infeasible"
    assert "scenario_parallel_down_100" in set(res["violations"].constraint)
    assert "No portfolio satisfies" in str(optimizer.results(c))


def test_constraint_toggles_change_the_rows():
    on = S.solve_controls(S.base_controls())
    off = S.solve_controls(replace(S.base_controls(), scenario_on=False, buffer_on=False))
    assert off["net_income"] >= on["net_income"] - 1e-9      # removing constraints cannot lower income
    assert "downgrade_buffer" in set(on["binding"].constraint)
    assert "downgrade_buffer" not in set(off["binding"].constraint)


def test_interaction_under_two_seconds_after_warmup():
    S.warm()
    for cached in (S.problem_data, S._solve_cached, S.frontier):     # worst case: nothing cached for this combination
        cached.cache_clear()
    c = replace(S.base_controls(), budget_k=1.85, overlay_on=True, regime="old_bond")
    t0 = time.perf_counter()
    optimizer.update(c.budget_k, ["scenario", "buffer", "overlay"], c.max_loss_down, c.regime)
    assert time.perf_counter() - t0 < 2.0


def test_reset_restores_base_case():
    c = S.base_controls()
    assert optimizer.reset(1) == (c.budget_k, ["scenario", "buffer"], c.max_loss_down, c.regime)


def test_maxloss_slider_disabled_with_hint_when_scenarios_off():
    disabled, hint = optimizer.toggle_maxloss(["buffer"])
    assert disabled and "scenario surplus constraints" in hint
    assert optimizer.toggle_maxloss(["scenario", "buffer"]) == (False, "")


def test_frontier_plotted_against_budget_k():
    c = S.base_controls()
    fig = optimizer.frontier_figure(S.frontier_for(c), S.solve_controls(c))
    assert list(fig.data[0].x) == list(S.frontier_for(c).budget_k)
    assert fig.data[1].x[0] == pytest.approx(c.budget_k)


def test_binding_capital_note_matches_metrics_flags():
    res = S.solve_controls(S.base_controls())
    flags = m("report.binding_capital")
    assert flags["downgrade_buffer_binding"] == ("downgrade_buffer" in set(res["binding"].constraint))
    assert flags["capital_row_binding"] == ("capital" in set(res["binding"].constraint))
    note = optimizer.binding_capital_note(res, D.base().policy)
    assert (note is not None) == (flags["downgrade_buffer_binding"] and not flags["capital_row_binding"])


# ---------- Rate risk, Cash flows, Capital ----------

from dashboard.pages import capital, cash_flows, rate_risk  # noqa: E402


@pytest.mark.parametrize("p", ["A", "C"])
@pytest.mark.parametrize("key,saved", [("exposure", "krd_exposure"), ("scen", "surplus_scenarios"),
                                       ("gaps", "cashflow_gaps")])
def test_live_portfolio_analysis_matches_saved_tables(p, key, saved):
    live = D.analysis(p)[key].reset_index(drop=True)
    ref = D.table(f"{saved}_portfolio_{p.lower()}")
    num = ref.select_dtypes("number").columns
    assert (live[num] - ref[num]).abs().to_numpy().max() < 1e-9


@pytest.mark.parametrize("p", ["A", "B", "C"])
def test_live_analysis_matches_metrics(p):
    a = D.analysis(p)
    assert a["dgap"]["surplus_dv01"] * 1000 == pytest.approx(m(f"phase5.portfolios.{p}.surplus_dv01_k"), abs=1e-9)
    assert a["cap"]["capital_charge"] == pytest.approx(m(f"phase5.portfolios.{p}.capital_charge"), abs=1e-9)
    dS = a["scen"].set_index("scenario").dS_full
    assert dS["parallel_down_100"] == pytest.approx(m(f"phase5.stress.{p}_parallel_down_100_dS"), abs=1e-9)


@pytest.mark.parametrize("p", ["A", "B", "C"])
def test_portfolio_pages_render(p):
    rate_risk.render_exposure(p)
    for name in rate_risk.SCENARIO_NAMES:
        rate_risk.render_scenario(p, name)
    cash_flows.render(p)


def test_capital_page_and_factor_provenance():
    text = str(capital.layout())
    assert "not yet checked" in text                       # bond factor provenance kept verbatim
    assert "checked against the adopted Proposal 2026-12-IRE LR002 form" in text


def test_efficiency_table_sorts():
    def first_col(table, col):
        body = table.children.children[1].children
        return [row.children[col].children for row in body]
    ids_desc = first_col(capital.efficiency_table("net_spread_bp", True), 0)
    rows = {r["id"]: r for r in capital.efficiency_rows()}
    vals = [rows[i]["net_spread_bp"] for i in ids_desc]
    assert vals == sorted(vals, reverse=True)
    eff = first_col(capital.efficiency_table("efficiency", False), 5)
    assert eff[-1] == "exempt"                             # exempt securities sort last


# ---------- Phase 4: hy_after toggle, Stress, Robustness, Methodology ----------

from dashboard.pages import methodology, robustness, stress  # noqa: E402


def test_hy_after_toggle_matches_metrics_cost():
    c = replace(S.base_controls(), hy_after_on=True)
    res = S.solve_controls(c)
    c6 = m("phase6.constraints")
    assert res["feasible"]
    assert res["net_income"] == pytest.approx(c6["income_with_hy_after_downgrade_limit"], abs=1e-6)
    assert S.hy_after_cost(c) == pytest.approx(c6["hy_after_downgrade_limit_cost"], abs=1e-6)
    assert "costs $" in str(optimizer.results(c))
    assert S.hy_after_cost(S.base_controls()) is None          # default off, matching policy.yaml


def test_hy_after_default_matches_policy():
    hl = D.base().policy["downgrade_buffer"]["hy_limit_after_downgrade"]
    assert S.base_controls().hy_after_on == hl["enabled"]


def test_stress_heatmap_covers_every_scenario_and_is_centered():
    fig = stress.heatmap_figure()
    hm = fig.data[0]
    assert len(hm.y) == len(set(D.table("stress_all").scenario))
    assert list(hm.x) == [c for c in (stress.PORTFOLIO_NAMES[p] for p in stress.PORTFOLIOS)]
    assert hm.zmid == 0 and hm.zmin == -hm.zmax
    assert len(fig.layout.annotations) == hm.z.size                # value printed in every cell
    colors = [c for _, c in stress.DIVERGING]
    assert not any(c.lower().startswith("#2e7d") for c in colors)  # red-blue, not red-green


@pytest.mark.parametrize("name", list(stress.LABELS))
def test_stress_detail_renders_with_pm_note(name):
    out = str(stress.render_detail(name))
    assert stress.pm_notes()[name][:40] in out


def test_robustness_and_methodology_render():
    text = str(robustness.layout())
    assert fmt_metric("phase5.robustness.income_fixed_C_mean") in text
    md = str(methodology.layout())
    assert "Assumptions register" in md and "GENERATED FILE" not in md


def fmt_metric(path, spec=".3f"):
    from dashboard.components import fmt
    return fmt(m(path), spec)


def test_income_only_scenarios_labelled_and_captioned():
    only = stress.income_only_scenarios()
    assert "hy_defaults_x2" in only
    s = D.table("stress_all")
    s = s[s.portfolio.isin(["A", "B", "C"]) & s.scenario.isin(only)]
    assert (s.dA.abs().max(), s.dL.abs().max()) == (0.0, 0.0) and s.d_net_income.abs().max() > 0
    rows = list(stress.heatmap_figure().data[0].y)
    assert sum(r.endswith("(income only)") for r in rows) == len(only)
    cap = stress.income_only_caption()
    c = s[(s.scenario == "hy_defaults_x2") & (s.portfolio == "C")].d_net_income.iloc[0]
    assert stress.fmt(c * 1000, "+.1f") in cap and "zero by construction" in cap
