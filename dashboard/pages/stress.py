"""Stress tests: surplus heatmap across scenarios and portfolios, scenario detail with the PM note."""

import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from alm.config import load_yaml
from dashboard.components import (PANEL, PORTFOLIO_NAMES, PORTFOLIOS, TEXT, chart, fmt, kpi, page_header,
                                  report_table, section)
from dashboard.data import m, table

LABELS = {
    "parallel_up_100": "Parallel +100bp", "parallel_down_100": "Parallel −100bp",
    "steepener": "Steepener", "flattener": "Flattener", "hy_spread_up_200": "HY (NAIC 3+) spreads +200bp",
    "ig_spread_up_50": "IG spreads +50bp", "structured_spread_up": "Structured spreads up (CLO +150bp)",
    "bbb_downgrade_one_notch": "BBB family down one notch", "fallen_angels_bbb_to_3A": "Fallen angels (BBB to 3.A)",
    "hy_defaults_x2": "HY defaults ×2, recovery −10pts", "prepay_fast_x1_5": "Faster prepayments (CPR ×1.5)",
    "prepay_slow_x0_5": "Slower prepayments (CPR ×0.5)", "surrenders_up_50pct_y1_5": "Surrenders +50%, years 1–5",
    "combined_adverse": "Combined adverse",
}
# Diverging red-blue (colorblind-safe pair) with a neutral midpoint at 0.
DIVERGING = [[0.0, "#B03A2E"], [0.25, "#DE9A8F"], [0.5, "#F4F2EE"], [0.75, "#93B5D0"], [1.0, "#1F5F8B"]]


def pm_notes() -> dict:
    cfg = load_yaml("scenarios.yaml")
    notes = {}
    for group in ("stress_scenarios", "stress_scenarios_phase6"):
        notes.update({k: v["pm_note"] for k, v in cfg[group].items() if "pm_note" in v})
    return notes


def stress_frame():
    s = table("stress_all")
    return s[s.portfolio.isin(PORTFOLIOS)]


def income_only_scenarios() -> list[str]:
    """Scenarios that move no market value (dA = dL = 0 in every portfolio) but do change income."""
    g = stress_frame().groupby("scenario")
    moves_mv = (g.dA.apply(lambda x: x.abs().max()) > 1e-12) | (g.dL.apply(lambda x: x.abs().max()) > 1e-12)
    moves_income = g.d_net_income.apply(lambda x: x.abs().max()) > 1e-12
    return [k for k in LABELS if k in moves_mv.index and not moves_mv[k] and moves_income[k]]


def row_label(name: str, income_only: list[str]) -> str:
    return LABELS[name] + (" (income only)" if name in income_only else "")


def income_only_caption() -> str:
    """Caption under the heatmap: why income-only rows show zero, with their income change from stress_all."""
    s = stress_frame().set_index(["scenario", "portfolio"])
    parts = []
    for k in income_only_scenarios():
        vals = ", ".join(f"{p} {fmt(s.loc[(k, p), 'd_net_income'] * 1000, '+.1f')}k" for p in PORTFOLIOS)
        parts.append(f"{LABELS[k]}: \u0394 net income {vals} per year.")
    if not parts:
        return ""
    return ("Default-rate stress changes expected income and cash flows, not spreads, so \u0394 surplus is zero by "
            "construction; see \u0394 net income in the detail table. " + " ".join(parts))


def heatmap_figure() -> go.Figure:
    s = stress_frame()
    order = [k for k in LABELS if k in set(s.scenario)]
    z = s.pivot(index="scenario", columns="portfolio", values="dS").reindex(order)[PORTFOLIOS]
    vmax = float(z.abs().to_numpy().max())
    income_only = income_only_scenarios()
    rows, cols = [row_label(k, income_only) for k in order], [PORTFOLIO_NAMES[p] for p in PORTFOLIOS]
    fig = go.Figure(go.Heatmap(z=z.to_numpy(), x=cols, y=rows, colorscale=DIVERGING, zmid=0, zmin=-vmax, zmax=vmax,
                               xgap=2, ygap=2, colorbar=dict(title=dict(text="ΔS, $M"), thickness=10, outlinewidth=0),
                               hovertemplate="%{y}<br>%{x}: Δ surplus %{z:+.3f} $M<extra></extra>"))
    for i, k in enumerate(order):
        for j, p in enumerate(PORTFOLIOS):
            v = float(z.loc[k, p])
            fig.add_annotation(x=cols[j], y=rows[i], text=fmt(v, "+.2f"), showarrow=False,
                               font=dict(size=12, color=PANEL if abs(v) > 0.55 * vmax else TEXT))
    fig.update_layout(yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(side="top", showline=False),
                      plot_bgcolor=PANEL)
    return fig


def layout():
    return html.Div([
        page_header("Stress tests",
                    "Each scenario reprices assets and liabilities in full and re-runs the cash flow, KRD and capital "
                    "modules. Surplus change ΔS = ΔA − ΔL. Liability cash flows are fixed under rate "
                    "shocks (no dynamic lapse)."),
        section("Surplus change by scenario", chart(
            "Δ economic surplus by scenario and portfolio, $M", heatmap_figure(), height=560,
            caption=f"Red: surplus loss. Blue: surplus gain. Saved results from scripts/run_all.py. "
                    f"{income_only_caption()}")),
        section("Scenario detail",
                dcc.Dropdown(id="st-scenario", value="combined_adverse", clearable=False, className="dropdown",
                             options=[{"label": v, "value": k} for k, v in LABELS.items()]),
                html.Div(id="st-detail")),
        section("Tighter capital budget (K − 25%), re-optimized", tight_capital_block()),
    ])


@callback(Output("st-detail", "children"), Input("st-scenario", "value"))
def render_detail(name):
    s = stress_frame().set_index(["scenario", "portfolio"])
    hy_limit, check = m("inputs.limits.hy_max"), m("inputs.funding.check_years")
    spec = [("Δ assets, $M", "dA", "+.3f"), ("Δ liabilities, $M", "dL", "+.3f"),
            ("Δ surplus, $M", "dS", "+.3f"), ("Δ net spread income, $M / yr", "d_net_income", "+.4f"),
            ("Capital charge (book-value proxy), $M", "capital_charge", ".3f"),
            ("Capital charge (mark-to-market), $M", "capital_charge_mtm", ".3f"),
            (f"Min funding balance, years 1–{check}, $M", "min_funding_y1_to_check", "+.2f"),
            ("Max cumulative shortfall, $M", "max_cum_shortfall", ".2f"),
            ("Surplus DV01 after shock, $k / bp", "surplus_dv01_k", "+.2f"),
            (f"NAIC 3+ holdings after, $M (limit {hy_limit:.0f})", "hy_after", ".1f")]
    rows, breaches = [], set()
    for i, (label, col, f) in enumerate(spec):
        row = [label]
        for j, p in enumerate(PORTFOLIOS, start=1):
            v = s.loc[(name, p), col]
            row.append(fmt(float(v), f))
            if col == "hy_after" and bool(s.loc[(name, p), "hy_limit_breached"]):
                breaches.add((i, j))
            if col == "min_funding_y1_to_check" and float(v) < 0:
                breaches.add((i, j))
        rows.append(row)
    note = pm_notes().get(name, "")
    return html.Div(className="grid-2 section-tight", children=[
        html.Div(className="panel", children=[
            report_table(["", *[PORTFOLIO_NAMES[p] for p in PORTFOLIOS]], rows, numeric_cols={1, 2, 3},
                         breaches=breaches),
            html.P("Red: NAIC 3+ above the HY limit, or a negative funding balance in the check years.",
                   className="chart-caption")]),
        html.Div(className="finding", children=[html.Div("What a PM might do", className="finding-index"),
                                                html.H3(LABELS[name]), html.P(note)]),
    ])


def tight_capital_block():
    t = m("phase6.tight_capital")
    h = table("tighter_capital_reoptimization", index_col=0)
    h = h[h.change.abs() > 1e-6].sort_values("change")
    return html.Div([
        html.Div(className="kpi-row", children=[
            kpi("Tighter budget K", fmt(t["budget"], ".2f"), "$M", f"from {fmt(m('inputs.capital_budget'), '.2f')}"),
            kpi("Re-optimized income", fmt(t["income"]), "$M / yr", f"{fmt(t['income_change'] * 1000, '+.1f')}k vs base C"),
            kpi("Base C at tight budget", fmt(t["C_utilization_at_tight_budget"], ".0%"), "of K",
                "old holdings exceed the new budget" if t["C_utilization_at_tight_budget"] > 1
                else "old holdings fit the new budget"),
            kpi("NAIC 3+ after re-optimization", fmt(t["hy_after"], ".1f"), "$M",
                f"limit {fmt(m('inputs.limits.hy_max'), '.0f')}"),
        ]),
        html.Div(className="panel section-tight", children=report_table(
            ["Security", "Base C, $M", "Tight K, $M", "Change, $M"],
            [[i, fmt(r.base, ".2f"), fmt(r.tight, ".2f"), fmt(r.change, "+.2f")] for i, r in h.iterrows()],
            numeric_cols={1, 2, 3})),
    ])

