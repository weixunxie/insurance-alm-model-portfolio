"""Rate risk (Module B): key rate DV01 exposure and curve scenarios, full repricing vs KRD estimate."""

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from alm.config import load_yaml
from alm.krd_exposure import scenario_curve
from dashboard.components import (LIABILITY, NAVY, PORTFOLIO_COLORS, PORTFOLIO_NAMES, chart, fmt, kpi,
                                  net_krd_figure, page_header, portfolio_selector, report_table, section)
from dashboard.data import analysis, base

SCENARIO_NAMES = {"parallel_up_100": "Parallel +100bp", "parallel_down_100": "Parallel −100bp",
                  "steepener": "Steepener (−50bp at 1Y to +50bp at 30Y)",
                  "flattener": "Flattener (+50bp at 1Y to −50bp at 30Y)"}


def layout():
    return html.Div([
        page_header("Rate risk (KRD)",
                    "Key rate DV01 of assets against liabilities, tenor by tenor. Duration matching protects "
                    "surplus against parallel moves only; key rate matching also protects against curve-shape moves."),
        html.Div(className="control-row", children=[html.Span("Portfolio", className="control-label inline"),
                                                    portfolio_selector("rr-portfolio")]),
        html.Div(id="rr-exposure"),
        section("Curve scenario explorer",
                html.Div(className="control-row", children=[
                    html.Span("Scenario", className="control-label inline"),
                    dcc.RadioItems(id="rr-scenario", value="steepener", className="segmented",
                                   inputClassName="segmented-input",
                                   options=[{"label": v.split(" (")[0], "value": k} for k, v in SCENARIO_NAMES.items()])]),
                html.Div(id="rr-scenario-body"),
                note="Full repricing re-projects rate-dependent cash flows; the KRD estimate is first order."),
    ])


def exposure_figure(ex, p: str) -> go.Figure:
    tenors = [f"{int(t)}Y" for t in ex.tenor]
    fig = go.Figure()
    fig.add_bar(x=tenors, y=ex.asset_krdv01 * 1000, name=f"Assets ({PORTFOLIO_NAMES[p]})",
                marker_color=PORTFOLIO_COLORS[p], hovertemplate="%{x}: %{y:.2f} $k/bp (assets)<extra></extra>")
    fig.add_bar(x=tenors, y=ex.liability_krdv01 * 1000, name="Liabilities", marker_color=LIABILITY,
                hovertemplate="%{x}: %{y:.2f} $k/bp (liabilities)<extra></extra>")
    fig.update_layout(barmode="group", yaxis_title="$k per bp", xaxis_title="Key tenor")
    return fig


@callback(Output("rr-exposure", "children"), Input("rr-portfolio", "value"))
def render_exposure(p):
    a = analysis(p)
    ex, dg = a["exposure"], a["dgap"]
    net, tol = ex.net_krdv01.to_numpy() * 1000, ex.tolerance.to_numpy() * 1000
    breaches = [f"{int(t)}Y" for t, v, b in zip(ex.tenor, net, tol) if abs(v) > b + 1e-6]
    dv01_max = base().policy["rate_risk"]["surplus_dv01_max"]
    return html.Div([
        html.Div(className="kpi-row", children=[
            kpi("Duration gap", fmt(dg["duration_gap"], "+.2f"), "years",
                f"D_A {fmt(dg['asset_duration'], '.2f')} − (L/A) D_L {fmt(dg['liability_duration'], '.2f')}"),
            kpi("Surplus DV01", fmt(dg["surplus_dv01"] * 1000, "+.2f"), "$k / bp", f"limit ±{fmt(dv01_max, '.0f')}"),
            kpi("Max |net key rate DV01|", fmt(np.abs(net).max(), ".1f"), "$k / bp",
                f"at {int(ex.tenor[np.abs(net).argmax()])}Y"),
            kpi("Tenors outside tolerance", str(len(breaches)), "", ", ".join(breaches) or "none"),
        ]),
        html.Div(className="grid-2 section-tight", children=[
            chart("Asset vs liability key rate DV01, $k per bp", exposure_figure(ex, p), height=320,
                  caption="1bp triangular bumps on the zero curve at each key tenor."),
            chart("Net key rate DV01 by tenor, $k per bp", net_krd_figure(ex.tenor, net, tol, PORTFOLIO_COLORS[p]),
                  height=320, caption="Gray band: per-tenor tolerance from config/policy.yaml."),
        ]),
    ])


def curve_shift_figure(name: str) -> go.Figure:
    b = base()
    t = np.arange(1, 31, dtype=float)
    shift_bp = (scenario_curve(b.curve, load_yaml("scenarios.yaml")["rate_scenarios"][name]).zero(t)
                - b.curve.zero(t)) * 1e4
    fig = go.Figure(go.Scatter(x=t, y=shift_bp, mode="lines", line=dict(color=NAVY, width=2), name="Shift",
                               hovertemplate="%{x:.0f}Y: %{y:+.0f}bp<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#C9C5BC", width=1))
    fig.update_layout(xaxis_title="Maturity, years", yaxis_title="Zero rate shift, bp", showlegend=False)
    return fig


@callback(Output("rr-scenario-body", "children"), Input("rr-portfolio", "value"), Input("rr-scenario", "value"))
def render_scenario(p, name):
    s = analysis(p)["scen"].set_index("scenario").loc[name]
    rows = [[label, fmt(s[f"{k}_full"], "+.3f"), fmt(s[f"{k}_krd"], "+.3f"), fmt(s[f"{k}_krd"] - s[f"{k}_full"], "+.3f")]
            for label, k in (("Δ assets", "dA"), ("Δ liabilities", "dL"), ("Δ surplus (ΔA − ΔL)", "dS"))]
    return html.Div(className="grid-2", children=[
        html.Div(className="panel", children=[
            html.Div(f"{SCENARIO_NAMES[name]}: {PORTFOLIO_NAMES[p]}, $M", className="chart-title"),
            report_table(["", "Full repricing", "KRD estimate", "Estimate − full"], rows, numeric_cols={1, 2, 3}),
            html.P("The gap between the two columns is convexity plus re-projected prepayment cash flows, "
                   "which the linear KRD estimate cannot see.", className="chart-caption"),
        ]),
        chart("Scenario shift to the zero curve, bp", curve_shift_figure(name), height=260),
    ])
