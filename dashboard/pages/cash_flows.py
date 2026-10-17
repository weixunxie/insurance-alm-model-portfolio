"""Cash flows (Module A): expected asset inflows vs liability outflows, gaps and the funding balance."""

import plotly.graph_objects as go
from dash import Input, Output, callback, html

from dashboard.components import (BAND, LIABILITY, NAVY, NEG, PORTFOLIO_COLORS, PORTFOLIO_NAMES, POS, chart, fmt,
                                  kpi, page_header, portfolio_selector)
from dashboard.data import analysis, base


def layout():
    f = base().policy["cash_flow_funding"]
    return html.Div([
        page_header("Cash flows",
                    f"Expected asset inflows (coupons, scheduled principal, prepayments, net of expected default "
                    f"loss) against liability outflows (benefits, surrenders, expenses), years 1 to 30. Positive gaps "
                    f"are reinvested at the 1Y forward rate; shortfalls are funded at forward + "
                    f"{f['borrow_spread'] * 1e4:.0f}bp, a proxy for forced asset sales."),
        html.Div(className="control-row", children=[html.Span("Portfolio", className="control-label inline"),
                                                    portfolio_selector("cf-portfolio")]),
        html.Div(id="cf-body"),
    ])


def _shortfall_text(year) -> str:
    return "None" if year is None else f"Year {int(year)}"


def flows_figure(g, p: str) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=g.t, y=g.asset_cf, name=f"Asset inflows ({PORTFOLIO_NAMES[p]})", marker_color=PORTFOLIO_COLORS[p],
                hovertemplate="Year %{x}: inflow $%{y:.2f}M<extra></extra>")
    fig.add_scatter(x=g.t, y=g.liability_cf, name="Liability outflows", mode="lines+markers",
                    line=dict(color=LIABILITY, width=2), marker=dict(size=5),
                    hovertemplate="Year %{x}: outflow $%{y:.2f}M<extra></extra>")
    fig.update_layout(xaxis_title="Year", yaxis_title="$M per year")
    return fig


def gap_figure(g) -> go.Figure:
    colors = [POS if v >= 0 else NEG for v in g.gap]
    fig = go.Figure(go.Bar(x=g.t, y=g.gap, marker_color=colors, name="Annual gap",
                           hovertemplate="Year %{x}: gap %{y:+.2f} $M<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=NAVY, width=1))
    fig.update_layout(xaxis_title="Year", yaxis_title="Inflow − outflow, $M", showlegend=False)
    return fig


def balance_figure(g, check_years: int) -> go.Figure:
    fig = go.Figure()
    fig.add_vrect(x0=0.5, x1=check_years + 0.5, fillcolor=BAND, opacity=0.6, line_width=0, layer="below")
    fig.add_scatter(x=g.t, y=g.cum_reinvest, name="With reinvestment / borrowing", mode="lines",
                    line=dict(color=NAVY, width=2), hovertemplate="Year %{x}: %{y:+.2f} $M<extra></extra>")
    fig.add_scatter(x=g.t, y=g.cum_no_reinvest, name="Without reinvestment", mode="lines",
                    line=dict(color=NAVY, width=2, dash="dot"), hovertemplate="Year %{x}: %{y:+.2f} $M<extra></extra>")
    fig.add_hline(y=0, line=dict(color=NEG, width=1))
    fig.update_layout(xaxis_title="Year", yaxis_title="Cumulative funding balance, $M")
    return fig


@callback(Output("cf-body", "children"), Input("cf-portfolio", "value"))
def render(p):
    a = analysis(p)
    g, st = a["gaps"], a["gap_stats"]
    n = base().policy["cash_flow_funding"]["check_years"]
    return html.Div([
        html.Div(className="kpi-row", children=[
            kpi("First shortfall, with reinvestment", _shortfall_text(st["first_shortfall_year_cum_reinvest"]), "",
                f"years 1–30; max cumulative shortfall {fmt(st['max_cum_shortfall_cum_reinvest'], '.2f')} $M"),
            kpi("First shortfall, no reinvestment", _shortfall_text(st["first_shortfall_year_cum_no_reinvest"]), "",
                f"years 1–30; max cumulative shortfall {fmt(st['max_cum_shortfall_cum_no_reinvest'], '.2f')} $M"),
            kpi(f"Min coverage, years 1–{n}", fmt(st["min_coverage_1_to_check"], ".2f"), "×",
                "asset inflow / liability outflow"),
            kpi("PV of annual gaps", fmt(st["pv_gaps"], "+.2f"), "$M", "discounted on the base spot curve"),
        ]),
        html.Div(className="section-tight", children=chart(
            "Asset inflows vs liability outflows, $M per year", flows_figure(g, p), height=340,
            caption="Expected cash flows (after expected default loss). Liabilities are a closed block ending in year 30.")),
        html.Div(className="grid-2 section-tight", children=[
            chart("Annual gap, $M", gap_figure(g), height=300, caption="Green: surplus year. Red: shortfall year."),
            chart("Cumulative funding balance, $M", balance_figure(g, n), height=300,
                  caption=f"Shaded: years 1–{n}, where the policy requires a balance ≥ 0."),
        ]),
    ])
