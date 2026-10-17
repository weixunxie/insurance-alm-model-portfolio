"""Optimizer (interactive): re-solve the integrated LP for user-chosen policy controls."""

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from dashboard.components import (BUCKET_NAMES, BUCKET_ORDER, NAVY, PORTFOLIO_COLORS, chart, fmt, kpi,
                                  net_krd_figure, page_header, report_table, section)
from dashboard.data import base, table
from dashboard.solve import Controls, base_controls, frontier_for, hy_after_cost, solve_controls

C_COLOR = PORTFOLIO_COLORS["C"]
TOGGLES = [("scenario", "Scenario surplus constraints"), ("buffer", "Downgrade capital buffer"),
           ("hy_after", "NAIC 3+ after one-notch downgrade \u2264 HY limit"),
           ("overlay", "Receive-fixed swap overlay")]
REGIMES = [("new_2026", "2026 CLO factors"), ("old_bond", "Old bond factors for CLOs")]


def _toggle_values(c: Controls) -> list:
    return [k for k, on in (("scenario", c.scenario_on), ("buffer", c.buffer_on), ("hy_after", c.hy_after_on),
                            ("overlay", c.overlay_on)) if on]


def controls_panel():
    c = base_controls()
    label = lambda text: html.Div(text, className="control-label")
    return html.Div(className="panel controls", children=[
        html.H3("Policy controls"),
        label("Capital budget K, $M"),
        dcc.Slider(id="opt-k", min=0.40, max=3.00, step=0.05, value=c.budget_k,
                   marks={str(v): f"{v:.2f}" for v in (0.4, 1.2, 2, 3)},
                   tooltip={"placement": "top", "always_visible": True}),
        label("Constraints"),
        dcc.Checklist(id="opt-toggles", options=[{"label": t, "value": k} for k, t in TOGGLES],
                      value=_toggle_values(c), className="checklist", inputClassName="check-input"),
        label("Max surplus loss at −100bp, $M"),
        dcc.Slider(id="opt-maxloss", min=0.0, max=0.50, step=0.05, value=c.max_loss_down,
                   marks={str(v): f"{v:.2f}" for v in (0, 0.25, 0.5)},
                   tooltip={"placement": "top", "always_visible": True}),
        html.Div(id="opt-maxloss-hint", className="control-hint"),
        label("CLO factor regime"),
        dcc.RadioItems(id="opt-regime", options=[{"label": t, "value": k} for k, t in REGIMES],
                       value=c.regime, className="checklist", inputClassName="check-input"),
        html.Button("Reset to base case", id="opt-reset", className="button", n_clicks=0),
        html.P("Other limits (KRD tolerance, surplus DV01, funding, bucket, issuer, liquidity) stay at "
               "config/policy.yaml values.", className="control-help"),
    ])


def layout():
    return html.Div([
        page_header("Optimizer",
                    "Change the capital budget and policy switches; the linear program is re-solved live "
                    "(CVXPY + HiGHS) on the cached universe, liabilities and scenario rows."),
        html.Div(className="opt-grid", children=[
            controls_panel(),
            dcc.Loading(html.Div(id="opt-results"), type="dot", color=NAVY, delay_show=300),
        ]),
    ])


# ---------- figures ----------

def allocation_figure(res: dict) -> go.Figure:
    live = res["allocation"].reindex(BUCKET_ORDER).fillna(0.0)
    saved = table("portfolio_allocation_by_bucket", index_col="bucket")["C"].reindex(BUCKET_ORDER).fillna(0.0)
    names = [BUCKET_NAMES[b] for b in BUCKET_ORDER]
    fig = go.Figure()
    fig.add_bar(y=names, x=live.to_numpy(), orientation="h", name="Live solve", marker_color=C_COLOR,
                text=[f"{v:.1f}" for v in live], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: $%{x:.2f}M (live)<extra></extra>")
    fig.add_scatter(y=names, x=saved.to_numpy(), mode="markers", name="Base case C (saved)",
                    marker=dict(symbol="line-ns", size=18, line=dict(width=2, color=NAVY)),
                    hovertemplate="%{y}: $%{x:.2f}M (saved base case)<extra></extra>")
    fig.update_layout(xaxis=dict(title="Market value, $M", showgrid=True, gridcolor="#EEEBE5",
                                 range=[0, max(live.max(), saved.max()) * 1.15]),
                      yaxis=dict(autorange="reversed", showgrid=False), bargap=0.35)
    return fig


def krd_figure(res: dict) -> go.Figure:
    return net_krd_figure(res["tenors"], res["net_krd_k"], res["tolerance_k"], C_COLOR)


def frontier_figure(f, res: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=f.budget_k, y=f.net_income, mode="lines+markers", name="Efficient frontier",
                    line=dict(color=C_COLOR, width=2), marker=dict(size=6, color=C_COLOR),
                    customdata=np.stack([f.capital_used, f.shadow_price_capital], axis=1),
                    hovertemplate=("K = $%{x:.2f}M<br>capital used $%{customdata[0]:.3f}M<br>"
                                   "income $%{y:.3f}M/yr<br>shadow price %{customdata[1]:.3f}<extra></extra>"))
    fig.add_scatter(x=[res["budget_k"]], y=[res["net_income"]], mode="markers+text", name="Current solution",
                    marker=dict(size=12, color=NAVY, line=dict(color="#FFFFFF", width=2)),
                    text=["Current"], textposition="top left", customdata=[res["capital_used"]],
                    hovertemplate=("Current: K = $%{x:.2f}M, capital used $%{customdata:.3f}M, "
                                   "income $%{y:.3f}M/yr<extra></extra>"))
    fig.update_layout(xaxis=dict(title="Capital budget K, $M", showgrid=False),
                      yaxis_title="Net spread income, $M per year")
    return fig


# ---------- result blocks ----------

def shadow_sentence(res: dict) -> str:
    sp = res["shadow_price"]
    if sp < 1e-6:
        return ("Capital is not binding at this budget: one more $1 of K adds no net spread; "
                "other limits bind instead.")
    return f"One more $1 of capital budget adds about ${sp:.3f} of net spread per year."


def binding_capital_note(res: dict, policy: dict):
    """One line when the post-downgrade buffer binds but the base C-1 row does not."""
    rows = set(res["binding"].constraint)
    if "downgrade_buffer" in rows and "capital" not in rows:
        mult = policy["downgrade_buffer"]["multiple"]
        return html.P(f"Binding capital limit: post-downgrade buffer ({mult:.2f} \u00d7 K), not the base C-1 charge.",
                      className="shadow-note")
    return None


def hy_after_note(c: Controls):
    cost = hy_after_cost(c)
    if cost is None:
        return None
    return html.P(f"NAIC 3+ after-downgrade limit is on: it costs ${cost * 1000:.1f}k of net spread a year "
                  "versus the same settings without it.", className="shadow-note")


def binding_table(res: dict):
    rows = []
    for r in res["binding"].itertuples(index=False):
        units = "$k/bp" if r.group in ("krd", "surplus_dv01") else "$M"
        rows.append([r.constraint, r.sense, fmt(r.value), fmt(r.limit), units, fmt(r.dual, ".4f"), r.dual_units])
    return report_table(["Constraint", "Sense", "Value", "Limit", "Units", "Dual", "Dual units"], rows,
                        numeric_cols={2, 3, 5})


def infeasible_block(res: dict):
    v = res.get("violations")
    rows = [] if v is None else [[r.constraint, fmt(r.slack_used)] for r in v.itertuples(index=False)]
    return html.Div(className="panel infeasible", children=[
        html.H3("No portfolio satisfies every constraint"),
        html.P(f"Hard solve status: {res.get('hard_status')}. A soft re-solve with penalized slack shows the "
               "smallest relaxation that restores feasibility:"),
        report_table(["Constraint", "Relaxation needed (row units)"], rows, numeric_cols={1}),
    ])


def results(c: Controls):
    res = solve_controls(c)
    policy = base().policy
    rr, fr = policy["rate_risk"], policy["capital"]["frontier"]
    timing = f"LP solved in {res['solve_ms']:.0f} ms (HiGHS)"
    tag = html.Div([html.Span("Live solve", className="live-label"), html.Span(timing, className="live-time")],
                   className="live-tag")
    if not res["feasible"]:
        return html.Div([tag, infeasible_block(res)])

    swaps = res["swaps"]
    swap_line = (html.P("Receive-fixed swap notionals ($M, zero modeled C-1): " + ", ".join(
        f"{i.removeprefix('SWAP_RECV_')} {v:.1f}" for i, v in swaps.items()), className="muted")
        if len(swaps) else None)
    dv01_max = res["surplus_dv01_max"]
    return html.Div([
        tag,
        html.Div(className="kpi-row", children=[
            kpi("Net spread income", fmt(res["net_income"]), "$M / yr", "after expected credit loss"),
            kpi("Capital used", fmt(res["capital_used"]), "$M",
                f"of {fmt(res['budget_k'], '.2f')} budget ({fmt(res['capital_used'] / res['budget_k'], '.0%')})"),
            kpi("Shadow price of K", fmt(res["shadow_price"]), "$ / $1 of K",
                f"dual_capital + {policy['downgrade_buffer']['multiple']:.2f} × dual_buffer" if c.buffer_on
                else "dual of the capital row"),
            kpi("Surplus DV01", fmt(res["surplus_dv01_k"], "+.2f"), "$k / bp", f"limit ±{fmt(dv01_max, '.0f')}"),
        ]),
        html.P(shadow_sentence(res), className="shadow-note"),
        binding_capital_note(res, policy),
        hy_after_note(c),
        html.Div(className="grid-2 section-tight", children=[
            chart("Allocation by bucket, $M of market value", allocation_figure(res), height=320,
                  caption="Bars: live solve. Ticks: saved base-case Portfolio C."),
            chart("Net key rate DV01 by tenor, $k per bp", krd_figure(res), height=320,
                  caption=(f"Gray band: per-tenor tolerance max(${rr['krd_floor']:.0f}k, "
                           f"{rr['krd_pct_of_liability']:.0%} of liability key rate DV01).")),
        ]),
        swap_line,
        html.Div(className="section-tight", children=chart(
            "Efficient frontier: net spread income vs capital budget K, $M", frontier_figure(frontier_for(c), res),
            height=340, caption=(f"Re-solved over K = ${fr['min']:.2f}M to ${fr['max']:.2f}M "
                                 "with the current switches. Hover shows capital used."))),
        section("Binding constraints", binding_table(res),
                note="KRD rows in $k per bp; all other rows in $M. Duals as reported by the LP."),
    ])


def _controls(k, toggles, max_loss, regime) -> Controls:
    toggles = toggles or []
    return Controls(float(k), "scenario" in toggles, "buffer" in toggles, "overlay" in toggles,
                    float(max_loss), regime, "hy_after" in toggles)


@callback(Output("opt-results", "children"),
          Input("opt-k", "value"), Input("opt-toggles", "value"), Input("opt-maxloss", "value"),
          Input("opt-regime", "value"))
def update(k, toggles, max_loss, regime):
    return results(_controls(k, toggles, max_loss, regime))


@callback(Output("opt-maxloss", "disabled"), Output("opt-maxloss-hint", "children"),
          Input("opt-toggles", "value"))
def toggle_maxloss(toggles):
    off = "scenario" not in (toggles or [])
    return off, ("Inactive: turn on scenario surplus constraints to use this limit." if off else "")


@callback(Output("opt-k", "value"), Output("opt-toggles", "value"), Output("opt-maxloss", "value"),
          Output("opt-regime", "value"), Input("opt-reset", "n_clicks"), prevent_initial_call=True)
def reset(n_clicks):
    c = base_controls()
    return c.budget_k, _toggle_values(c), c.max_loss_down, c.regime
