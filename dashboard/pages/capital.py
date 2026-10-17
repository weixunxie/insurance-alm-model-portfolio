"""Capital (Module C): modeled NAIC C-1 charges, capital efficiency, factor provenance, CLO regime, downgrades."""

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, ctx, dcc, html

from dashboard.components import (BUCKET_NAMES, BUCKET_ORDER, NAVY, PORTFOLIO_COLORS, PORTFOLIO_NAMES, PORTFOLIOS,
                                  chart, fmt, kpi, page_header, report_table, section)
from dashboard.data import analysis, base, clo_regime, config_csv, config_csv_header, holdings, m, table

BOND_CSV, CLO_CSV = "naic_life_rbc_bond_factors.csv", "naic_life_rbc_clo_factors.csv"
REGIME_NAMES = {"new_2026": "2026 CLO factors", "old_bond": "Old bond factors"}


def _kpis():
    budget = base().policy["capital"]["budget"]
    tiles = []
    for p in PORTFOLIOS:
        cap = analysis(p)["cap"]
        tiles.append(kpi(f"Modeled C-1 charge, {p}", fmt(cap["capital_charge"]), "$M",
                         f"{fmt(cap['utilization'], '.0%')} of {fmt(budget, '.2f')} budget; "
                         f"efficiency {fmt(cap['capital_efficiency'], '.2f')}"))
    tiles.append(kpi("Downgrade buffer limit", fmt(m("report.downgrade_buffer_limit"), ".2f"), "$M",
                     f"{fmt(m('inputs.downgrade_buffer_multiple'), '.2f')} × K after a one-notch BBB-family downgrade"))
    return html.Div(className="kpi-row", children=tiles)


def bucket_figure() -> go.Figure:
    fig = go.Figure()
    names = [BUCKET_NAMES[b] for b in BUCKET_ORDER]
    for p in PORTFOLIOS:
        by = analysis(p)["cap_rows"].groupby("bucket").charge.sum().reindex(BUCKET_ORDER).fillna(0.0)
        fig.add_bar(y=names, x=by.to_numpy(), orientation="h", name=PORTFOLIO_NAMES[p],
                    marker_color=PORTFOLIO_COLORS[p], hovertemplate="%{y}: $%{x:.3f}M (" + p + ")<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25, xaxis=dict(title="Modeled C-1 charge, $M", showgrid=True,
                      gridcolor="#EEEBE5"), yaxis=dict(autorange="reversed", showgrid=False))
    return fig


EFF_COLS = [("id", "Security", None), ("bucket", "Bucket", None), ("naic", "NAIC", None),
            ("net_spread_bp", "Net spread (bp)", ".1f"), ("factor_pct", "C-1 factor (%)", ".3f"),
            ("efficiency", "Efficiency (net spread / factor)", ".2f"), ("held_c", "Held in C ($M)", ".2f")]


def efficiency_rows() -> list[dict]:
    u = base().universe
    held = holdings("C").reindex(u.id).fillna(0.0).to_numpy()
    exempt = ~np.isfinite(u.capital_efficiency.to_numpy())
    return [{"id": r.id, "bucket": BUCKET_NAMES.get(r.bucket, r.bucket), "naic": r.naic_designation,
             "net_spread_bp": r.net_spread * 1e4, "factor_pct": r.cap_factor * 100,
             "efficiency": None if ex else float(r.capital_efficiency), "held_c": float(h)}
            for r, ex, h in zip(u.itertuples(index=False), exempt, held)]


def efficiency_table(sort_col: str = "efficiency", descending: bool = True):
    rows = efficiency_rows()
    present = [r for r in rows if r[sort_col] is not None]
    missing = [r for r in rows if r[sort_col] is None]          # exempt rows always last
    present.sort(key=lambda r: r[sort_col], reverse=descending)
    arrow = lambda c: (" \u2193" if descending else " \u2191") if c == sort_col else ""
    header = [html.Button(label + arrow(c), id={"type": "eff-sort", "col": c}, className="sort-button")
              for c, label, _ in EFF_COLS]
    body = [[("exempt" if r[c] is None else fmt(r[c], spec)) if spec else r[c] for c, _, spec in EFF_COLS]
            for r in present + missing]
    return report_table(header, body, numeric_cols={3, 4, 5, 6})


@callback(Output("eff-table", "children"), Output("eff-sort-state", "data"),
          Input({"type": "eff-sort", "col": ALL}, "n_clicks"), State("eff-sort-state", "data"),
          prevent_initial_call=True)
def sort_efficiency(_clicks, state):
    col = ctx.triggered_id["col"]
    descending = not state["descending"] if col == state["col"] else col not in ("id", "bucket", "naic")
    return efficiency_table(col, descending), {"col": col, "descending": descending}


def factor_table(name: str, factor_col: str):
    df = config_csv(name).fillna("")
    cols = [c for c in df.columns if c != factor_col]
    header = ["Designation" if c == "designation" else c.replace("_", " ").capitalize() for c in cols]
    header.insert(1, "Factor (pre-tax)")
    rows = [[r["designation"], fmt(float(r[factor_col]), ".3%")] + [r[c] for c in cols[1:]] for _, r in df.iterrows()]
    return html.Div([
        html.Div([html.P(line, className="provenance-line") for line in config_csv_header(name)],
                 className="provenance-block"),
        report_table(header, rows, numeric_cols={1}),
        html.P(f"Source file: config/{name}", className="chart-caption"),
    ])


def clo_regime_block():
    rows = []
    for p in PORTFOLIOS:
        r = clo_regime(p).set_index("regime")
        rows.append([PORTFOLIO_NAMES[p], fmt(r.loc["new_2026", "clo_charge"]), fmt(r.loc["old_bond", "clo_charge"]),
                     fmt(r.loc["new_2026", "total_charge"]), fmt(r.loc["old_bond", "total_charge"])])
    reopt = table("clo_regime_reoptimization")
    shift = table("clo_regime_reoptimized_holdings", index_col=0)
    shift = shift[shift.change.abs() > 1e-6]
    return html.Div([
        html.H3("(a) Same holdings, repriced", className="sub-head"),
        report_table(["Portfolio", "CLO charge, 2026 factors", "CLO charge, old factors",
                      "Total charge, 2026 factors", "Total charge, old factors"], rows, numeric_cols={1, 2, 3, 4}),
        html.H3("(b) Re-optimized under each regime", className="sub-head"),
        html.Div(className="grid-2", children=[
            report_table(["Case", "Net income, $M / yr", "Capital used, $M"],
                         [[r.case, fmt(r.net_income), fmt(r.capital_used)] for r in reopt.itertuples(index=False)],
                         numeric_cols={1, 2}),
            report_table(["Security", "2026 factors, $M", "Old factors, $M", "Change, $M"],
                         [[i, fmt(r.new_2026, ".2f"), fmt(r.old_bond, ".2f"), fmt(r.change, "+.2f")]
                          for i, r in shift.iterrows()], numeric_cols={1, 2, 3}),
        ]),
        html.P("Saved results from scripts/run_all.py (Portfolio C and its re-optimization). The Optimizer page can "
               "re-solve live under either regime.", className="chart-caption"),
    ])


def downgrade_figure(hy_limit: float) -> go.Figure:
    ds = table("downgrade_sensitivity")
    ds = ds[ds.bucket == "ALL"].set_index(["portfolio", "notches"])
    stages = ["Today", "One notch", "Two notches"]
    fig = go.Figure()
    for p in PORTFOLIOS:
        vals = [m(f"phase5.portfolios.{p}.hy"), ds.loc[(p, 1), "hy_after"], ds.loc[(p, 2), "hy_after"]]
        fig.add_bar(x=stages, y=vals, name=PORTFOLIO_NAMES[p], marker_color=PORTFOLIO_COLORS[p],
                    hovertemplate="%{x}: NAIC 3+ $%{y:.1f}M (" + p + ")<extra></extra>")
    fig.add_hline(y=hy_limit, line=dict(color=NAVY, width=1, dash="dash"),
                  annotation_text=f"HY limit {hy_limit:.0f}", annotation_position="top left")
    fig.update_layout(barmode="group", yaxis_title="NAIC 3+ holdings, $M", xaxis_title="Downgrade applied")
    return fig


def downgrade_table(hy_limit: float):
    ds = table("downgrade_sensitivity")
    ds = ds[ds.bucket == "ALL"]
    rows, breaches = [], set()
    for i, r in enumerate(ds.itertuples(index=False)):
        charge = analysis(r.portfolio)["cap"]["capital_charge"]
        rows.append([PORTFOLIO_NAMES[r.portfolio], str(r.notches), fmt(r.d_capital, "+.3f"),
                     fmt(charge + r.d_capital), fmt(r.d_net_income * 1000, "+.1f"), fmt(r.hy_after, ".1f"),
                     "Yes" if r.hy_limit_breached else "No"])
        if r.hy_limit_breached:
            breaches |= {(i, 5), (i, 6)}
    return report_table(["Portfolio", "Notches", "Δ capital, $M", "Capital after, $M", "Δ net income, $k / yr",
                         "NAIC 3+ after, $M", f"Above HY limit ({hy_limit:.0f})"], rows,
                        numeric_cols={1, 2, 3, 4, 5}, breaches=breaches)


def layout():
    hy_limit = base().policy["limits"]["hy_max"]
    return html.Div([
        page_header("Capital (NAIC C-1)",
                    "Modeled charge = t0 market value × Life RBC C-1 factor by NAIC designation. CLO tranches use "
                    "the 12/31/2026 CLO factors. Simplified: no bond size factor, no tax effect, no covariance, "
                    "no RBC ratio."),
        _kpis(),
        section("Capital by bucket", chart("Modeled C-1 charge by bucket, $M", bucket_figure(), height=360,
                                           caption="Treasuries and GNMA are exempt (0%) in this model.")),
        section("Capital efficiency by security",
                dcc.Store(id="eff-sort-state", data={"col": "efficiency", "descending": True}),
                html.Div(efficiency_table(), id="eff-table"),
                note="Net spread after expected loss per unit of C-1 factor. Click a header to sort."),
        section("C-1 bond factors", factor_table(BOND_CSV, "factor_pre_tax")),
        section("CLO factors, effective 12/31/2026", factor_table(CLO_CSV, "factor")),
        section("CLO factor regime: 2026 CLO factors vs old bond factors", clo_regime_block()),
        section("Downgrade sensitivity",
                chart("NAIC 3+ holdings after downgrades, $M", downgrade_figure(hy_limit), height=300,
                      caption="Every eligible holding moved down one or two notches (Treasuries and agencies exempt)."),
                html.Div(className="panel section-tight", children=[
                    downgrade_table(hy_limit),
                    html.P(f"These rows downgrade every eligible holding. The LP's downgrade buffer uses a narrower "
                           f"rule (BBB family only, one notch): C's capital after that rule is "
                           f"{fmt(m('phase6.constraints.capital_downgraded_after'))} $M against a limit of "
                           f"{fmt(m('report.downgrade_buffer_limit'), '.2f')}.", className="chart-caption")]),
                note="Capital at t0 market value (book-value proxy)."),
    ])
