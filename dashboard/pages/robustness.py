"""Robustness: Monte Carlo re-optimization on perturbed spreads and PDs, and the optimizer's curse."""

import numpy as np
import plotly.graph_objects as go
from dash import html

from dashboard.components import (BUCKET_NAMES, NAVY, PORTFOLIO_COLORS, chart, fmt, kpi, page_header, report_table,
                                  section)
from dashboard.data import m, table

SERIES = [("net_income", "Re-optimized, in-sample", NAVY, "dash"),
          ("income_fixed_C", "Base C held fixed", PORTFOLIO_COLORS["C"], "solid"),
          ("income_reopt_oos", "Re-optimized, out-of-sample", "#6655b8", "dot")]
HEADLINE_ROWS = [("income_reopt_in_sample", "Re-optimized, in-sample"), ("income_fixed_C", "Base C held fixed"),
                 ("income_reopt_out_of_sample", "Re-optimized, out-of-sample"), ("capital_used", "Capital used")]


def draws():
    d = table("robustness_draws")
    return d[d.status == "optimal"]


def income_figure() -> go.Figure:
    d = draws()
    lo = min(d[c].min() for c, *_ in SERIES)
    hi = max(d[c].max() for c, *_ in SERIES)
    edges = np.linspace(lo, hi, 31)
    mids = 0.5 * (edges[:-1] + edges[1:])
    fig = go.Figure()
    for col, name, color, dash in SERIES:
        counts, _ = np.histogram(d[col], bins=edges)
        fig.add_scatter(x=mids, y=counts, mode="lines", line=dict(color=color, width=2, dash=dash, shape="hvh"),
                        name=name, hovertemplate=name + ": %{y} draws near $%{x:.3f}M<extra></extra>")
        fig.add_vline(x=float(d[col].mean()), line=dict(color=color, width=1, dash=dash))
    fig.update_layout(xaxis_title="Net spread income, $M per year", yaxis_title="Number of draws")
    return fig


def layout():
    rob, inp = m("phase5.robustness"), m("inputs.robustness")
    curse = (f"Re-optimizing on each noisy draw earns {fmt(rob['income_reopt_in_sample_mean'])} $M a year on average "
             f"when judged on that same draw, {fmt(rob['optimizers_curse_gap'] * 1000, '.1f')}k more than base C held "
             f"fixed ({fmt(rob['income_fixed_C_mean'])}); that gain is the optimizer fitting the noise. Judged on an "
             f"independent draw, the re-optimized portfolios earn {fmt(rob['income_reopt_out_of_sample_mean'])}, "
             f"{fmt(abs(rob['oos_minus_fixed_C']) * 1000, '.1f')}k "
             f"{'less' if rob['oos_minus_fixed_C'] < 0 else 'more'} than simply keeping C.")
    head = table("robustness_headline", index_col=0)
    pos = table("robustness_positions", index_col=0)
    pos = pos[pos.pct_draws_held > 0].sort_values(["pct_draws_at_issuer_cap", "mean"], ascending=False)
    buckets = table("robustness_bucket_weights", index_col="bucket")
    return html.Div([
        page_header("Robustness",
                    f"{rob['n_draws']} Monte Carlo draws (seed {inp['seed']}): every spread × N(1, "
                    f"{fmt(inp['spread_sd'], '.2f')}) and every PD × a mean-one lognormal with σ = "
                    f"{fmt(inp['pd_sigma'], '.2f')}. The LP is re-solved on each draw."),
        html.Div(className="kpi-row", children=[
            kpi("Re-optimized, in-sample", fmt(rob["income_reopt_in_sample_mean"]), "$M / yr", "mean over draws"),
            kpi("Base C held fixed", fmt(rob["income_fixed_C_mean"]), "$M / yr", "mean over the same draws"),
            kpi("Re-optimized, out-of-sample", fmt(rob["income_reopt_out_of_sample_mean"]), "$M / yr",
                "evaluated on a fresh draw"),
            kpi("Optimizer's curse gap", fmt(rob["optimizers_curse_gap"] * 1000, "+.1f"), "$k / yr",
                f"re-optimized beats fixed C in-sample in {fmt(rob['pct_draws_reopt_beats_fixed_C_in_sample'], '.0%')} of draws"),
        ]),
        html.P(curse, className="shadow-note"),
        html.Div(className="section-tight", children=chart(
            "Distribution of net spread income across draws, $M per year", income_figure(), height=360,
            caption=f"Outlines on common bins; vertical lines mark the means. {rob['n_optimal']} of "
                    f"{rob['n_draws']} draws solved to optimality.")),
        section("Income and capital percentiles",
                report_table(["", "p5", "p25", "p50", "p75", "p95", "Mean"],
                             [[label] + [fmt(head.loc[k, c]) for c in ("p5", "p25", "p50", "p75", "p95", "mean")]
                              for k, label in HEADLINE_ROWS], numeric_cols={1, 2, 3, 4, 5, 6}),
                note="USD millions."),
        section("Position stability",
                report_table(["Security", "Base C, $M", "Mean across draws, $M", "Held in % of draws",
                              "At issuer cap in % of draws"],
                             [[i, fmt(r.base_C, ".2f"), fmt(r["mean"], ".2f"), fmt(r.pct_draws_held, ".1%"),
                               "–" if np.isnan(r.pct_draws_at_issuer_cap) else fmt(r.pct_draws_at_issuer_cap, ".0%")]
                              for i, r in pos.iterrows()], numeric_cols={1, 2, 3, 4}),
                note="Securities held in at least one draw. – : issuer exempt from the single-issuer cap."),
        section("Bucket weights across draws",
                report_table(["Bucket", "Base C, $M", "Mean, $M", "p5, $M", "p95, $M"],
                             [[BUCKET_NAMES.get(b, b), fmt(r.base_C, ".2f"), fmt(r["mean"], ".2f"), fmt(r.p5, ".2f"),
                               fmt(r.p95, ".2f")] for b, r in buckets.iterrows()], numeric_cols={1, 2, 3, 4})),
    ])
