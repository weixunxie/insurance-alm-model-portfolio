"""Overview: business question, Portfolio C headline KPIs, A / B / C comparison, allocation, findings."""

import plotly.graph_objects as go
from dash import html

from dashboard.components import (BUCKET_COLORS, BUCKET_NAMES, BUCKET_ORDER, PORTFOLIO_NAMES, PORTFOLIOS,
                                  chart, fmt, fmt_usd_k, kpi, page_header, report_table, section, swatch,
                                  PORTFOLIO_COLORS)
from dashboard.data import m, table

TOL = 1e-6


def _kpis():
    pa, pc = m("phase5.portfolios.A"), m("phase5.portfolios.C")
    budget, max_loss = m("inputs.capital_budget"), m("inputs.scenario_max_loss")
    return html.Div(className="kpi-row", children=[
        kpi("Net spread income, C", fmt(pc["net_spread_income"]), "$M / yr",
            f"vs {fmt(pa['net_spread_income'])} for A"),
        kpi("Capital used, C", fmt(pc["capital_charge"]), "$M",
            f"of {fmt(budget, '.2f')} budget ({fmt(pc['budget_utilization'], '.0%')})"),
        kpi("Max |net key rate DV01|, C", fmt(pc["max_abs_net_krd_k"], ".1f"), "$k / bp",
            f"vs {fmt(pa['max_abs_net_krd_k'], '.1f')} for A"),
        kpi("Surplus change at −100bp, C", fmt(m("phase5.stress.C_parallel_down_100_dS"), "+.2f"), "$M",
            f"limit −{fmt(max_loss, '.2f')} (full repricing)"),
    ])


def _comparison_rows():
    """(label, value per portfolio, format, breach test or None). Values from metrics.json."""
    port = lambda key: {p: m(f"phase5.portfolios.{p}.{key}") for p in PORTFOLIOS}
    stress = lambda sc: {p: m(f"phase5.stress.{p}_{sc}_dS") for p in PORTFOLIOS}
    dv01_max, budget = m("inputs.rate_risk.surplus_dv01_max"), m("inputs.capital_budget")
    max_loss = m("inputs.scenario_max_loss")
    loss_breach = lambda v: v < -max_loss - TOL
    return [
        ("Net spread income (after EL), $M / yr", port("net_spread_income"), ".3f", None),
        ("Expected credit loss, $M / yr", port("expected_credit_loss"), ".3f", None),
        ("Portfolio yield", port("yield"), ".2%", None),
        ("Asset effective duration, years", port("effective_duration"), ".2f", None),
        (f"Surplus DV01, $k / bp (limit ±{fmt(dv01_max, '.0f')})", port("surplus_dv01_k"), "+.1f",
         lambda v: abs(v) > dv01_max + TOL),
        ("Max |net key rate DV01|, $k / bp", port("max_abs_net_krd_k"), ".1f", None),
        ("Key rate tenors outside tolerance", port("krd_tenors_breaching"), ".0f", lambda v: v > 0),
        ("Rate limits met (surplus DV01 and every tenor)", port("rate_limits_ok"), None, lambda v: not v),
        ("Bucket, issuer and liquidity limits met", port("bucket_limits_ok"), None, lambda v: not v),
        (f"Modeled C-1 charge, $M (budget {fmt(budget, '.2f')})", port("capital_charge"), ".3f",
         lambda v: v > budget + TOL),
        ("Capital efficiency (net income / charge)", port("capital_efficiency"), ".2f", None),
        ("Δ surplus, rates +100bp, $M", stress("parallel_up_100"), "+.2f", loss_breach),
        ("Δ surplus, rates −100bp, $M", stress("parallel_down_100"), "+.2f", loss_breach),
        ("Δ surplus, steepener, $M", stress("steepener"), "+.2f", loss_breach),
        ("Δ surplus, HY (NAIC 3+) spreads +200bp, $M", stress("hy_spread_up_200"), "+.2f", None),
    ]


def _comparison_table():
    rows, breaches = [], set()
    for i, (label, values, spec, breach) in enumerate(_comparison_rows()):
        cells = [label]
        for j, p in enumerate(PORTFOLIOS, start=1):
            v = values[p]
            cells.append(("Yes" if v else "No") if isinstance(v, bool) else fmt(v, spec))
            if breach is not None and breach(v):
                breaches.add((i, j))
        rows.append(cells)
    header = ["Metric"] + [html.Span([swatch(PORTFOLIO_COLORS[p]), PORTFOLIO_NAMES[p]]) for p in PORTFOLIOS]
    return report_table(header, rows, numeric_cols={1, 2, 3}, breaches=breaches)


def allocation_figure(alloc) -> go.Figure:
    """Stacked horizontal bars: one bar per portfolio, one segment per bucket (USD millions = % of book)."""
    alloc = alloc.reindex(BUCKET_ORDER).fillna(0.0)
    names = [PORTFOLIO_NAMES[p] for p in alloc.columns]
    fig = go.Figure()
    for b in BUCKET_ORDER:
        vals = alloc.loc[b].to_numpy()
        fig.add_bar(y=names, x=vals, name=BUCKET_NAMES[b], orientation="h",
                    marker=dict(color=BUCKET_COLORS[b], line=dict(color="#FFFFFF", width=2)),
                    text=[f"{v:.0f}" if v >= 5 else "" for v in vals], textposition="inside",
                    insidetextanchor="middle", textfont=dict(color="#FFFFFF", size=12),
                    hovertemplate="%{y}<br>" + BUCKET_NAMES[b] + ": $%{x:.2f}M<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.35, legend=dict(traceorder="normal"),
                      xaxis=dict(title="Market value, $M (= % of the $100M book)", range=[0, 100.5],
                                 showgrid=True, gridcolor="#EEEBE5"),
                      yaxis=dict(autorange="reversed", showgrid=False))
    return fig


def _findings():
    c6, cr = m("phase6.constraints"), m("phase4.clo_regime")
    rob = m("phase5.robustness")
    items = [
        ("The LP cannot see convexity, so scenario rows were added",
         f"Duration and key rate rows are linear, and Agency MBS prepayment makes asset value concave in rates. "
         f"Without scenario rows, C would lose {fmt(-c6['dS_parallel_down_100_C_before'], '.2f')} $M of surplus "
         f"at −100bp, beyond the {fmt(c6['max_loss'], '.2f')} limit. Full-repricing rows hold it at "
         f"{fmt(c6['dS_parallel_down_100_C_after'], '+.2f')}. Scenario rows cost {fmt_usd_k(c6['scenario_cost'])} "
         f"a year; the downgrade buffer a further {fmt_usd_k(c6['buffer_cost'])}."),
        ("2026 CLO factors reshape the allocation",
         f"C's holdings need {fmt(cr['C_capital_new_2026'])} $M of capital under the 12/31/2026 CLO factors "
         f"versus {fmt(cr['C_capital_old_bond_same_holdings'])} $M under the old bond factors. Re-optimized on "
         f"the old factors, the LP lifts CLO BB from {fmt(m('phase4.clo_bb_new_2026'), '.2f')} to "
         f"{fmt(m('phase4.clo_bb_old_bond'), '.2f')} $M and earns {fmt(cr['reoptimized_old_bond_income'])} "
         f"$M / yr instead of {fmt(m('phase5.portfolios.C.net_spread_income'))}."),
        ("The optimizer's curse",
         f"Over {rob['n_draws']} perturbed spread and PD draws, re-optimized portfolios earn "
         f"{fmt(rob['income_reopt_in_sample_mean'])} $M / yr in-sample but "
         f"{fmt(rob['income_reopt_out_of_sample_mean'])} out-of-sample, versus "
         f"{fmt(rob['income_fixed_C_mean'])} for base C held fixed. The in-sample gap of "
         f"{fmt(rob['optimizers_curse_gap'], '+.4f')} is fitting to noise, not skill."),
    ]
    return html.Div(className="grid-3", children=[
        html.Div(className="finding", children=[html.Div(f"Finding {i}", className="finding-index"),
                                                html.H3(title), html.P(text)])
        for i, (title, text) in enumerate(items, start=1)])


def provenance():
    p = m("provenance")
    dirty = " (with uncommitted changes)" if p["dirty"] else ""
    return html.P(f"Results generated {p['generated']} from commit {p['commit']}{dirty} by scripts/run_all.py. "
                  "This page shows saved results; the Optimizer page solves live.", className="provenance")


def layout():
    lede = (f"How much credit spread can a ${fmt(m('inputs.total_assets'), '.0f')}M life insurer portfolio "
            f"earn while staying inside a modeled NAIC C-1 capital budget of ${fmt(m('inputs.capital_budget'), '.2f')}M, "
            f"key rate DV01 limits against its liabilities (PV ${fmt(m('inputs.liability_pv_target'), '.0f')}M), "
            f"and a cash flow funding test for years 1 to {m('inputs.funding.check_years')}? Portfolio C is the "
            "linear program's answer; A and B are rule-based references.")
    return html.Div([
        page_header("Overview", lede),
        provenance(),
        _kpis(),
        section("A / B / C comparison", _comparison_table(),
                note="USD millions unless noted. Red: outside a policy limit."),
        section("Allocation by bucket",
                chart("Allocation by bucket, $M of market value", allocation_figure(
                    table("portfolio_allocation_by_bucket", index_col="bucket")), height=300)),
        section("Findings", _findings()),
    ])
