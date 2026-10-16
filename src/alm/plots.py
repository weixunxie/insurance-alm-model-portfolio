"""Charts. Every chart has a title, units and the caption 'Illustrative, synthetic data.'"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ASSET = "#2a78d6"        # categorical slot 1 (blue)
LIAB = "#eb6834"         # categorical slot 2 (orange)
POS, NEG = "#2a78d6", "#e34948"   # diverging poles
NEUTRAL = "#8a8985"
INK, INK_2 = "#0b0b0b", "#52514e"
CAPTION = "Illustrative, synthetic data."


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4af")
    ax.tick_params(colors=INK_2)


def _save(fig, path: Path):
    fig.text(0.01, 0.01, CAPTION, fontsize=8, color=INK_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def cashflows_asset_vs_liability(table: pd.DataFrame, path: Path, label: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.4
    ax.bar(table.t - w / 2, table.asset_cf, w, color=ASSET, label=f"Asset inflows ({label})")
    ax.bar(table.t + w / 2, table.liability_cf, w, color=LIAB, label="Liability outflows")
    _style(ax, "Projected asset vs liability cash flows", "Year", "USD millions")
    ax.legend(frameon=False)
    _save(fig, path)


def annual_gap(table: pd.DataFrame, path: Path, label: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = np.where(table.gap >= 0, POS, NEG)
    ax.bar(table.t, table.gap, 0.7, color=colors)
    ax.axhline(0, color=NEUTRAL, linewidth=1)
    _style(ax, f"Annual cash flow gap (assets - liabilities), {label}", "Year", "USD millions")
    _save(fig, path)


def cumulative_funding(table: pd.DataFrame, path: Path, label: str, check_years: int):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(table.t, table.cum_reinvest, color=ASSET, lw=2, marker="o", ms=4,
            label="With reinvestment / borrowing at 1Y forward")
    ax.plot(table.t, table.cum_no_reinvest, color=NEUTRAL, lw=2, ls="--", marker="o", ms=4,
            label="Without reinvestment (sum of gaps)")
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.axvspan(0.5, check_years + 0.5, color="#f0efec", zorder=0, label=f"Funding test window (years 1-{check_years})")
    _style(ax, f"Cumulative funding balance, {label}", "Year", "USD millions")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, path)


def krd_asset_vs_liability(exposure: pd.DataFrame, path: Path, label: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(exposure))
    w = 0.38
    ax.bar(x - w / 2, exposure.asset_krdv01 * 1000, w, color=ASSET, label=f"Assets ({label})")
    ax.bar(x + w / 2, exposure.liability_krdv01 * 1000, w, color=LIAB, label="Liabilities")
    ax.set_xticks(x, [f"{int(k)}Y" for k in exposure.tenor])
    _style(ax, "Key rate DV01: assets vs liabilities", "Key tenor", "USD thousands per bp")
    ax.legend(frameon=False)
    _save(fig, path)


def _tolerance_marks(ax, x, tol_k, width=0.8):
    """Per-tenor +/- tolerance as short dotted segments."""
    for xi, t in zip(x, tol_k):
        for y in (t, -t):
            ax.hlines(y, xi - width / 2, xi + width / 2, color=INK_2, linewidth=1, linestyles=":")


def krd_net_mismatch(exposure: pd.DataFrame, path: Path, label: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(exposure))
    net = exposure.net_krdv01 * 1000
    ax.bar(x, net, 0.6, color=np.where(net >= 0, POS, NEG))
    ax.axhline(0, color=NEUTRAL, linewidth=1)
    _tolerance_marks(ax, x, exposure.tolerance * 1000)
    for xi, v in zip(x, net):
        ax.text(xi, v, f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
    ax.set_xticks(x, [f"{int(k)}Y" for k in exposure.tenor])
    _style(ax, f"Net key rate DV01 (assets - liabilities), {label}", "Key tenor (dotted: +/- tolerance)",
           "USD thousands per bp")
    _save(fig, path)


def krd_bucket_heatmap(bucket_krd: pd.DataFrame, path: Path, label: str):
    data = bucket_krd.to_numpy() * 1000
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(bucket_krd) + 1.8))
    vmax = np.abs(data).max()
    im = ax.imshow(data, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")   # red = negative, blue = positive
    ax.set_xticks(range(data.shape[1]), [c.upper() for c in bucket_krd.columns])
    ax.set_yticks(range(data.shape[0]), bucket_krd.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = round(data[i, j], 1) + 0.0          # avoid "-0.0" from rounding noise
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 * vmax else INK)
    ax.set_title(f"Key rate DV01 by bucket and tenor, {label}", loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel("Key tenor", color=INK_2)
    fig.colorbar(im, ax=ax, label="USD thousands per bp", shrink=0.8)
    _save(fig, path)


BEFORE = NEUTRAL          # Portfolio A (rule-based) in comparison charts
AFTER = ASSET             # Portfolio C (optimized)


def frontier_income_vs_capital(frontier: pd.DataFrame, path: Path, base_k: float):
    f = frontier[frontier.status == "optimal"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(f.capital_used, f.net_income, color=ASSET, lw=2, marker="o", ms=5)
    base = f.loc[(f.budget_k - base_k).abs().idxmin()]
    ax.annotate(f"Base K = {base_k:.2f}: income {base.net_income:.3f}",
                (base.capital_used, base.net_income), xytext=(15, -25), textcoords="offset points",
                fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=INK_2))
    _style(ax, "Efficient frontier: net spread income vs modeled C-1 capital used",
           "Capital used (USD millions)", "Net spread income (USD millions per year)")
    _save(fig, path)


def frontier_krd_vs_income(frontier: pd.DataFrame, path: Path):
    f = frontier[frontier.status == "optimal"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(f.net_income, f.max_abs_net_krd_k, color=ASSET, lw=2, marker="o", ms=5,
            label="Max |net key rate DV01| across tenors")
    ax.plot(f.net_income, f.surplus_dv01_k.abs(), color=LIAB, lw=2, marker="o", ms=5,
            label="|Surplus DV01|")
    _style(ax, "Rate mismatch along the frontier", "Net spread income (USD millions per year)",
           "USD thousands per bp")
    ax.legend(frameon=False)
    _save(fig, path)


def compare_net_krd(exp_a: pd.DataFrame, exp_c: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(exp_a))
    w = 0.38
    ax.bar(x - w / 2, exp_a.net_krdv01 * 1000, w, color=BEFORE, label="Portfolio A (rule-based)")
    ax.bar(x + w / 2, exp_c.net_krdv01 * 1000, w, color=AFTER, label="Portfolio C (optimized)")
    ax.axhline(0, color=INK_2, linewidth=1)
    _tolerance_marks(ax, x, exp_c.tolerance * 1000, width=0.9)
    ax.set_xticks(x, [f"{int(k)}Y" for k in exp_a.tenor])
    _style(ax, "Net key rate DV01 before vs after optimization", "Key tenor (dotted: +/- tolerance)",
           "USD thousands per bp")
    ax.legend(frameon=False)
    _save(fig, path)


def compare_funding(gaps_a: pd.DataFrame, gaps_c: pd.DataFrame, path: Path, check_years: int):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axvspan(0.5, check_years + 0.5, color="#f0efec", zorder=0, label=f"Funding test window (years 1-{check_years})")
    ax.plot(gaps_a.t, gaps_a.cum_reinvest, color=BEFORE, lw=2, marker="o", ms=4, label="Portfolio A (rule-based)")
    ax.plot(gaps_c.t, gaps_c.cum_reinvest, color=AFTER, lw=2, marker="o", ms=4, label="Portfolio C (optimized)")
    ax.axhline(0, color=INK_2, linewidth=1)
    _style(ax, "Cumulative funding balance (with reinvestment) before vs after optimization", "Year",
           "USD millions")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, path)


def compare_capital_by_bucket(cap_buckets: pd.DataFrame, path: Path, budget: float):
    charge = cap_buckets["charge"].unstack(0).fillna(0.0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(charge))
    w = 0.38
    labels = list(charge.columns)
    ax.bar(x - w / 2, charge[labels[0]], w, color=BEFORE, label=labels[0])
    ax.bar(x + w / 2, charge[labels[1]], w, color=AFTER, label=labels[1])
    ax.set_xticks(x, charge.index)
    totals = ", ".join(f"{c}: {charge[c].sum():.3f}" for c in labels)
    _style(ax, f"Modeled C-1 capital charge by bucket (budget {budget:.2f}; totals {totals})", "Bucket",
           "USD millions")
    ax.legend(frameon=False)
    _save(fig, path)


PORTFOLIO_COLORS = {"A": NEUTRAL, "B": "#1baf7a", "C": ASSET, "C+swaps": "#4a3aa7"}   # gray, aqua, blue, violet
PORTFOLIO_LABELS = {"A": "A Conservative IG", "B": "B Yield seeking", "C": "C Optimized ALM",
                    "C+swaps": "C with swap overlay"}


def _grouped(ax, categories, series: dict):
    x = np.arange(len(categories))
    w = 0.8 / len(series)
    for i, (name, values) in enumerate(series.items()):
        ax.bar(x - 0.4 + w * (i + 0.5), values, w * 0.92, color=PORTFOLIO_COLORS[name],
               label=PORTFOLIO_LABELS[name])
    ax.set_xticks(x, categories)


def allocation_by_bucket(alloc: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _grouped(ax, list(alloc.index), {c: alloc[c].to_numpy() for c in alloc.columns})
    _style(ax, "Allocation by bucket, three model portfolios", "Bucket", "USD millions (= % of $100M)")
    ax.legend(frameon=False)
    _save(fig, path)


def stress_surplus(stress: pd.DataFrame, path: Path):
    piv = stress.pivot(index="scenario", columns="portfolio", values="dS").loc[stress.scenario.unique()]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _grouped(ax, list(piv.index), {c: piv[c].to_numpy() for c in piv.columns})
    ax.axhline(0, color=INK_2, linewidth=1)
    _style(ax, "Change in economic surplus under MVP stress scenarios (full repricing)", "Scenario",
           "USD millions")
    ax.legend(frameon=False)
    _save(fig, path)


def net_krd_three(summary: pd.DataFrame, keys, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cats = [f"{int(k)}Y" for k in keys]
    _grouped(ax, cats, {c: summary.loc[[f"net_krd_{int(k)}y_k" for k in keys], c].to_numpy(dtype=float)
                        for c in summary.columns})
    ax.axhline(0, color=INK_2, linewidth=1)
    _style(ax, "Net key rate DV01 (assets - liabilities), three model portfolios", "Key tenor",
           "USD thousands per bp")
    ax.legend(frameon=False)
    _save(fig, path)


def robustness_income(draws: pd.DataFrame, base_income: float, path: Path):
    ok = draws[draws.status == "optimal"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(min(ok.income_fixed_C.min(), ok.income_reopt_oos.min()), ok.net_income.max(), 40)
    ax.hist(ok.income_fixed_C, bins=bins, color=NEUTRAL, alpha=0.75, label="Base C held fixed (same draws)")
    ax.hist(ok.net_income, bins=bins, color=ASSET, alpha=0.75, label="Re-optimized, in-sample")
    ax.axvline(base_income, color=INK, lw=1.5, ls="--")
    ax.text(base_income, ax.get_ylim()[1] * 0.95, f" base C {base_income:.3f}", fontsize=8, color=INK, va="top")
    _style(ax, f"Net spread income across {len(ok)} perturbed draws: fixed C vs re-optimized",
           "Net spread income (USD millions per year)", "Number of draws")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, path)


def robustness_positions(positions: pd.DataFrame, path: Path):
    p = positions[positions.pct_draws_held > 0].sort_values("pct_draws_held")
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(p) + 1.6))
    y = np.arange(len(p))
    ax.barh(y + 0.2, p.pct_draws_held * 100, 0.38, color=ASSET, label="Held (> 0)")
    ax.barh(y - 0.2, p.pct_draws_at_issuer_cap.fillna(0) * 100, 0.38, color=LIAB,
            label="At 5% issuer cap (government / agency issuers exempt)")
    ax.set_yticks(y, p.index)
    _style(ax, "How often each position is held / hits the issuer cap", "Percent of draws", "")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path)


def mbs_negative_convexity(grid: pd.DataFrame, path: Path, sec_id: str):
    """grid: shift_bp, price_constant, price_rate_dependent."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(grid.shift_bp, grid.price_constant, color=NEUTRAL, lw=2, marker="o", ms=4, label="Constant CPR")
    ax.plot(grid.shift_bp, grid.price_rate_dependent, color=ASSET, lw=2, marker="o", ms=4,
            label="Rate-dependent CPR (S-curve)")
    _style(ax, f"{sec_id}: price vs parallel rate shift (negative convexity)", "Parallel shift (bp)",
           "Price per 100 face")
    ax.legend(frameon=False)
    _save(fig, path)


def spread_dv01_by_bucket(table: pd.DataFrame, path: Path):
    """table: bucket x portfolio (A, B, C), USD thousands per bp; Treasuries already excluded."""
    order = [b for b in table.index if b != "AGENCY_MBS"] + [b for b in table.index if b == "AGENCY_MBS"]
    t = table.loc[order]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _grouped(ax, [("AGENCY_MBS\n(OAS / basis)" if b == "AGENCY_MBS" else b) for b in t.index],
             {c: t[c].to_numpy() for c in t.columns})
    if "AGENCY_MBS" in order:
        ax.axvline(len(order) - 1.5, color=INK_2, lw=1, ls=":")
    _style(ax, "Spread DV01 by bucket: credit spread vs Agency MBS basis (Treasuries excluded)", "Bucket",
           "USD thousands per bp")
    ax.legend(frameon=False)
    _save(fig, path)


def downgrade_capital(table: pd.DataFrame, path: Path, label: str):
    """table: rows bucket x notches with d_capital."""
    piv = table.pivot(index="bucket", columns="notches", values="d_capital")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(piv))
    ax.bar(x - 0.2, piv[1], 0.38, color="#86b6ef", label="One notch")
    ax.bar(x + 0.2, piv[2], 0.38, color="#1c5cab", label="Two notches")
    ax.set_xticks(x, piv.index)
    _style(ax, f"Capital charge increase from downgrades, {label}", "Bucket downgraded (ALL = every eligible holding)",
           "USD millions")
    ax.legend(frameon=False)
    _save(fig, path)


def stress_all(stress: pd.DataFrame, path: Path):
    piv = stress.pivot(index="scenario", columns="portfolio", values="dS").loc[stress.scenario.unique()[::-1]]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(piv) + 1.8))
    y = np.arange(len(piv))
    h = 0.8 / len(piv.columns)
    for i, c in enumerate(piv.columns):
        ax.barh(y - 0.4 + h * (i + 0.5), piv[c], h * 0.92, color=PORTFOLIO_COLORS.get(c, NEUTRAL),
                label=PORTFOLIO_LABELS.get(c, c))
    ax.set_yticks(y, piv.index)
    ax.axvline(0, color=INK_2, linewidth=1)
    _style(ax, "Change in economic surplus, all stress scenarios", "USD millions", "")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper left")
    _save(fig, path)


def overlay_net_krd(net_c: np.ndarray, net_ov: np.ndarray, keys, tol: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, net_c, 0.38, color=ASSET, label="C (no overlay)")
    ax.bar(x + 0.2, net_ov, 0.38, color="#4a3aa7", label="C with receive-fixed swap overlay")
    ax.axhline(0, color=INK_2, linewidth=1)
    _tolerance_marks(ax, x, tol, width=0.9)
    ax.set_xticks(x, [f"{int(k)}Y" for k in keys])
    _style(ax, "Net key rate DV01: C with vs without swap overlay", "Key tenor (dotted: +/- tolerance)",
           "USD thousands per bp")
    ax.legend(frameon=False)
    _save(fig, path)


def scenario_before_after(table: pd.DataFrame, path: Path):
    piv = table.pivot(index="scenario", columns="portfolio", values="dS").loc[table.scenario.unique()]
    colors = {"C_before": NEUTRAL, "C_after": ASSET, "B": "#1baf7a"}
    labels = {"C_before": "C before scenario constraints", "C_after": "C after scenario constraints",
              "B": "B Yield seeking"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(piv))
    cols = [c for c in ("C_before", "C_after", "B") if c in piv.columns]
    w = 0.8 / len(cols)
    for i, c in enumerate(cols):
        ax.bar(x - 0.4 + w * (i + 0.5), piv[c], w * 0.92, color=colors[c], label=labels[c])
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xticks(x, piv.index)
    _style(ax, "Change in economic surplus: C before vs after scenario constraints", "Scenario", "USD millions")
    ax.legend(frameon=False)
    _save(fig, path)


def downside_limit_sweep(sweep: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(sweep.max_loss_parallel_down, sweep.income_no_overlay, color=ASSET, lw=2, marker="o", ms=5,
            label="Cash bonds only")
    ax.plot(sweep.max_loss_parallel_down, sweep.income_with_overlay, color="#4a3aa7", lw=2, marker="o", ms=5,
            label="With receive-fixed swap overlay")
    ax.invert_xaxis()
    _style(ax, "Cost of tightening the -100bp surplus loss limit", "Allowed surplus loss at -100bp (USD millions)",
           "Net spread income (USD millions per year)")
    ax.legend(frameon=False)
    _save(fig, path)
