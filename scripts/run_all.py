"""Reproduce every table and chart. Run from the repo root: python scripts/run_all.py

Phase 1: curve and single-bond analytics.
Phase 2: investment universe, liability cash flows, waterfall demo.
Phase 3: Module A cash flow matching and Module B KRD exposure (Portfolio A).
Phase 4: Module C capital charges, integrated LP optimization (Portfolio C), efficient frontier.
Phase 5: three model portfolios, MVP stress tests, robustness simulation of the LP solution.
Phase 6: rate-dependent prepayment, spread DV01, downgrade sensitivity, scenarios 5-13,
         tighter capital budget, receive-fixed swap overlay.
All outputs are illustrative and based on synthetic data.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alm import plots  # noqa: E402
from alm.bonds import (convexity, dv01, dv01_fd, effective_convexity, effective_duration,  # noqa: E402
                       fixed_cashflows, fixed_price_fn, macaulay_duration, modified_duration,
                       par_coupon, pv_on_curve, ytm, zero_cashflows)
from alm.capital import (add_capital_factors, bucket_capital, clo_regime_impact,  # noqa: E402
                         ig_to_hy_reallocation)
from alm.cashflow_matching import per_dollar_cashflows  # noqa: E402
from alm.config import OUTPUT_DIR, load_yaml  # noqa: E402
from alm.curve import flat_curve, load_curve, load_key_tenors  # noqa: E402
from alm.krd import key_rate_durations, key_rate_dv01  # noqa: E402
from alm.krd_exposure import krd_exposure_table  # noqa: E402
from alm.liabilities import liability_price_fn, project_liabilities, truncation_diagnostics  # noqa: E402
from alm.pipeline import build_data, portfolio_analysis  # noqa: E402
from alm.optimizer import (budget_shadow_price, efficient_frontier, is_hy,  # noqa: E402
                           krd_tolerances,
                           shadow_cost_breakdown, solve, solve_or_diagnose)
from alm.structured import sequential_waterfall  # noqa: E402
from alm.report import render_file  # noqa: E402
from alm.bonds import effective_convexity as eff_conv, effective_duration as eff_dur  # noqa: E402
from alm.curve import parallel_shift  # noqa: E402
from alm.migration import downgrade_sensitivity  # noqa: E402
from alm.overlay import SwapOverlay  # noqa: E402
from alm.portfolios import (bucket_allocation, load_rule_based, portfolio_summary,  # noqa: E402
                            spread_dv01_by_bucket, spread_dv01_summary)
from alm.universe import load_assumptions, price as sec_price, price_fn as sec_price_fn, solve_coupon  # noqa: E402
from alm.robustness import run_robustness, summarize  # noqa: E402
from alm.stress import stress_table  # noqa: E402
from alm.universe import build_universe, cashflow_matrix, cashflow_table  # noqa: E402

TABLES = OUTPUT_DIR / "tables"
FIGURES = OUTPUT_DIR / "figures"


def curve_table(curve, keys) -> pd.DataFrame:
    t = np.arange(1, 31, dtype=float)
    fwd = [curve.forward_rate(x - 1, x) for x in t]
    return pd.DataFrame({"t": t, "spot": curve.zero(t), "df": curve.discount_factor(t),
                         "fwd_1y": fwd, "is_key_tenor": np.isin(t, keys)})


def bond_analytics(name, times, cfs, curve, keys, spread=0.0) -> dict:
    """All Phase 1 measures for one bond, per 100 face."""
    pf = fixed_price_fn(times, cfs, spread)
    price = pv_on_curve(times, cfs, curve, spread)
    y = ytm(times, cfs, price)
    krd = key_rate_durations(pf, curve, keys)
    krdv01 = key_rate_dv01(pf, curve, keys)
    row = {"bond": name, "price": price, "ytm": y,
           "macaulay": macaulay_duration(times, cfs, y), "modified": modified_duration(times, cfs, y),
           "effective_dur": effective_duration(pf, curve), "convexity": convexity(times, cfs, y),
           "effective_conv": effective_convexity(pf, curve),
           "dv01_analytic": dv01(times, cfs, y), "dv01_fd": dv01_fd(pf, curve),
           "sum_krd": krd.sum(), "sum_krdv01": krdv01.sum()}
    row.update({f"krd_{int(k)}y": v for k, v in zip(keys, krd)})
    return row


def phase1(curve, keys) -> pd.DataFrame:
    rows = [bond_analytics("Textbook 3Y 5% @ flat 4%", *fixed_cashflows(100, 0.05, 3),
                           flat_curve(0.04), keys)]
    for m in (2, 5, 10, 30):
        c = par_coupon(curve, m)
        rows.append(bond_analytics(f"UST {m}Y par ({c:.3%})", *fixed_cashflows(100, c, m), curve, keys))
    rows.append(bond_analytics("UST 30Y STRIPS", *zero_cashflows(100, 30), curve, keys))
    return pd.DataFrame(rows)


def liability_summary(liab, curve, keys) -> pd.DataFrame:
    pf = liability_price_fn(liab)
    t, cf = liab.t.to_numpy(float), liab.total.to_numpy()
    pv = pf(curve)
    y = ytm(t, cf, pv)
    row = {"pv": pv, "irr": y, "macaulay": macaulay_duration(t, cf, y),
           "modified": modified_duration(t, cf, y), "effective_dur": effective_duration(pf, curve),
           "effective_conv": effective_convexity(pf, curve), "dv01": dv01_fd(pf, curve)}
    row.update({f"krdv01_{int(k)}y": v for k, v in zip(keys, key_rate_dv01(pf, curve, keys))})
    return pd.DataFrame([row])


def waterfall_demo() -> pd.DataFrame:
    w = load_yaml("waterfall_demo.yaml")
    rows = []
    for cdr in w["cdr_scenarios"]:
        r = sequential_waterfall(w["collateral"], w["collateral_coupon"], cdr, w["recovery"],
                                 w["repay_rate"], w["maturity"], w["tranches"])
        rows.append(pd.DataFrame({"cdr": cdr, **r}))
    return pd.concat(rows, ignore_index=True)


def phase2(curve, keys):
    universe = build_universe(curve, keys)
    universe.to_csv(TABLES / "universe.csv", index=False)
    cashflow_table(universe, curve).to_csv(TABLES / "universe_cashflows.csv", index=False)
    liab = project_liabilities(curve)
    liab.to_csv(TABLES / "liabilities.csv", index=False)
    liab_sum = liability_summary(liab, curve, keys)
    liab_sum.to_csv(TABLES / "liability_summary.csv", index=False)
    wf = waterfall_demo()
    wf.to_csv(TABLES / "waterfall_demo.csv", index=False)
    metrics = {"liability_pv": float(liab_sum.pv[0]), "liability_macaulay": float(liab_sum.macaulay[0]),
               "liability_effective_duration": float(liab_sum.effective_dur[0]),
               "liability_dv01_usd_k_per_bp": float(liab_sum.dv01[0] * 1000),
               "universe_size": int(len(universe))}

    print("\n=== Phase 2: universe (per 100 face) ===")
    cols = ["id", "naic_designation", "coupon", "price", "ytm", "wal", "effective_dur",
            "spread_dur", "effective_conv", "spread", "el_annual", "net_spread"]
    print(universe[cols].to_string(index=False))
    print("\nLiability cash flows (USD millions):")
    print(liab[["t", "lapse_rate", "benefits", "surrenders", "expenses", "total", "pv"]].to_string(index=False))
    print("\nLiability summary (USD millions; DV01 in USD millions per bp):")
    print(liab_sum.T.to_string(header=False))
    print("\nWaterfall demo (illustrative deal; loss_pct = written down / original size):")
    print(wf[["cdr", "tranche", "size", "principal_paid", "interest_paid", "written_down",
              "loss_pct"]].to_string(index=False))
    return universe, liab, metrics


def analyze_portfolio(holdings, label, curve, keys, universe, liab, per_dollar, policy) -> dict:
    """Module A + Module B + capital for one portfolio; writes tables tagged by label."""
    tag = label.lower().replace(" ", "_")
    r = portfolio_analysis(holdings, curve, keys, universe, liab, per_dollar, policy)
    r["gaps"].to_csv(TABLES / f"cashflow_gaps_{tag}.csv", index=False)
    r["exposure"].to_csv(TABLES / f"krd_exposure_{tag}.csv", index=False)
    r["buckets"].to_csv(TABLES / f"krd_by_bucket_{tag}.csv")
    r["scen"].to_csv(TABLES / f"surplus_scenarios_{tag}.csv", index=False)
    r["cap_rows"].to_csv(TABLES / f"capital_by_security_{tag}.csv")
    return {"label": label, **r}


def print_portfolio(r):
    ex = r["exposure"]
    show = ex.assign(**{c: ex[c] * 1000 for c in ("asset_krdv01", "liability_krdv01", "net_krdv01", "tolerance")})
    print(f"\n--- {r['label']} ---")
    print(r["gaps"][["t", "asset_cf", "liability_cf", "gap", "coverage", "cum_no_reinvest",
                     "cum_reinvest"]].iloc[:12].to_string(index=False))
    print("Cash flow summary:", json.dumps(r["gap_stats"]))
    print("Key rate DV01 (USD thousands per bp) and KRD (years):")
    print(show.to_string(index=False))
    print("Tenors breaching per-tenor tolerance:",
          [f"{int(k)}Y" for k, v, t in zip(ex.tenor, ex.net_krdv01, ex.tolerance) if abs(v) > t + 1e-9])
    print("Duration gap:", {k: round(float(v), 5) for k, v in r["dgap"].items()})
    print("Surplus scenarios (USD millions):")
    print(r["scen"].to_string(index=False))
    print("Capital:", {k: round(float(v), 5) for k, v in r["cap"].items()})


def portfolio_metrics(r) -> dict:
    return {**r["gap_stats"],
            "max_abs_net_krdv01_usd_k": float(r["exposure"].net_krdv01.abs().max() * 1000),
            "duration_gap_years": float(r["dgap"]["duration_gap"]),
            "surplus_dv01_usd_k_per_bp": float(r["dgap"]["surplus_dv01"] * 1000),
            **{f"dS_full_{s.scenario}": float(s.dS_full) for s in r["scen"].itertuples()},
            **{f"dS_krd_{s.scenario}": float(s.dS_krd) for s in r["scen"].itertuples()},
            **{k: float(v) for k, v in r["cap"].items()}}


def phase3(curve, keys, universe, liab, per_dollar, policy):
    """Module A and Module B for the rule-based Portfolio A (pre-optimization)."""
    holdings = pd.Series(load_yaml("portfolios.yaml")["A_conservative_ig"]["holdings"], dtype=float)
    r = analyze_portfolio(holdings, "Portfolio A", curve, keys, universe, liab, per_dollar, policy)
    label, check = r["label"], policy["cash_flow_funding"]["check_years"]
    plots.cashflows_asset_vs_liability(r["gaps"], FIGURES / "a1_cashflows_asset_vs_liability.png", label)
    plots.annual_gap(r["gaps"], FIGURES / "a2_annual_gap.png", label)
    plots.cumulative_funding(r["gaps"], FIGURES / "a3_cumulative_funding.png", label, check)
    plots.krd_asset_vs_liability(r["exposure"], FIGURES / "b1_krd_asset_vs_liability.png", label)
    plots.krd_net_mismatch(r["exposure"], FIGURES / "b2_krd_net_mismatch.png", label)
    plots.krd_bucket_heatmap(r["buckets"], FIGURES / "b3_krd_bucket_heatmap.png", label)
    print("\n=== Phase 3: Module A + B (USD millions) ===")
    print_portfolio(r)
    return r


def phase4(curve, keys, universe, liab, per_dollar, policy, port_a):
    """Capital module, LP optimization (Portfolio C), frontier, before vs after."""
    data = build_data(universe, per_dollar, liab, curve, keys, policy)
    k_base = policy["capital"]["budget"]
    res = solve_or_diagnose(data, policy, k_base)
    if "hard_status" in res:
        print("Hard problem", res["hard_status"], "- conflicting constraints:\n", res["violations"])
    cons = res["constraints"]
    cons.to_csv(TABLES / "optimizer_constraints.csv", index=False)
    x = res["x"][res["x"] > 1e-9]
    x.rename("market_value").to_csv(TABLES / "portfolio_C_holdings.csv")

    port_c = analyze_portfolio(x, "Portfolio C", curve, keys, universe, liab, per_dollar, policy)
    cfg = policy["capital"]["frontier"]
    budgets = np.round(np.arange(cfg["min"], cfg["max"] + 1e-9, cfg["step"]), 2)
    frontier = efficient_frontier(data, policy, budgets)
    frontier.to_csv(TABLES / "efficient_frontier.csv", index=False)
    regime = pd.concat([clo_regime_impact(p["holdings"], universe).assign(portfolio=p["label"])
                        for p in (port_a, port_c)])
    regime.to_csv(TABLES / "clo_regime_impact.csv", index=False)

    # (b) re-optimize with CLOs on the old corporate bond factors
    old_u = add_capital_factors(universe, "old_bond")
    data_old = build_data(old_u, per_dollar, liab, curve, keys, policy, regime="old_bond")
    res_old = solve(data_old, policy, k_base)
    x_old = res_old["x"]
    shift = pd.DataFrame({"new_2026": res["x"], "old_bond": x_old}).fillna(0.0)
    shift["change"] = shift.old_bond - shift.new_2026
    shift = shift[(shift.new_2026 > 1e-9) | (shift.old_bond > 1e-9)]
    shift.to_csv(TABLES / "clo_regime_reoptimized_holdings.csv")
    regime_reopt = pd.DataFrame([
        {"case": "(a) C holdings, new 2026 CLO factors", "net_income": res["objective"],
         "capital_used": float(data.cap_factor @ res["x"].to_numpy())},
        {"case": "(a) C holdings repriced, old bond factors", "net_income": res["objective"],
         "capital_used": float(data_old.cap_factor @ res["x"].to_numpy())},
        {"case": "(b) re-optimized, old bond factors", "net_income": res_old["objective"],
         "capital_used": float(data_old.cap_factor @ x_old.to_numpy())}])
    regime_reopt.to_csv(TABLES / "clo_regime_reoptimization.csv", index=False)

    # why the LP does not swap more IG BBB into HY: price the trade at the duals
    swap = shadow_cost_breakdown(data, policy, res, {"HY_BB+_5Y": 1.0, "IG_BBB_10Y": -1.0})
    swap.to_csv(TABLES / "swap_bbb_to_bbplus_shadow_costs.csv", index=False)
    realloc = ig_to_hy_reallocation(universe, "IG_BBB_10Y")
    realloc.to_csv(TABLES / "ig_to_hy_reallocation.csv", index=False)
    cap_buckets = pd.concat({p["label"]: bucket_capital(p["cap_rows"]) for p in (port_a, port_c)})
    cap_buckets.to_csv(TABLES / "capital_by_bucket.csv")

    plots.frontier_income_vs_capital(frontier, FIGURES / "c1_frontier_income_vs_capital.png", k_base)
    plots.frontier_krd_vs_income(frontier, FIGURES / "c2_frontier_krd_vs_income.png")
    plots.compare_net_krd(port_a["exposure"], port_c["exposure"], FIGURES / "c3_net_krd_before_after.png")
    plots.compare_funding(port_a["gaps"], port_c["gaps"], FIGURES / "c4_funding_before_after.png",
                          policy["cash_flow_funding"]["check_years"])
    plots.compare_capital_by_bucket(cap_buckets, FIGURES / "c5_capital_by_bucket.png", k_base)

    print("\n=== Phase 4: capital + optimization ===")
    print("Universe capital factors and efficiency:")
    print(universe[["id", "naic_designation", "cap_factor", "net_spread", "capital_efficiency"]].to_string(index=False))
    print(f"\nLP status: {res['status']}; net spread income = {res['objective']:.5f} USD millions per year")
    print("Portfolio C holdings (USD millions):")
    print(x.round(4).to_string())
    print("\nConstraint table (KRD rows in USD thousands per bp):")
    print(cons.to_string(index=False))
    print_portfolio(port_c)
    print("\nCapital by bucket:")
    print(cap_buckets.to_string())
    print("\nCLO factor regime impact:")
    print(regime.to_string(index=False))
    print("\nCLO regime: (a) repriced vs (b) re-optimized:")
    print(regime_reopt.to_string(index=False))
    print("Allocation shift (old - new):")
    print(shift.round(4).to_string())
    print("\nShadow cost of swapping USD 1M IG_BBB_10Y -> HY_BB+_5Y (USD millions per year):")
    print(swap.to_string(index=False))
    print(f"income gain {swap.attrs['income_gain']:.6f}, reduced cost {swap.attrs['reduced_cost']:.2e}")
    print("\nIG -> HY reallocation of USD 1M out of IG_BBB_10Y:")
    print(realloc.to_string(index=False))
    print("\nEfficient frontier:")
    print(frontier.to_string(index=False))

    shadow = budget_shadow_price(data, res)
    return port_c, data, {"lp_status": res["status"], "net_spread_income": res["objective"],
                    "shadow_price_capital": shadow,
                    "binding_constraints": cons[cons.binding].constraint.tolist(),
                    "frontier_income_at_min_budget": float(frontier.net_income.iloc[0]),
                    "frontier_income_at_max_budget": float(frontier.net_income.iloc[-1]),
                    "portfolio_C": portfolio_metrics(port_c),
                    "clo_regime": {"C_capital_new_2026": float(regime_reopt.capital_used[0]),
                                   "C_capital_old_bond_same_holdings": float(regime_reopt.capital_used[1]),
                                   "reoptimized_old_bond_income": float(regime_reopt.net_income[2]),
                                   "reoptimized_old_bond_capital": float(regime_reopt.capital_used[2])},
                    "swap_bbb_to_bbplus": {"amount": 1.0, "income_gain": swap.attrs["income_gain"],
                                           "capital_change": float(data.cap_factor[data.ids.index("HY_BB+_5Y")]
                                                                   - data.cap_factor[data.ids.index("IG_BBB_10Y")]),
                                           "reduced_cost": swap.attrs["reduced_cost"],
                                           "hy_max_dual": float(cons.set_index("constraint").loc["hy_max", "dual"]),
                                           **{f"shadow_cost_{r.constraint}": float(r.shadow_cost)
                                              for r in swap.itertuples()}},
                    "frontier_shadow_at_min_budget": float(frontier.shadow_price_capital.iloc[0]),
                    "frontier_capital_at_max_budget": float(frontier.capital_used.iloc[-1]),
                    "clo_bb_new_2026": float(shift.loc["CLO_BB", "new_2026"]),
                    "clo_bb_old_bond": float(shift.loc["CLO_BB", "old_bond"])}


def phase5(curve, keys, universe, liab, per_dollar, policy, port_c, data):
    """Three portfolios side by side, MVP stress tests, robustness of the LP corner solution."""
    rb = load_rule_based()
    portfolios = {"A": rb["A_conservative_ig"], "B": rb["B_yield_seeking"], "C": port_c["holdings"]}
    summary = pd.DataFrame({k: portfolio_summary(h, universe, liab, curve, keys, per_dollar, policy)
                            for k, h in portfolios.items()})
    summary.to_csv(TABLES / "portfolio_comparison.csv")
    alloc = bucket_allocation(portfolios, universe)
    alloc.to_csv(TABLES / "portfolio_allocation_by_bucket.csv")

    scen_cfg = load_yaml("scenarios.yaml")["stress_scenarios"]
    stress = stress_table(portfolios, universe, liab, curve, keys, scen_cfg, policy)
    stress.to_csv(TABLES / "stress_mvp.csv", index=False)
    notes = pd.DataFrame({"scenario": list(scen_cfg), "pm_note": [v["pm_note"] for v in scen_cfg.values()]})
    notes.to_csv(TABLES / "stress_pm_notes.csv", index=False)

    rcfg = load_yaml("robustness.yaml")
    draws = run_robustness(data, universe, policy, rcfg, port_c["holdings"])
    draws.to_csv(TABLES / "robustness_draws.csv", index=False)
    rob = summarize(draws, data, universe, policy, port_c["holdings"])
    rob["headline"].to_csv(TABLES / "robustness_headline.csv")
    rob["buckets"].to_csv(TABLES / "robustness_bucket_weights.csv")
    rob["positions"].to_csv(TABLES / "robustness_positions.csv")

    plots.allocation_by_bucket(alloc, FIGURES / "d1_allocation_by_bucket.png")
    plots.stress_surplus(stress, FIGURES / "d2_stress_surplus.png")
    plots.net_krd_three(summary, keys, FIGURES / "d3_net_krd_three_portfolios.png")
    plots.robustness_income(draws, port_c["cap"]["net_spread_income"], FIGURES / "e1_robustness_income.png")
    plots.robustness_positions(rob["positions"], FIGURES / "e2_robustness_positions.png")

    print("\n=== Phase 5: model portfolios ===")
    print(summary.to_string())
    print("\nAllocation by bucket (USD millions):")
    print(alloc.to_string())
    print("\nMVP stress tests (USD millions; DV01 in USD thousands per bp):")
    print(stress.to_string(index=False))
    print(f"\nRobustness: {rob['n_ok']}/{rob['n_draws']} draws optimal")
    print(rob["headline"].to_string())
    print("Optimizer's curse:", rob["curse"])
    print(rob["buckets"].to_string())
    print(rob["positions"][rob["positions"].pct_draws_held > 0].to_string())

    def clean(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        return float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v

    m = {"portfolios": {k: {kk: clean(vv) for kk, vv in summary[k].items()} for k in summary.columns},
         "stress": {f"{r.portfolio}_{r.scenario}_dS": float(r.dS) for r in stress.itertuples()},
         "robustness": {"n_draws": rob["n_draws"], "n_optimal": rob["n_ok"], **rob["curse"],
                        **{f"bucket_{b}_{c}": float(rob["buckets"].loc[b, c]) for b in rob["buckets"].index
                           for c in ("p5", "p95")},
                        **{f"{row}_{c}": float(v) for row in rob["headline"].index
                           for c, v in rob["headline"].loc[row].items()}}}
    return m


def mbs_prepayment_comparison(curve, keys, policy):
    """Agency MBS analytics under constant vs rate-dependent CPR, plus a price-vs-shift grid."""
    rows, grids = [], {}
    for model in ("constant", "rate_dependent"):
        a = load_assumptions(prepay_model=model)
        for sec in a[a.bucket == "AGENCY_MBS"].itertuples(index=False):
            cpn = solve_coupon(sec, curve)
            pf = sec_price_fn(sec, cpn, curve)
            rows.append({"id": sec.id, "model": model, "effective_dur": eff_dur(pf, curve),
                         "effective_conv": eff_conv(pf, curve), "effective_conv_100bp": eff_conv(pf, curve, 0.01)})
            shifts = np.arange(-200, 201, 25)
            grids.setdefault(sec.id, {"shift_bp": shifts})[f"price_{model}"] = [
                pf(parallel_shift(curve, b * 1e-4)) for b in shifts]
    table = pd.DataFrame(rows).pivot(index="id", columns="model")
    table.columns = [f"{m}_{c}" for c, m in table.columns]
    return table, {k: pd.DataFrame(v) for k, v in grids.items()}


def phase6(curve, keys, universe, liab, per_dollar, policy, port_c, data):
    rb = load_rule_based()
    portfolios = {"A": rb["A_conservative_ig"], "B": rb["B_yield_seeking"], "C": port_c["holdings"]}
    k_base = policy["capital"]["budget"]

    # 1. rate-dependent prepayment
    mbs, grids = mbs_prepayment_comparison(curve, keys, policy)
    mbs.to_csv(TABLES / "mbs_prepayment_comparison.csv")
    u_const = add_capital_factors(build_universe(curve, keys, load_assumptions(prepay_model="constant")),
                                  policy["capital"]["clo_regime"])
    d_const = build_data(u_const, per_dollar, liab, curve, keys, policy)
    res_const = solve(d_const, policy, k_base)
    agency = lambda x: float(x[[i for i in x.index if i in set(universe.id[universe.bucket == "AGENCY_MBS"])]].sum())

    # 2. spread DV01 by bucket
    sdv = pd.DataFrame({k: spread_dv01_by_bucket(h, universe) for k, h in portfolios.items()}).fillna(0.0)
    sdv_sum = spread_dv01_summary(sdv)
    pd.concat([sdv, sdv_sum]).to_csv(TABLES / "spread_dv01_by_bucket.csv")

    # 3. downgrade sensitivity
    down = pd.concat([downgrade_sensitivity(h, universe, policy["limits"]["hy_max"]).assign(portfolio=k)
                      for k, h in portfolios.items()])
    down.to_csv(TABLES / "downgrade_sensitivity.csv", index=False)

    # 4. swap overlay
    ov_cfg = load_yaml("overlay.yaml")
    d_ov = build_data(universe, per_dollar, liab, curve, keys, policy, overlay=True)
    res_ov = solve(d_ov, policy, k_base)
    x_ov_all = res_ov["x"]
    swaps = {int(i.split("_")[-1][:-1]): float(v) for i, v in x_ov_all.items() if i.startswith("SWAP_") and v > 1e-9}
    x_ov = x_ov_all[[i for i in x_ov_all.index if not i.startswith("SWAP_") and x_ov_all[i] > 1e-9]]
    overlay = SwapOverlay(swaps, curve)
    res_ov["constraints"].to_csv(TABLES / "optimizer_constraints_overlay.csv", index=False)
    x_ov_all[x_ov_all > 1e-9].rename("market_value_or_notional").to_csv(TABLES / "portfolio_C_overlay_holdings.csv")
    net_c = data.krd.T @ port_c["holdings"].reindex(data.ids).fillna(0).to_numpy() - data.liab_krd
    net_ov = d_ov.krd.T @ x_ov_all.reindex(d_ov.ids).fillna(0).to_numpy() - d_ov.liab_krd
    rr = policy["rate_risk"]
    tol = krd_tolerances(data.liab_krd, rr["krd_floor"], rr["krd_pct_of_liability"])

    # 5. all stress scenarios (MVP + Phase 6) for A, B, C and C with overlay
    scen = {**load_yaml("scenarios.yaml")["stress_scenarios"], **load_yaml("scenarios.yaml")["stress_scenarios_phase6"]}
    stress = stress_table({**portfolios, "C+swaps": x_ov}, universe, liab, curve, keys, scen, policy,
                          overlays={"C+swaps": overlay})
    stress.to_csv(TABLES / "stress_all.csv", index=False)
    notes = pd.DataFrame({"scenario": list(scen), "pm_note": [v["pm_note"] for v in scen.values()]})
    notes.to_csv(TABLES / "stress_pm_notes_all.csv", index=False)

    # 6. tighter capital budget (scenario 12): re-optimize
    mult = load_yaml("scenarios.yaml")["capital_budget_stress"]["tighter_capital_25pct"]["budget_mult"]
    res_tight = solve(data, policy, k_base * mult)
    tight = pd.DataFrame({"base": port_c["holdings"], "tight": res_tight["x"]}).fillna(0.0)
    tight = tight[(tight.base > 1e-9) | (tight.tight > 1e-9)].assign(change=lambda t: t.tight - t.base)
    tight.to_csv(TABLES / "tighter_capital_reoptimization.csv")

    alloc_ov = bucket_allocation({"C": port_c["holdings"], "C+swaps": x_ov}, universe)
    alloc_ov.to_csv(TABLES / "overlay_allocation_by_bucket.csv")

    plots.mbs_negative_convexity(grids["FNMA_30Y"], FIGURES / "f1_mbs_negative_convexity.png", "FNMA_30Y")
    plots.spread_dv01_by_bucket(sdv, FIGURES / "f2_spread_dv01_by_bucket.png")
    plots.downgrade_capital(down[down.portfolio == "C"], FIGURES / "f3_downgrade_capital_C.png", "Portfolio C")
    plots.stress_all(stress, FIGURES / "f4_stress_all_scenarios.png")
    plots.overlay_net_krd(net_c, net_ov, keys, tol, FIGURES / "f5_overlay_net_krd.png")

    print("\n=== Phase 6 ===")
    print("Agency MBS: constant vs rate-dependent CPR")
    print(mbs.to_string())
    print(f"C re-optimized under constant CPR: income {res_const['objective']:.5f}, Agency MBS {agency(res_const['x']):.3f}"
          f" vs rate-dependent: income {data.net_spread @ port_c['holdings'].reindex(data.ids).fillna(0).to_numpy():.5f},"
          f" Agency MBS {agency(port_c['holdings']):.3f}")
    print("\nSpread DV01 by bucket (USD thousands per bp; Treasuries excluded):")
    print(pd.concat([sdv, sdv_sum]).to_string())
    print("\nDowngrade sensitivity:")
    print(down.to_string(index=False))
    print("\nAll stress scenarios:")
    print(stress.to_string(index=False))
    print(f"\nTighter capital (K x {mult}): income {res_tight['objective']:.5f}")
    print(tight.to_string())
    print(f"\nSwap overlay: income {res_ov['objective']:.5f} vs {data.net_spread @ port_c['holdings'].reindex(data.ids).fillna(0).to_numpy():.5f}; swaps {swaps}")
    print(alloc_ov.to_string())
    print("net KRD C:", net_c.round(2), "\nnet KRD C+swaps:", net_ov.round(2))

    def st(p, sc, col="dS"):
        return float(stress[(stress.portfolio == p) & (stress.scenario == sc)][col].iloc[0])

    c_income = float(data.net_spread @ port_c["holdings"].reindex(data.ids).fillna(0).to_numpy())
    return {
        "mbs": {f"{i}_{c}": float(v) for i in mbs.index for c, v in mbs.loc[i].items()},
        "prepay_C_income_constant_cpr": float(res_const["objective"]),
        "prepay_C_agency_constant_cpr": agency(res_const["x"]),
        "prepay_C_income_rate_dependent": c_income,
        "prepay_C_agency_rate_dependent": agency(port_c["holdings"]),
        "spread_dv01": {r: {k: float(sdv_sum.loc[r, k]) for k in sdv_sum.columns} for r in sdv_sum.index},
        "downgrade": {f"{r.portfolio}_{r.bucket}_{r.notches}n_{c}": float(getattr(r, c))
                      for r in down.itertuples() for c in ("d_capital", "d_net_income", "hy_after")},
        "stress": {f"{r.portfolio.replace('+', '_')}_{r.scenario}_{c}": float(getattr(r, c))
                   for r in stress.itertuples()
                   for c in ("dS", "capital_charge", "d_net_income", "hy_after", "min_funding_y1_to_check")},
        "hy_breach": {f"{r.portfolio.replace('+', '_')}_{r.scenario}": bool(r.hy_limit_breached)
                      for r in stress.itertuples()},
        "tight_capital": {"budget": k_base * mult, "income": float(res_tight["objective"]),
                          "income_change": float(res_tight["objective"] - c_income),
                          "C_utilization_at_tight_budget": float(port_c["cap"]["capital_charge"] / (k_base * mult)),
                          "hy_after": float(res_tight["x"][is_hy(data.designation)].sum())},
        "overlay": {"income": float(res_ov["objective"]), "income_change": float(res_ov["objective"] - c_income),
                    "carry_bp": ov_cfg["carry_bp_per_year"], "max_notional": ov_cfg["max_total_notional"],
                    **{f"notional_{t}y": swaps.get(t, 0.0) for t in ov_cfg["tenors"]},
                    "total_notional": float(sum(swaps.values())),
                    "treasury_C": float(alloc_ov.loc["TSY", "C"]) if "TSY" in alloc_ov.index else 0.0,
                    "treasury_C_swaps": float(alloc_ov.loc["TSY", "C+swaps"]) if "TSY" in alloc_ov.index else 0.0,
                    "max_abs_net_krd_C": float(np.abs(net_c).max()), "max_abs_net_krd_C_swaps": float(np.abs(net_ov).max()),
                    "surplus_dv01_C_swaps": float(net_ov.sum()),
                    "dual_parallel_down_C_swaps": float(res_ov["constraints"].set_index("constraint")
                                                        .loc["scenario_parallel_down_100", "dual"]),
                    **{f"dS_{sc}_{p.replace('+', '_')}": st(p, sc) for p in ("C", "C+swaps")
                       for sc in ("parallel_down_100", "parallel_up_100", "steepener", "flattener", "combined_adverse")}},
    }


def scenario_constraint_analysis(curve, keys, universe, liab, per_dollar, policy, port_c):
    """C before vs after the scenario constraints / downgrade buffer, their costs, and the -100bp limit sweep."""
    import copy
    k = policy["capital"]["budget"]
    variants = {"before": (False, False, None), "scenario_only": (True, False, None), "after": (None, None, None),
                "hy_after_downgrade_on": (None, None, True)}
    res, data = {}, {}
    for name, (so, bo, ho) in variants.items():
        data[name] = build_data(universe, per_dollar, liab, curve, keys, policy, scenario_on=so, buffer_on=bo,
                                hy_after_on=ho)
        res[name] = solve_or_diagnose(data[name], policy, k)
    x_before = res["before"]["x"][res["before"]["x"] > 1e-9]
    x_after = port_c["holdings"]
    b = load_rule_based()["B_yield_seeking"]

    all_scen = {**load_yaml("scenarios.yaml")["stress_scenarios"], **load_yaml("scenarios.yaml")["stress_scenarios_phase6"]}
    names = ["parallel_up_100", "parallel_down_100", "steepener", "flattener", "combined_adverse"]
    table = stress_table({"C_before": x_before, "C_after": x_after, "B": b}, universe, liab, curve, keys,
                         {n: all_scen[n] for n in names}, policy)
    table.to_csv(TABLES / "scenario_constraints_before_after.csv", index=False)

    cons = res["after"]["constraints"]
    sc_rows = cons[cons.group.isin(["scenario", "downgrade_buffer", "capital", "surplus_dv01"])]
    sc_rows.to_csv(TABLES / "scenario_constraint_rows.csv", index=False)
    d_after = data["after"]
    c_down_before = float(d_after.extra["cap_factor_downgraded"] @ x_before.reindex(d_after.ids).fillna(0).to_numpy())
    c_down_after = float(d_after.extra["cap_factor_downgraded"] @ x_after.reindex(d_after.ids).fillna(0).to_numpy())

    def net_krd(x, d):
        return d.krd.T @ x.reindex(d.ids).fillna(0).to_numpy() - d.liab_krd

    # -100bp loss-limit sweep, with and without the swap overlay
    sweep = []
    for lmax in policy["scenario_constraints"].get("sweep_parallel_down", [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]):
        pol = copy.deepcopy(policy)
        base_loss = pol["scenario_constraints"]["max_loss"]
        pol["scenario_constraints"]["max_loss"] = {n: (lmax if n == "parallel_down_100" else
                                                       (base_loss[n] if isinstance(base_loss, dict) else base_loss))
                                                   for n in pol["scenario_constraints"]["scenarios"]}
        row = {"max_loss_parallel_down": lmax}
        for ov in (False, True):
            r = solve_or_diagnose(build_data(universe, per_dollar, liab, curve, keys, pol, overlay=ov), pol, k)
            tag = "with_overlay" if ov else "no_overlay"
            row[f"income_{tag}"] = r["objective"] if "hard_status" not in r else np.nan
            row[f"status_{tag}"] = r.get("hard_status", r["status"])
        sweep.append(row)
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(TABLES / "downside_limit_sweep.csv", index=False)

    plots.scenario_before_after(table, FIGURES / "g1_scenario_constraints_before_after.png")
    plots.downside_limit_sweep(sweep, FIGURES / "g2_downside_limit_sweep.png")

    print("\n=== Scenario constraints and downgrade buffer ===")
    print(f"income before {res['before']['objective']:.5f} | scenario rows only {res['scenario_only']['objective']:.5f}"
          f" | + downgrade buffer {res['after']['objective']:.5f}")
    print(sc_rows.to_string(index=False))
    print(table[["portfolio", "scenario", "dA", "dL", "dS", "capital_charge", "hy_after"]].to_string(index=False))
    print(f"capital after one-notch BBB downgrade: before {c_down_before:.4f}, after {c_down_after:.4f}")
    print(sweep.to_string(index=False))

    def ds(p, sc):
        return float(table[(table.portfolio == p) & (table.scenario == sc)].dS.iloc[0])

    binding = sc_rows[(sc_rows.group == "scenario") & sc_rows.binding]
    return {"income_before": float(res["before"]["objective"]),
            "income_scenario_only": float(res["scenario_only"]["objective"]),
            "income_after": float(res["after"]["objective"]),
            "scenario_cost": float(res["before"]["objective"] - res["scenario_only"]["objective"]),
            "buffer_cost": float(res["scenario_only"]["objective"] - res["after"]["objective"]),
            "hy_after_downgrade_constraint_enabled": bool(policy["downgrade_buffer"]["hy_limit_after_downgrade"]["enabled"]),
            "income_with_hy_after_downgrade_limit": float(res["hy_after_downgrade_on"]["objective"]),
            "hy_after_downgrade_limit_cost": float(res["after"]["objective"] - res["hy_after_downgrade_on"]["objective"]),
            "hy_after_downgrade_with_limit": float(data["hy_after_downgrade_on"].extra["hy_after_downgrade"]
                                                   @ res["hy_after_downgrade_on"]["x"].to_numpy()),
            "max_loss": policy["scenario_constraints"]["max_loss"],
            "buffer_multiple": policy["downgrade_buffer"]["multiple"],
            "binding_scenarios": binding.constraint.str.removeprefix("scenario_").tolist(),
            **{f"dual_{r.constraint}": float(r.dual) for r in sc_rows.itertuples()},
            "capital_downgraded_before": c_down_before, "capital_downgraded_after": c_down_after,
            "capital_used_before": float(data["before"].cap_factor @ res["before"]["x"].to_numpy()),
            "capital_used_after": float(data["after"].cap_factor @ res["after"]["x"].to_numpy()),
            "surplus_dv01_before": float(net_krd(x_before, data["before"]).sum()),
            "surplus_dv01_after": float(net_krd(x_after, data["after"]).sum()),
            "max_abs_net_krd_before": float(np.abs(net_krd(x_before, data["before"])).max()),
            "max_abs_net_krd_after": float(np.abs(net_krd(x_after, data["after"])).max()),
            **{f"dS_{sc}_{p}": ds(p, sc) for p in ("C_before", "C_after", "B") for sc in names},
            "sweep": {f"{r.max_loss_parallel_down:.2f}".replace(".", "_"): {
                "no_overlay": None if np.isnan(r.income_no_overlay) else float(r.income_no_overlay),
                "with_overlay": None if np.isnan(r.income_with_overlay) else float(r.income_with_overlay)}
                for r in sweep.itertuples()}}


def prepayment_sensitivity(curve, keys, universe, liab, policy, portfolios: dict) -> pd.DataFrame:
    """Surplus DV01 and +/-100bp surplus change under rate-dependent (default) vs constant CPR."""
    u_const = add_capital_factors(build_universe(curve, keys, load_assumptions(prepay_model="constant")),
                                  policy["capital"]["clo_regime"])
    scen = {n: {"curve": load_yaml("scenarios.yaml")["rate_scenarios"][n]}
            for n in ("parallel_up_100", "parallel_down_100")}
    lpf = liability_price_fn(liab)
    rows = []
    for model, u in (("rate_dependent", universe), ("constant_cpr", u_const)):
        for pname, h in portfolios.items():
            exp = krd_exposure_table(h, u, lpf, curve, keys)
            st = stress_table({pname: h}, u, liab, curve, keys, scen, policy).set_index("scenario")
            rows.append({"portfolio": pname, "prepay_model": model,
                         "surplus_dv01_k": float(exp.net_krdv01.sum() * 1000),
                         "dS_up_100": float(st.loc["parallel_up_100", "dS"]),
                         "dS_down_100": float(st.loc["parallel_down_100", "dS"])})
    out = pd.DataFrame(rows).pivot(index="portfolio", columns="prepay_model")
    out.columns = [f"{c}_{m}" for c, m in out.columns]
    return out


TEMPLATES = Path(__file__).resolve().parents[1] / "docs" / "templates"
DOCS = {"README.md.tmpl": Path(__file__).resolve().parents[1] / "README.md",
        "investment_committee_summary.md.tmpl":
            Path(__file__).resolve().parents[1] / "docs" / "investment_committee_summary.md",
        "methodology.md.tmpl": Path(__file__).resolve().parents[1] / "docs" / "methodology.md",
        "interview_notes.md.tmpl": Path(__file__).resolve().parents[1] / "docs" / "interview_notes.md"}


def report_inputs(policy, liab) -> dict:
    """Configured inputs that the docs quote (so no number is hand-typed)."""
    from alm.capital import load_bond_factors, load_clo_factors
    bond, clo = load_bond_factors(), load_clo_factors()
    return {"total_assets": policy["total_assets"], "liability_pv_target": float(liab.pv.sum()),
            "capital_budget": policy["capital"]["budget"], "frontier": policy["capital"]["frontier"],
            "limits": {k: v for k, v in policy["limits"].items() if k != "single_issuer_exempt"},
            "rate_risk": policy["rate_risk"], "funding": policy["cash_flow_funding"],
            "robustness": load_yaml("robustness.yaml"),
            "scenario_max_loss": policy["scenario_constraints"]["max_loss"],
            "downgrade_buffer_multiple": policy["downgrade_buffer"]["multiple"],
            "scenarios": {
                "lapse_increase": load_yaml("scenarios.yaml")["stress_scenarios_phase6"]["surrenders_up_50pct_y1_5"]["lapse"]["mult"] - 1.0,
                "tight_budget_mult": load_yaml("scenarios.yaml")["capital_budget_stress"]["tighter_capital_25pct"]["budget_mult"]},
            "factors": {"bond_2B": float(bond["2.B"]), "bond_3B": float(bond["3.B"]),
                        "clo_2B": float(clo.loc["2.B", "factor"]), "clo_3B": float(clo.loc["3.B", "factor"]),
                        "thin_tranche_surcharge": float(load_yaml("clo_thin_tranche.yaml")["surcharge"]),
                        "residual": float(clo.loc["RESIDUAL", "factor"])}}


def provenance() -> dict:
    """When and from which commit these results were generated ('dirty' = uncommitted changes present)."""
    import datetime
    import subprocess
    root = Path(__file__).resolve().parents[1]
    git = lambda *args: subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    try:
        head = git("rev-parse", "--short", "HEAD")
        status = git("status", "--porcelain", "--", ".", ":(exclude)outputs", ":(exclude)README.md", ":(exclude)docs")
        commit, dirty = (head.stdout.strip() or "unknown"), bool(status.stdout.strip())
    except OSError:
        commit, dirty = "unknown", False
    return {"generated": datetime.date.today().isoformat(), "commit": commit, "dirty": dirty}


def count_tests() -> int:
    """Number of collected pytest tests (quoted in interview notes)."""
    import subprocess
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1]).stdout
    return sum(1 for line in out.splitlines() if "::" in line)


def binding_capital(m: dict) -> dict:
    """Which capital row binds in the base-case LP (from the solver's binding flags), as doc text."""
    rows = set(m["phase4"]["binding_constraints"])
    cap, buf = "capital" in rows, "downgrade_buffer" in rows
    buffer_text = f"the post-downgrade buffer ({m['inputs']['downgrade_buffer_multiple']:.2f} \u00d7 K)"
    if buf and not cap:
        text, short = f"{buffer_text}, not the base C-1 charge", "the downgrade buffer"
    elif cap and not buf:
        text, short = "the base C-1 charge (\u2264 K), not the post-downgrade buffer", "the capital row"
    elif cap and buf:
        text, short = f"both the base C-1 charge and {buffer_text}", "the capital row and the downgrade buffer"
    else:
        text, short = "neither: capital is slack at the base budget", "no capital row"
    return {"capital_row_binding": cap, "downgrade_buffer_binding": buf, "text": text, "short": short,
            "buffer_status": "binding" if buf else "not binding"}


def derived_report(m: dict) -> dict:
    """Differences and ratios quoted in the docs, computed once here."""
    p = m["phase5"]["portfolios"]
    gross = {k: p[k]["net_spread_income"] + p[k]["expected_credit_loss"] for k in p}
    a_dur = p["A"]["effective_duration"]
    sweep = m["phase6"]["constraints"]["sweep"]
    keys = sorted(sweep, key=lambda k: float(k.replace("_", ".")))
    lo, hi = keys[-1], keys[0]
    return {"liability_duration": (a_dur - p["A"]["duration_gap"]) / (m["inputs"]["liability_pv_target"]
                                                                       / m["inputs"]["total_assets"]),
            "L_over_A": m["inputs"]["liability_pv_target"] / m["inputs"]["total_assets"],
            "downgrade_buffer_limit": m["inputs"]["downgrade_buffer_multiple"] * m["inputs"]["capital_budget"],
            "sweep": {"loosest_limit": float(lo.replace("_", ".")), "tightest_limit": float(hi.replace("_", ".")),
                      "loosest_no_overlay": sweep[lo]["no_overlay"], "loosest_with_overlay": sweep[lo]["with_overlay"],
                      "tightest_no_overlay": sweep[hi]["no_overlay"], "tightest_with_overlay": sweep[hi]["with_overlay"]},
            "test_count": count_tests(),
            "binding_capital": binding_capital(m),
            "C_minus_A_income": p["C"]["net_spread_income"] - p["A"]["net_spread_income"],
            "C_over_A_income": p["C"]["net_spread_income"] / p["A"]["net_spread_income"],
            "B_minus_C_income_k": 1000 * (p["B"]["net_spread_income"] - p["C"]["net_spread_income"]),
            "B_minus_C_gross_spread_k": 1000 * (gross["B"] - gross["C"]),
            "B_minus_C_el_k": 1000 * (p["B"]["expected_credit_loss"] - p["C"]["expected_credit_loss"]),
            "B_capital_over_budget": p["B"]["budget_utilization"] - 1.0,
            "C_minus_A_steepener_dS": m["phase5"]["stress"]["C_steepener_dS"] - m["phase5"]["stress"]["A_steepener_dS"]}


def headline(m: dict) -> dict:
    """Single place for the project's headline numbers (resume / interview material reads from here)."""
    p, c6 = m["phase5"]["portfolios"], m["phase6"]["constraints"]
    rob, ov = m["phase5"]["robustness"], m["phase6"]["overlay"]
    return {
        "income_A": p["A"]["net_spread_income"], "income_C": p["C"]["net_spread_income"],
        "income_C_minus_A": p["C"]["net_spread_income"] - p["A"]["net_spread_income"],
        "capital_budget": m["inputs"]["capital_budget"], "capital_used_C": p["C"]["capital_charge"],
        "shadow_price_capital_budget": m["phase4"]["shadow_price_capital"],
        "max_abs_net_krd_k_A": p["A"]["max_abs_net_krd_k"], "max_abs_net_krd_k_C": p["C"]["max_abs_net_krd_k"],
        "krd_tenors_breaching_A": p["A"]["krd_tenors_breaching"], "krd_tenors_breaching_C": p["C"]["krd_tenors_breaching"],
        "dS_down_100_C_before_scenario_rows": c6["dS_parallel_down_100_C_before"],
        "dS_down_100_C_after_scenario_rows": c6["dS_parallel_down_100_C_after"],
        "dS_combined_adverse_C_before": c6["dS_combined_adverse_C_before"],
        "dS_combined_adverse_C_after": c6["dS_combined_adverse_C_after"],
        "dS_combined_adverse_B": c6["dS_combined_adverse_B"],
        "scenario_rows_income_cost": c6["scenario_cost"], "downgrade_buffer_income_cost": c6["buffer_cost"],
        "clo_capital_C_new_2026": m["phase4"]["clo_regime"]["C_capital_new_2026"],
        "clo_capital_C_old_bond": m["phase4"]["clo_regime"]["C_capital_old_bond_same_holdings"],
        "clo_reoptimized_old_bond_income": m["phase4"]["clo_regime"]["reoptimized_old_bond_income"],
        "robustness_reopt_in_sample": rob["income_reopt_in_sample_mean"],
        "robustness_fixed_C": rob["income_fixed_C_mean"],
        "robustness_reopt_out_of_sample": rob["income_reopt_out_of_sample_mean"],
        "optimizers_curse_gap": rob["optimizers_curse_gap"],
        "overlay_income_gain": ov["income_change"],
        "cost_to_zero_down_100_loss_no_overlay": (m["report"]["sweep"]["loosest_no_overlay"]
                                                  - m["report"]["sweep"]["tightest_no_overlay"]),
        "cost_to_zero_down_100_loss_with_overlay": (m["report"]["sweep"]["loosest_with_overlay"]
                                                    - m["report"]["sweep"]["tightest_with_overlay"]),
    }


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    curve, keys = load_curve(), load_key_tenors()

    curve_df = curve_table(curve, keys)
    curve_df.to_csv(TABLES / "phase1_curve.csv", index=False)
    bonds_df = phase1(curve, keys)
    bonds_df.to_csv(TABLES / "phase1_bond_analytics.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.precision", 4,
                  "display.max_rows", 200)
    print("Base spot curve (synthetic):")
    print(curve_df[curve_df.is_key_tenor].drop(columns="is_key_tenor").to_string(index=False))
    print("\nBond analytics (per 100 face; durations in years):")
    print(bonds_df.iloc[:, :12].to_string(index=False))
    print("\nKey rate durations (years):")
    print(bonds_df[["bond"] + [c for c in bonds_df if c.startswith("krd_")]].to_string(index=False))

    policy = load_yaml("policy.yaml")
    universe, liab, m2 = phase2(curve, keys)
    universe = add_capital_factors(universe, policy["capital"]["clo_regime"])
    universe.to_csv(TABLES / "universe.csv", index=False)
    per_dollar = per_dollar_cashflows(cashflow_matrix(cashflow_table(universe, curve), "expected"),
                                      universe.set_index("id")["price"])
    port_a = phase3(curve, keys, universe, liab, per_dollar, policy)
    port_c, data, m4 = phase4(curve, keys, universe, liab, per_dollar, policy, port_a)
    m5 = phase5(curve, keys, universe, liab, per_dollar, policy, port_c, data)
    m6 = phase6(curve, keys, universe, liab, per_dollar, policy, port_c, data)
    m6["constraints"] = scenario_constraint_analysis(curve, keys, universe, liab, per_dollar, policy, port_c)
    rb = load_rule_based()
    sens = prepayment_sensitivity(curve, keys, universe, liab, policy,
                                  {"A": rb["A_conservative_ig"], "B": rb["B_yield_seeking"], "C": port_c["holdings"]})
    sens.to_csv(TABLES / "prepayment_sensitivity.csv")
    print("\nPrepayment sensitivity (rate-dependent vs constant CPR):")
    print(sens.to_string())
    m6["prepay_sensitivity"] = {f"{p}_{c}": float(sens.loc[p, c]) for p in sens.index for c in sens.columns}

    m2["liability_truncation"] = truncation_diagnostics(curve)
    metrics = {"provenance": provenance(), "inputs": report_inputs(policy, liab),
               "phase2": m2, "phase3": {"portfolio_A": portfolio_metrics(port_a)}, "phase4": m4, "phase5": m5, "phase6": m6}
    metrics["report"] = derived_report(metrics)
    metrics["headline"] = headline(metrics)
    metrics = json.loads(json.dumps(metrics, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    for name, out in DOCS.items():
        render_file(TEMPLATES / name, out, metrics)
    print(f"\nTables written to {TABLES}; figures to {FIGURES}; metrics to {OUTPUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
