"""Robustness of the LP corner solution to the illustrative spread / PD inputs.

Each Monte Carlo draw perturbs the objective coefficients only:
    s_i'  = s_i * max(0, 1 + spread_sd * Z1_i)
    PD_i' = PD_i * exp(pd_sigma * Z2_i - pd_sigma^2 / 2)          (mean-one lognormal)
    net spread_i' = s_i' - PD_i' * (1 - R_i)
then re-solves the LP at the base capital budget. Prices, KRDs and expected cash
flows stay at base values (simplification), so every draw faces the same constraints.

Three incomes per draw (optimizer's curse diagnostics):
  income_reopt      re-optimized portfolio x_k evaluated on its own draw k (in-sample)
  income_fixed_C    base-case Portfolio C, held fixed, evaluated on draw k
  income_reopt_oos  x_k evaluated on an independent fresh draw (out-of-sample)
Because each draw's inputs are noisy around the base case, max_x (ns_k . x) is
biased upward; the gap mean(income_reopt) - mean(income_fixed_C) measures that bias.
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from alm.optimizer import ProblemData, solve


def perturbed_net_spread(spread, pd_, recovery, rng, spread_sd: float, pd_sigma: float) -> np.ndarray:
    s = spread * np.maximum(0.0, 1.0 + spread_sd * rng.standard_normal(len(spread)))
    p = pd_ * np.exp(pd_sigma * rng.standard_normal(len(pd_)) - 0.5 * pd_sigma ** 2)
    return s - p * (1.0 - recovery)


def run_robustness(d: ProblemData, universe: pd.DataFrame, policy: dict, cfg: dict,
                   base_x: pd.Series) -> pd.DataFrame:
    """One row per draw: status, the three incomes, capital used and holdings (columns = security ids)."""
    u = universe.set_index("id").loc[d.ids]
    rng = np.random.default_rng(cfg["seed"])
    rng_oos = np.random.default_rng(cfg["seed"] + 1)            # independent stream for out-of-sample draws
    args = (u.spread.to_numpy(), u.pd.to_numpy(), u.recovery.to_numpy())
    xc = base_x.reindex(d.ids).fillna(0.0).to_numpy()
    k = policy["capital"]["budget"]
    rows = []
    for i in range(cfg["n_draws"]):
        ns = perturbed_net_spread(*args, rng, cfg["spread_sd"], cfg["pd_sigma"])
        ns_fresh = perturbed_net_spread(*args, rng_oos, cfg["spread_sd"], cfg["pd_sigma"])
        res = solve(replace(d, net_spread=ns), policy, k)
        x = res["x"].to_numpy() if "x" in res else np.full(len(d.ids), np.nan)
        rows.append({"draw": i, "status": res["status"], "net_income": res.get("objective", np.nan),
                     "income_fixed_C": float(ns @ xc), "income_reopt_oos": float(ns_fresh @ x),
                     "capital_used": float(d.cap_factor @ x), **dict(zip(d.ids, x))})
    return pd.DataFrame(rows)


def summarize(draws: pd.DataFrame, d: ProblemData, universe: pd.DataFrame, policy: dict, base_x: pd.Series):
    """Distribution of income / capital / bucket weights and per-position issuer-cap frequency."""
    ok = draws[draws.status == "optimal"]
    x = ok[d.ids]
    q = [0.05, 0.25, 0.50, 0.75, 0.95]
    cols = {"income_reopt_in_sample": ok.net_income, "income_fixed_C": ok.income_fixed_C,
            "income_reopt_out_of_sample": ok.income_reopt_oos, "capital_used": ok.capital_used}
    headline = pd.DataFrame({k: v.quantile(q) for k, v in cols.items()}).T
    headline.columns = [f"p{int(p * 100)}" for p in q]
    headline["mean"] = [v.mean() for v in cols.values()]
    curse = {"optimizers_curse_gap": float(ok.net_income.mean() - ok.income_fixed_C.mean()),
             "oos_minus_fixed_C": float(ok.income_reopt_oos.mean() - ok.income_fixed_C.mean()),
             "pct_draws_reopt_beats_fixed_C_in_sample": float((ok.net_income > ok.income_fixed_C).mean())}

    buckets = universe.set_index("id").loc[d.ids, "bucket"]
    bw = x.T.groupby(buckets).sum().T
    bucket_stats = pd.DataFrame({"base_C": base_x.groupby(buckets.loc[base_x.index]).sum(),
                                 "mean": bw.mean(), "p5": bw.quantile(0.05), "p95": bw.quantile(0.95)}).fillna(0.0)

    lim = policy["limits"]
    capped = ~np.isin(d.issuer, lim["single_issuer_exempt"])
    cap = lim["single_issuer_max"]
    positions = pd.DataFrame({
        "base_C": base_x.reindex(d.ids).fillna(0.0).to_numpy(),
        "mean": x.mean().to_numpy(),
        "pct_draws_held": (x > 1e-6).mean().to_numpy(),
        "pct_draws_at_issuer_cap": np.where(capped, (x >= cap - 1e-6).mean().to_numpy(), np.nan)},
        index=d.ids)
    return {"n_ok": int(len(ok)), "n_draws": int(len(draws)), "headline": headline, "curse": curse,
            "buckets": bucket_stats, "positions": positions}
