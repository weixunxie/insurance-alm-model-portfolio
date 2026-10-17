"""Thin wrapper: dashboard controls -> existing LP (alm.optimizer) -> plain results dict.

No finance logic lives here. LP data for each toggle combination is built once with
alm.pipeline.build_data and cached; each interaction only re-solves the LP (HiGHS).
The -100bp loss slider changes only the right-hand side of that one scenario row.
"""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import cvxpy as cp
import numpy as np
import pandas as pd

from alm.capital import add_capital_factors
from alm.optimizer import (ProblemData, budget_shadow_price, efficient_frontier, krd_tolerances,
                           solve_or_diagnose)
from alm.pipeline import build_data
from dashboard.data import base

DOWN_SCENARIO = "parallel_down_100"


@dataclass(frozen=True)
class Controls:
    budget_k: float           # capital budget K, USD millions
    scenario_on: bool         # full-repricing scenario surplus rows
    buffer_on: bool           # downgrade capital buffer
    overlay_on: bool          # receive-fixed swap overlay
    max_loss_down: float      # allowed surplus loss at -100bp, USD millions
    regime: str               # 'new_2026' (CLO factors) or 'old_bond'
    hy_after_on: bool = False # NAIC 3+ holdings after a one-notch downgrade <= HY limit + allowance

    def rounded(self) -> "Controls":
        return replace(self, budget_k=round(float(self.budget_k), 2),
                       max_loss_down=round(float(self.max_loss_down), 2))


def base_controls() -> Controls:
    p = base().policy
    sc = p["scenario_constraints"]
    max_loss = sc["max_loss"][DOWN_SCENARIO] if isinstance(sc["max_loss"], dict) else sc["max_loss"]
    hy_after = p["downgrade_buffer"].get("hy_limit_after_downgrade", {}).get("enabled", False)
    return Controls(p["capital"]["budget"], sc["enabled"], p["downgrade_buffer"]["enabled"], False,
                    float(max_loss), p["capital"]["clo_regime"], hy_after)


@lru_cache(maxsize=2)
def _universe(regime: str) -> pd.DataFrame:
    b = base()
    return b.universe if regime == b.policy["capital"]["clo_regime"] else add_capital_factors(b.universe, regime)


@lru_cache(maxsize=32)
def problem_data(regime: str, overlay_on: bool, scenario_on: bool, buffer_on: bool,
                 hy_after_on: bool = False) -> ProblemData:
    b = base()
    return build_data(_universe(regime), b.per_dollar, b.liab, b.curve, b.keys, b.policy, regime=regime,
                      overlay=overlay_on, scenario_on=scenario_on, buffer_on=buffer_on, hy_after_on=hy_after_on)


def _with_down_limit(d: ProblemData, max_loss_down: float) -> ProblemData:
    names = d.extra.get("scenario_names", [])
    if DOWN_SCENARIO not in names:
        return d
    losses = [max_loss_down if n == DOWN_SCENARIO else v for n, v in zip(names, d.extra["scenario_max_loss"])]
    return replace(d, extra={**d.extra, "scenario_max_loss": losses})


def _data(c: Controls) -> ProblemData:
    return _with_down_limit(problem_data(c.regime, c.overlay_on, c.scenario_on, c.buffer_on, c.hy_after_on),
                            c.max_loss_down)


def _summarize(d: ProblemData, res: dict, c: Controls, solve_ms: float) -> dict:
    policy = base().policy
    x = res["x"].to_numpy()
    rr = policy["rate_risk"]
    net_krd = d.krd.T @ x - d.liab_krd                    # USD thousands per bp
    overlay = d.extra.get("is_overlay", np.zeros(len(d.ids), bool))
    alloc = pd.Series(x[~overlay], index=d.bucket[~overlay]).groupby(level=0).sum()
    cons = res["constraints"]
    return {
        "status": res["status"],
        "feasible": "hard_status" not in res,
        "solve_ms": solve_ms,
        "net_income": float(res["objective"]),
        "capital_used": float(d.cap_factor @ x),
        "budget_k": c.budget_k,
        "shadow_price": budget_shadow_price(d, res) if "hard_status" not in res else float("nan"),
        "surplus_dv01_k": float(net_krd.sum()),
        "surplus_dv01_max": float(rr["surplus_dv01_max"]),
        "tenors": [int(k) for k in d.keys],
        "net_krd_k": net_krd,
        "tolerance_k": krd_tolerances(d.liab_krd, rr["krd_floor"], rr["krd_pct_of_liability"]),
        "allocation": alloc,
        "swaps": pd.Series(x[overlay], index=np.array(d.ids)[overlay]),
        "holdings": res["x"][res["x"] > 1e-9],
        "binding": cons[cons.binding][["constraint", "group", "sense", "value", "limit", "dual", "dual_units"]],
        "violations": res.get("violations"),
        "hard_status": res.get("hard_status"),
    }


@lru_cache(maxsize=256)
def _solve_cached(c: Controls) -> dict:
    d = _data(c)
    t0 = time.perf_counter()
    res = solve_or_diagnose(d, base().policy, c.budget_k)
    solve_ms = (time.perf_counter() - t0) * 1000.0
    if res.get("x") is None:
        return {"status": res["status"], "feasible": False, "solve_ms": solve_ms, "hard_status": res.get("hard_status"),
                "violations": None, "budget_k": c.budget_k}
    return _summarize(d, res, c, solve_ms)


def solve_controls(c: Controls) -> dict:
    """LP result for the given controls (cached on rounded control values)."""
    return _solve_cached(c.rounded())


@lru_cache(maxsize=32)
def frontier(regime: str, overlay_on: bool, scenario_on: bool, buffer_on: bool, max_loss_down: float,
             hy_after_on: bool = False) -> pd.DataFrame:
    """Efficient frontier over the policy's K grid for the given toggles (optimal points only)."""
    policy = base().policy
    cfg = policy["capital"]["frontier"]
    budgets = np.round(np.arange(cfg["min"], cfg["max"] + 1e-9, cfg["step"]), 2)
    d = _with_down_limit(problem_data(regime, overlay_on, scenario_on, buffer_on, hy_after_on),
                         round(max_loss_down, 2))
    f = efficient_frontier(d, policy, budgets)
    return f[f.status == cp.OPTIMAL].reset_index(drop=True)


def frontier_for(c: Controls) -> pd.DataFrame:
    c = c.rounded()
    return frontier(c.regime, c.overlay_on, c.scenario_on, c.buffer_on, c.max_loss_down, c.hy_after_on)


def hy_after_cost(c: Controls) -> float | None:
    """Income given up by the NAIC 3+ after-downgrade limit: same controls with it off minus with it on."""
    if not c.hy_after_on:
        return None
    on, off = solve_controls(c), solve_controls(replace(c, hy_after_on=False))
    if not (on["feasible"] and off["feasible"]):
        return None
    return off["net_income"] - on["net_income"]


def warm() -> None:
    """Pre-build the base-case LP data, solution and frontier."""
    c = base_controls()
    solve_controls(c)
    frontier_for(c)
