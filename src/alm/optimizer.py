"""Integrated ALM optimization (LP, CVXPY).

Decision variables: x_i >= 0, USD millions of market value in security i.
Objective: maximize expected net spread income  sum_i x_i * (s_i - EL_i)   (USD millions per year).

Every constraint is stored as one linear row  a . x (<= or ==) b  so that the
post-solve table (value, limit, slack, binding, dual) is computed the same way
for all of them. Units: KRD rows in USD thousands per bp; everything else in
USD millions.

Dual values: for a '<=' row, the dual is the gain in objective per unit
increase of b. For the capital row this is the shadow price of capital:
extra net spread income (USD per year) per extra USD 1 of capital budget.
"""

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd

from alm.cashflow_matching import funding_matrix


@dataclass
class Row:
    name: str
    group: str
    a: np.ndarray
    b: float
    sense: str = "<="        # '<=' or '=='
    offset: float = 0.0      # reporting only: displayed value = -a.x + offset for '>=' rows stored negated


@dataclass
class ProblemData:
    ids: list
    net_spread: np.ndarray          # s - EL per USD of MV
    cap_factor: np.ndarray          # capital charge per USD of MV
    krd: np.ndarray                 # (n, K) key rate DV01, USD thousands per bp per USD 1M MV
    liab_krd: np.ndarray            # (K,) liability key rate DV01, USD thousands per bp
    cf: np.ndarray                  # (T, n) expected cash flow per USD 1 MV
    liab_cf: np.ndarray             # (T,) liability outflows, USD millions
    fwd: np.ndarray                 # (T,) 1Y forward rates for reinvestment
    keys: np.ndarray
    bucket: np.ndarray
    designation: np.ndarray
    issuer: np.ndarray
    extra: dict = field(default_factory=dict)


def build_problem_data(universe: pd.DataFrame, per_dollar_cf: pd.DataFrame, liab_cf, liab_krdv01,
                       fwd, keys) -> ProblemData:
    """universe must contain cap_factor; liab_krdv01 in USD millions per bp (converted to $k)."""
    u = universe.reset_index(drop=True)
    krd_cols = [f"krdv01_{int(k)}y" for k in keys]
    krd_per_musd = u[krd_cols].to_numpy() / u["price"].to_numpy()[:, None] * 1000.0   # $k/bp per $1M
    return ProblemData(
        ids=list(u.id), net_spread=u.net_spread.to_numpy(), cap_factor=u.cap_factor.to_numpy(),
        krd=krd_per_musd, liab_krd=np.asarray(liab_krdv01) * 1000.0,
        cf=per_dollar_cf[u.id].to_numpy(), liab_cf=np.asarray(liab_cf, dtype=float),
        fwd=np.asarray(fwd), keys=np.asarray(keys), bucket=u.bucket.to_numpy(),
        designation=u.naic_designation.to_numpy(), issuer=u.issuer.to_numpy())


def is_hy(designation) -> np.ndarray:
    """NAIC 3 or below."""
    return np.array([str(d)[0] in "3456" for d in designation])


def krd_tolerances(liab_krd_k, floor: float, pct: float) -> np.ndarray:
    """Per-tenor tolerance (USD thousands per bp): max(floor, pct * |liability key rate DV01|)."""
    return np.maximum(floor, pct * np.abs(np.asarray(liab_krd_k)))


def build_rows(d: ProblemData, policy: dict, budget_k: float) -> list[Row]:
    lim, rr, fund = policy["limits"], policy["rate_risk"], policy["cash_flow_funding"]
    n = len(d.ids)
    overlay = d.extra.get("is_overlay", np.zeros(n, bool))
    rows = [Row("budget", "budget", (~overlay).astype(float), policy["total_assets"], "=="),
            Row("capital", "capital", d.cap_factor, budget_k)]
    if overlay.any():
        rows.append(Row("swap_notional_max", "overlay", overlay.astype(float), d.extra["max_total_notional"]))

    tol = krd_tolerances(d.liab_krd, rr["krd_floor"], rr["krd_pct_of_liability"])
    for k, tenor in enumerate(d.keys):
        rows.append(Row(f"krd_{int(tenor)}y_upper", "krd", d.krd[:, k], d.liab_krd[k] + tol[k]))
        rows.append(Row(f"krd_{int(tenor)}y_lower", "krd", -d.krd[:, k], -(d.liab_krd[k] - tol[k])))
    total_a, total_l = d.krd.sum(axis=1), d.liab_krd.sum()
    rows.append(Row("surplus_dv01_upper", "surplus_dv01", total_a, total_l + rr["surplus_dv01_max"]))
    rows.append(Row("surplus_dv01_lower", "surplus_dv01", -total_a, -(total_l - rr["surplus_dv01_max"])))

    # funding: C = M (CF x - L) >= 0 for years 1..check_years  ->  -(M CF) x <= -(M L)
    m = funding_matrix(d.fwd)
    for t in range(fund["check_years"]):
        rows.append(Row(f"funding_y{t + 1}", "funding", -(m[t] @ d.cf), -(m[t] @ d.liab_cf)))

    ind = lambda mask: mask.astype(float)
    rows.append(Row("hy_max", "bucket", ind(is_hy(d.designation)), lim["hy_max"]))
    rows.append(Row("structured_max", "bucket", ind(np.isin(d.bucket, policy["structured_buckets"])),
                    lim["structured_max"]))
    rows.append(Row("agency_mbs_max", "bucket", ind(d.bucket == "AGENCY_MBS"), lim["agency_mbs_max"]))
    rows.append(Row("clo_max", "bucket", ind(d.bucket == "CLO"), lim["clo_max"]))
    for issuer in sorted(set(d.issuer) - set(lim["single_issuer_exempt"]) - {"SWAP_OVERLAY"}):
        rows.append(Row(f"issuer_{issuer}", "issuer", ind(d.issuer == issuer), lim["single_issuer_max"]))
    rows.append(Row("liquidity_min", "liquidity", -ind(np.isin(d.bucket, ["TSY", "AGENCY_MBS"])),
                    -lim["liquidity_min"]))

    # scenario surplus constraints: dA_s(x) - dL_s >= -max_loss_s  ->  -dP_s . x <= max_loss_s - dL_s
    for name, dp, dl, max_loss in zip(d.extra.get("scenario_names", []), d.extra.get("scenario_dp", []),
                                      d.extra.get("scenario_dl", []), d.extra.get("scenario_max_loss", [])):
        rows.append(Row(f"scenario_{name}", "scenario", -dp, max_loss - dl, offset=-dl))

    # downgrade buffer: capital after the downgrade rule <= multiple * K
    if "cap_factor_downgraded" in d.extra:
        rows.append(Row("downgrade_buffer", "downgrade_buffer", d.extra["cap_factor_downgraded"],
                        d.extra["downgrade_buffer_multiple"] * budget_k))
    if "hy_after_downgrade" in d.extra:
        rows.append(Row("hy_after_downgrade", "bucket", d.extra["hy_after_downgrade"],
                        d.extra["hy_after_downgrade_limit"]))
    return rows


DUAL_UNITS = {
    "budget": "USD income/yr per USD 1 of extra assets",
    "capital": "USD income/yr per USD 1 of capital budget",
    "krd": "USD M income/yr per USD 1k/bp of tolerance",
    "surplus_dv01": "USD M income/yr per USD 1k/bp of tolerance",
    "funding": "USD income/yr per USD 1 of required funding balance",
    "bucket": "USD income/yr per USD 1 of limit (MV)",
    "issuer": "USD income/yr per USD 1 of limit (MV)",
    "liquidity": "USD income/yr per USD 1 of limit (MV)",
    "overlay": "USD income/yr per USD 1 of swap notional limit",
    "scenario": "USD income/yr per USD 1 of allowed scenario surplus loss",
    "downgrade_buffer": "USD income/yr per USD 1 of post-downgrade capital limit",
}


def _display(row: Row, ax: float) -> tuple[str, float, float]:
    """Sense, value and limit in natural (un-negated) form for reporting."""
    if row.name.endswith("_lower") or row.group in ("funding", "liquidity", "scenario"):
        return ">=", -ax + row.offset, -row.b + row.offset
    return row.sense, ax, row.b


def solve(d: ProblemData, policy: dict, budget_k: float, soft: bool = False) -> dict:
    """Solve the LP. With soft=True every '<=' row gets a penalized slack s >= 0."""
    rows = build_rows(d, policy, budget_k)
    le = [r for r in rows if r.sense == "<="]
    eq = [r for r in rows if r.sense == "=="]
    a_le, b_le = np.array([r.a for r in le]), np.array([r.b for r in le])
    a_eq, b_eq = np.array([r.a for r in eq]), np.array([r.b for r in eq])

    x = cp.Variable(len(d.ids), nonneg=True)
    income = d.net_spread @ x
    if soft:
        s = cp.Variable(len(le), nonneg=True)
        objective = cp.Maximize(income - policy["solver"]["soft_penalty"] * cp.sum(s))
        cons_le = a_le @ x <= b_le + s
    else:
        objective = cp.Maximize(income)
        cons_le = a_le @ x <= b_le
    cons_eq = a_eq @ x == b_eq
    prob = cp.Problem(objective, [cons_le, cons_eq])
    prob.solve(solver=policy["solver"]["name"])

    out = {"status": prob.status, "budget_k": budget_k, "soft": soft}
    if x.value is None:
        return out
    xv = np.maximum(x.value, 0.0)
    tol = policy["solver"]["binding_tol"]
    table = []
    for r, dual in zip(le + eq, np.concatenate([cons_le.dual_value, cons_eq.dual_value])):
        ax = float(r.a @ xv)
        sense, value, limit = _display(r, ax)
        slack = r.b - ax
        table.append({"constraint": r.name, "group": r.group, "sense": sense, "value": value,
                      "limit": limit, "slack": slack,
                      "binding": bool(abs(slack) <= tol * max(1.0, abs(r.b))), "dual": float(dual),
                      "dual_units": DUAL_UNITS[r.group]})
    out.update({"x": pd.Series(xv, index=d.ids), "objective": float(d.net_spread @ xv),
                "constraints": pd.DataFrame(table)})
    if soft:
        out["violations"] = pd.DataFrame({"constraint": [r.name for r in le], "slack_used": s.value})
        out["violations"] = out["violations"][out["violations"].slack_used > tol]
    return out


def shadow_cost_breakdown(d: ProblemData, policy: dict, res: dict, direction: dict) -> pd.DataFrame:
    """Price a trade direction dx (e.g. {+1 in A, -1 in B}, USD millions) at the optimal duals.

    shadow_cost_r = dual_r * (a_r . dx). At an LP optimum, income gain - sum of shadow costs
    is the trade's reduced cost: <= 0 means the trade cannot improve the objective.
    Rows with zero dual (slack constraints) cost nothing at the margin and are omitted.
    """
    dx = np.zeros(len(d.ids))
    for sec, amount in direction.items():
        dx[d.ids.index(sec)] = amount
    cons = res["constraints"].set_index("constraint")
    out = []
    for r in build_rows(d, policy, res["budget_k"]):
        change, dual = float(r.a @ dx), float(cons.loc[r.name, "dual"])
        if abs(dual) > 1e-12 and abs(change) > 1e-12:
            out.append({"constraint": r.name, "row_change": change, "dual": dual, "shadow_cost": dual * change})
    table = pd.DataFrame(out)
    table.attrs["income_gain"] = float(d.net_spread @ dx)
    table.attrs["reduced_cost"] = table.attrs["income_gain"] - float(table.shadow_cost.sum())
    return table


def solve_or_diagnose(d: ProblemData, policy: dict, budget_k: float) -> dict:
    """Hard solve; if infeasible, re-solve with penalized slacks and report which rows conflict."""
    res = solve(d, policy, budget_k)
    if res["status"] == cp.OPTIMAL:
        return res
    soft = solve(d, policy, budget_k, soft=True)
    soft["hard_status"] = res["status"]
    return soft


def budget_shadow_price(d: ProblemData, res: dict) -> float:
    """d(objective)/dK: every row whose limit scales with K (capital, and the downgrade buffer at multiple x K)."""
    cons = res["constraints"].set_index("constraint")
    out = float(cons.loc["capital", "dual"])
    if "downgrade_buffer" in cons.index:
        out += d.extra["downgrade_buffer_multiple"] * float(cons.loc["downgrade_buffer", "dual"])
    return out


def efficient_frontier(d: ProblemData, policy: dict, budgets) -> pd.DataFrame:
    """Re-solve over capital budgets K; record income, capital used, KRD mismatch, shadow price."""
    rows = []
    for k in budgets:
        res = solve(d, policy, float(k))
        if res["status"] != cp.OPTIMAL:
            rows.append({"budget_k": k, "status": res["status"]})
            continue
        x = res["x"].to_numpy()
        net_krd = d.krd.T @ x - d.liab_krd
        cap_row = res["constraints"].set_index("constraint").loc["capital"]
        rows.append({"budget_k": k, "status": res["status"], "net_income": res["objective"],
                     "capital_used": float(d.cap_factor @ x), "shadow_price_capital": budget_shadow_price(d, res),
                     "capital_binding": cap_row.binding, "max_abs_net_krd_k": float(np.abs(net_krd).max()),
                     "surplus_dv01_k": float(net_krd.sum()), "hy": float(x[is_hy(d.designation)].sum())})
    return pd.DataFrame(rows)
