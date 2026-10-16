"""Module A: cash flow projection and matching.

Holdings are market values in USD millions. A position of x_i USD millions in a
security priced P_i per 100 face holds x_i / P_i * 100 face, so its cash flows
are x_i * CF_i,t / P_i where CF is per 100 face.

Funding rules (config/policy.yaml):
    C_0 = beginning cash (0)
    C_t = C_{t-1} * (1 + r_t) + gap_t,  gap_t = asset inflow_t - liability outflow_t
    r_t = 1Y forward f(t-1, t) if C_{t-1} >= 0 (reinvest), f + borrow_spread if C_{t-1} < 0 (borrow)
"Without reinvestment" sets r_t = 0, i.e. C_t = cumulative sum of gaps.
"""

import numpy as np
import pandas as pd

from alm.curve import SpotCurve


def forward_rates(curve: SpotCurve, horizon: int) -> np.ndarray:
    """f_t = annually compounded forward rate for year t (from t-1 to t), t = 1..horizon."""
    return np.array([curve.forward_rate(t - 1, t) for t in range(1, horizon + 1)])


def per_dollar_cashflows(cf_matrix: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """Cash flow per USD 1 of market value: CF per 100 face / price."""
    return cf_matrix.div(prices[cf_matrix.columns], axis=1)


def asset_cashflows(holdings: pd.Series, per_dollar: pd.DataFrame) -> pd.Series:
    """Portfolio asset inflows by year (USD millions)."""
    return per_dollar[holdings.index] @ holdings


def funding_balance(gaps, fwd=None, borrow_spread: float = 0.0, beginning_cash: float = 0.0) -> np.ndarray:
    """Cumulative funding balance C_t. fwd=None means no reinvestment (plain cumulative sum)."""
    gaps = np.asarray(gaps, dtype=float)
    if fwd is None:
        return beginning_cash + np.cumsum(gaps)
    c, out = beginning_cash, np.empty(len(gaps))
    for t, gap in enumerate(gaps):
        rate = fwd[t] if c >= 0 else fwd[t] + borrow_spread
        c = c * (1.0 + rate) + gap
        out[t] = c
    return out


def funding_matrix(fwd) -> np.ndarray:
    """M with C = M @ gap for the reinvestment-only recursion (no borrowing, C_0 = 0).

    M[t, s] = prod_{u=s+1..t} (1 + f_u) for s <= t. Exact whenever C stays >= 0,
    which is what the optimizer's funding constraint C >= 0 enforces; it keeps
    the constraint linear in holdings.
    """
    n = len(fwd)
    growth = np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(fwd))])   # growth[t] = prod_{u<=t}(1+f_u)
    m = np.zeros((n, n))
    for t in range(n):
        m[t, : t + 1] = growth[t + 1] / growth[1 : t + 2]
    return m


def gap_table(asset_cf, liability_cf, curve: SpotCurve, borrow_spread: float,
              beginning_cash: float = 0.0) -> pd.DataFrame:
    """Annual gap analysis, years 1..N."""
    a = np.asarray(asset_cf, dtype=float)
    l = np.asarray(liability_cf, dtype=float)
    n = len(a)
    fwd = forward_rates(curve, n)
    gap = a - l
    return pd.DataFrame({
        "t": np.arange(1, n + 1), "asset_cf": a, "liability_cf": l, "gap": gap,
        "coverage": np.divide(a, l, out=np.full(n, np.nan), where=l > 0),
        "fwd_1y": fwd,
        "cum_no_reinvest": funding_balance(gap, None, beginning_cash=beginning_cash),
        "cum_reinvest": funding_balance(gap, fwd, borrow_spread, beginning_cash),
        "pv_gap": gap * curve.discount_factor(np.arange(1, n + 1, dtype=float))})


def first_shortfall_year(balance) -> int | None:
    idx = np.flatnonzero(np.asarray(balance) < 0)
    return int(idx[0] + 1) if idx.size else None


def gap_summary(table: pd.DataFrame, check_years: int) -> dict:
    """Headline cash flow matching metrics."""
    out = {"pv_gaps": float(table.pv_gap.sum()),
           "min_coverage_1_to_check": float(table.coverage.iloc[:check_years].min())}
    for col in ("cum_no_reinvest", "cum_reinvest"):
        bal = table[col].to_numpy()
        out[f"first_shortfall_year_{col}"] = first_shortfall_year(bal)
        out[f"max_cum_shortfall_{col}"] = float(max(0.0, -bal.min()))
        out[f"funding_ok_years_1_to_{check_years}_{col}"] = bool(np.all(bal[:check_years] >= 0))
    return out
