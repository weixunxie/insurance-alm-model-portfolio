"""Model portfolios A (conservative IG), B (yield seeking), C (optimized) and their side-by-side metrics."""

import numpy as np
import pandas as pd

from alm.capital import capital_report
from alm.cashflow_matching import asset_cashflows, gap_summary, gap_table
from alm.config import load_yaml
from alm.krd_exposure import duration_gap, krd_exposure_table
from alm.liabilities import liability_price_fn
from alm.optimizer import is_hy, krd_tolerances


def load_rule_based() -> dict:
    """Rule-based portfolios from config/portfolios.yaml as {name: holdings Series}."""
    cfg = load_yaml("portfolios.yaml")
    return {name: pd.Series(p["holdings"], dtype=float) for name, p in cfg.items()}


def policy_checks(holdings: pd.Series, universe: pd.DataFrame, policy: dict) -> dict:
    """Bucket / issuer / liquidity usage (USD millions) and whether each limit holds."""
    u = universe.set_index("id").loc[holdings.index]
    lim = policy["limits"]
    hy = float(holdings[is_hy(u.naic_designation)].sum())
    structured = float(holdings[u.bucket.isin(policy["structured_buckets"])].sum())
    agency = float(holdings[u.bucket == "AGENCY_MBS"].sum())
    clo = float(holdings[u.bucket == "CLO"].sum())
    liquid = float(holdings[u.bucket.isin(["TSY", "AGENCY_MBS"])].sum())
    by_issuer = holdings.groupby(u.issuer).sum().drop(lim["single_issuer_exempt"], errors="ignore")
    max_issuer = float(by_issuer.max()) if len(by_issuer) else 0.0
    eps = 1e-6
    ok = (hy <= lim["hy_max"] + eps and structured <= lim["structured_max"] + eps
          and agency <= lim["agency_mbs_max"] + eps and clo <= lim["clo_max"] + eps
          and max_issuer <= lim["single_issuer_max"] + eps and liquid >= lim["liquidity_min"] - eps)
    return {"hy": hy, "structured_ex_agency": structured, "agency_mbs": agency, "clo": clo,
            "treasury_plus_agency": liquid, "max_single_issuer": max_issuer, "bucket_limits_ok": ok}


def portfolio_summary(holdings, universe, liab, curve, keys, per_dollar, policy) -> dict:
    """Section 9 metrics for one portfolio (USD millions unless noted)."""
    u = universe.set_index("id").loc[holdings.index]
    lpf = liability_price_fn(liab)
    funding = policy["cash_flow_funding"]
    rr = policy["rate_risk"]

    exposure = krd_exposure_table(holdings, universe, lpf, curve, keys)
    tol = krd_tolerances(exposure.liability_krdv01 * 1000, rr["krd_floor"], rr["krd_pct_of_liability"])
    net_k = exposure.net_krdv01.to_numpy() * 1000
    dgap = duration_gap(exposure, holdings.sum(), lpf(curve))
    gaps = gap_table(asset_cashflows(holdings, per_dollar).to_numpy(), liab.total.to_numpy(), curve,
                     funding["borrow_spread"], funding["beginning_cash"])
    g = gap_summary(gaps, funding["check_years"])
    _, cap = capital_report(holdings, universe, policy["capital"]["budget"])
    el = float(holdings @ u.el_annual)

    out = {"market_value": float(holdings.sum()),
           "yield": float(holdings @ u.ytm / holdings.sum()),
           "expected_income": float(holdings @ (u.ytm - u.el_annual)),
           "net_spread_income": cap["net_spread_income"],
           "expected_credit_loss": el,
           "effective_duration": float(dgap["asset_duration"]),
           "asset_dv01_k": float(dgap["asset_dv01"] * 1000),
           "surplus_dv01_k": float(dgap["surplus_dv01"] * 1000),
           "duration_gap": float(dgap["duration_gap"]),
           "max_abs_net_krd_k": float(np.abs(net_k).max()),
           "krd_tenors_breaching": int(np.sum(np.abs(net_k) > tol + 1e-6)),
           "surplus_dv01_ok": bool(abs(dgap["surplus_dv01"] * 1000) <= rr["surplus_dv01_max"] + 1e-6),
           "rate_limits_ok": bool(abs(dgap["surplus_dv01"] * 1000) <= rr["surplus_dv01_max"] + 1e-6
                                  and int(np.sum(np.abs(net_k) > tol + 1e-6)) == 0),
           "min_annual_gap_y1_to_check": float(gaps.gap.iloc[:funding["check_years"]].min()),
           "min_funding_balance_y1_to_check": float(gaps.cum_reinvest.iloc[:funding["check_years"]].min()),
           "max_cum_shortfall_no_reinvest": g["max_cum_shortfall_cum_no_reinvest"],
           "max_cum_shortfall_reinvest": g["max_cum_shortfall_cum_reinvest"],
           "capital_charge": cap["capital_charge"],
           "budget_utilization": cap["utilization"],
           "capital_efficiency": cap["capital_efficiency"]}
    out.update({f"net_krd_{int(k)}y_k": v for k, v in zip(keys, net_k)})
    out.update(policy_checks(holdings, universe, policy))
    return out


def bucket_allocation(portfolios: dict, universe: pd.DataFrame) -> pd.DataFrame:
    """Bucket x portfolio market value (USD millions = % of the $100M book)."""
    b = universe.set_index("id")["bucket"]
    return pd.DataFrame({name: h.groupby(b.loc[h.index]).sum() for name, h in portfolios.items()}).fillna(0.0)


NO_CREDIT_SPREAD = ("TSY",)              # Treasuries / STRIPS: no credit spread, excluded
BASIS_RISK = ("AGENCY_MBS",)              # spread is OAS / basis risk, reported separately from credit


def spread_dv01_by_bucket(holdings: pd.Series, universe: pd.DataFrame) -> pd.Series:
    """Value lost per +1bp of spread, by bucket (USD thousands per bp) = sum MV * spread duration * 1bp.

    Treasuries are excluded (no credit spread). Agency MBS stays in the table but is OAS / basis risk.
    """
    u = universe.set_index("id").loc[holdings.index]
    out = (holdings * u.spread_dur * 1e-4 * 1000).groupby(u.bucket).sum()
    return out.drop(index=list(NO_CREDIT_SPREAD), errors="ignore")


def spread_dv01_summary(by_bucket: pd.DataFrame) -> pd.DataFrame:
    """Rows: credit_total (excludes Agency MBS) and agency_mbs (OAS / basis); columns = portfolios."""
    basis = by_bucket.reindex(list(BASIS_RISK)).fillna(0.0).sum()
    credit = by_bucket.drop(index=list(BASIS_RISK), errors="ignore").sum()
    return pd.DataFrame({"credit_total": credit, "agency_mbs": basis}).T
