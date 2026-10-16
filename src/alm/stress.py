"""Scenario engine (Section 10).

A scenario combines any of (all optional):
  curve           curve shift (parallel / twist)
  spread_shocks   list of {select, bp}: mark-to-market spread moves by NAIC class or bucket
  downgrade       designation migration rule (see migration.py): new capital factor, PD, spread widening
  credit          list of {select, pd_mult, recovery_shift}: default-rate stress (income and expected cash flows)
  cpr_mult        multiplier on base CPR for every prepaying security
  lapse           {mult, years: [first, last]}: liability lapse-rate multiplier (same block, base scale)
Holdings are fixed in face terms (units = MV / base price):
  - asset MV = sum units * P_i(shocked curve, spread + shock)
  - liability PV = fixed liability cash flows discounted on the shocked curve
  - cash flows are re-projected (floating coupons follow the shocked forwards) and
    surpluses are reinvested at the shocked 1Y forwards
  - capital charge (default) = BOOK-VALUE PROXY: t0 market value x factor. Statutory
    RBC uses book / adjusted carrying value, which a pure market move does not change.
    The mark-to-market charge (shocked MV x factor) is kept as a sensitivity column.
"""

import numpy as np
import pandas as pd

from alm.cashflow_matching import gap_summary, gap_table
from alm.curve import SpotCurve
from alm.krd import key_rate_dv01
from alm.krd_exposure import BP, scenario_curve
from alm.liabilities import liability_price_fn, project_liabilities
from alm.migration import downgrade_rule, migrate
from alm.optimizer import is_hy
from alm.structured import expected_cashflows
from alm.universe import price, schedule


def select(u: pd.DataFrame, selector: dict) -> np.ndarray:
    """Boolean mask over securities for a spread-shock selector."""
    if "naic" in selector:
        hy = is_hy(u.naic_designation)
        if selector["naic"] == "hy":
            return hy
        if selector["naic"] == "ig":
            return ~hy & (u.naic_designation != "EXEMPT").to_numpy()
        raise ValueError(f"unknown naic selector {selector['naic']!r}")
    if "buckets" in selector:
        return u.bucket.isin(selector["buckets"]).to_numpy()
    raise ValueError(f"unknown selector {selector}")


def spread_shift(u: pd.DataFrame, shocks: list) -> np.ndarray:
    """Total spread shift per security (decimal) from a list of {select, bp}."""
    out = np.zeros(len(u))
    for s in shocks or []:
        out += np.where(select(u, s["select"]), s["bp"] * BP, 0.0)
    return out


def _securities(holdings: pd.Series, universe: pd.DataFrame) -> pd.DataFrame:
    return universe.set_index("id", drop=False).loc[holdings.index]


def shocked_state(holdings: pd.Series, universe: pd.DataFrame, liab: pd.DataFrame, base: SpotCurve, spec: dict):
    """Apply the non-curve parts of a scenario. Returns (securities frame, spread shift, liabilities)."""
    u = _securities(holdings, universe).copy()
    dspread = spread_shift(u, spec.get("spread_shocks"))
    if "downgrade" in spec:
        u, widening = migrate(u, downgrade_rule(u, spec["downgrade"]))
        dspread = dspread + widening
    for c in spec.get("credit", []):
        m = select(u, c["select"])
        u.loc[m, "pd"] = u.loc[m, "pd"] * c.get("pd_mult", 1.0)
        u.loc[m, "recovery"] = np.clip(u.loc[m, "recovery"] + c.get("recovery_shift", 0.0), 0.0, 1.0)
        u["el_annual"] = u["pd"] * (1.0 - u["recovery"])
        u["net_spread"] = u["spread"] - u["el_annual"]
    if "cpr_mult" in spec:
        u["cpr"] = np.clip(u["cpr"] * spec["cpr_mult"], 0.0, 1.0)
    if "lapse" in spec:
        t = liab["t"].to_numpy()
        first, last = spec["lapse"]["years"]
        mult = np.where((t >= first) & (t <= last), spec["lapse"]["mult"], 1.0)
        liab = project_liabilities(base, lapse_multiplier=mult, scale=liab.attrs["scale"])
    return u, dspread, liab


def run_scenario(holdings: pd.Series, universe: pd.DataFrame, liab: pd.DataFrame, base: SpotCurve,
                 keys, spec: dict, policy: dict, overlay=None) -> dict:
    """Full revaluation of one scenario. overlay: optional SwapOverlay (receive-fixed swaps) held alongside."""
    u0 = _securities(holdings, universe)
    units = holdings / u0["price"]
    shocked = scenario_curve(base, spec["curve"]) if "curve" in spec else base
    u, dspread, liab1 = shocked_state(holdings, universe, liab, base, spec)
    secs0, secs = list(u0.itertuples(index=False)), list(u.itertuples(index=False))

    p0 = np.array([price(s, s.coupon, base, base) for s in secs0])
    p1 = np.array([price(s, s.coupon, shocked, base, ds) for s, ds in zip(secs, dspread)])
    lpf0, lpf1 = liability_price_fn(liab), liability_price_fn(liab1)
    d_a = float(units @ (p1 - p0))
    d_l = lpf1(shocked) - lpf0(base)

    a_krd = sum(units.iloc[i] * key_rate_dv01(lambda c, s=s, ds=ds: price(s, s.coupon, c, base, ds), shocked, keys)
                for i, (s, ds) in enumerate(zip(secs, dspread)))
    if overlay is not None:
        d_a += overlay.value(shocked) - overlay.value(base)
        a_krd = a_krd + overlay.krdv01(shocked, keys)
    net_krd = (a_krd - key_rate_dv01(lpf1, shocked, keys)) * 1000

    horizon = len(liab1)
    cf = np.zeros(horizon)
    for n, s in zip(units, secs):
        e = expected_cashflows(schedule(s, s.coupon, shocked, base), s.pd, s.recovery)
        t = e["t"].astype(int)
        cf[t[t <= horizon] - 1] += n * e["expected"][t <= horizon]
    funding = policy["cash_flow_funding"]
    gaps = gap_table(cf, liab1.total.to_numpy(), shocked, funding["borrow_spread"], funding["beginning_cash"])
    g = gap_summary(gaps, funding["check_years"])

    hy_after = float(holdings[is_hy(u.naic_designation)].sum())
    return {"dA": d_a, "dL": d_l, "dS": d_a - d_l,
            "d_net_income": float(holdings @ (u.net_spread - u0.net_spread)),
            "max_abs_net_krd_k": float(np.abs(net_krd).max()), "surplus_dv01_k": float(net_krd.sum()),
            "min_funding_y1_to_check": float(gaps.cum_reinvest.iloc[:funding["check_years"]].min()),
            "first_shortfall_year": g["first_shortfall_year_cum_reinvest"],
            "max_cum_shortfall": g["max_cum_shortfall_cum_reinvest"],
            "capital_charge": float(holdings.to_numpy() @ u.cap_factor.to_numpy()),     # book-value proxy (t0 MV)
            "capital_charge_mtm": float(units @ (p1 * u.cap_factor.to_numpy())),      # sensitivity: shocked MV
            "hy_after": hy_after, "hy_limit_breached": bool(hy_after > policy["limits"]["hy_max"] + 1e-9),
            "net_krd": pd.Series(net_krd, index=[f"{int(k)}y" for k in keys]), "gaps": gaps}


def stress_table(portfolios: dict, universe, liab, base, keys, scenarios: dict, policy,
                 overlays: dict | None = None) -> pd.DataFrame:
    """Rows = portfolio x scenario, headline numbers only (USD millions; DV01 in USD thousands per bp)."""
    rows = []
    overlays = overlays or {}
    for pname, holdings in portfolios.items():
        for sname, spec in scenarios.items():
            r = run_scenario(holdings, universe, liab, base, keys, spec, policy, overlays.get(pname))
            rows.append({"portfolio": pname, "scenario": sname,
                         **{k: v for k, v in r.items() if k not in ("net_krd", "gaps")}})
    return pd.DataFrame(rows)
