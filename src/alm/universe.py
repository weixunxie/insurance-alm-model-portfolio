"""Investment universe: load assumptions, project cash flows, price, and compute risk measures.

Pricing convention: price = PV of PROMISED cash flows discounted at z(t) + spread
(spread = Z-spread; discount margin for floaters). Expected (credit-adjusted)
cash flows are kept separately for cash flow matching (Module A).
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from alm.bonds import (BP, convexity, effective_convexity, effective_duration, macaulay_duration,
                       modified_duration, ytm)
from alm.config import CONFIG_DIR, load_yaml
from alm.curve import SpotCurve
from alm.krd import key_rate_durations, key_rate_dv01
from alm.structured import (amortizing_schedule, bullet_schedule, expected_cashflows,
                            floating_schedule, rate_dependent_cpr, wal)

FACE = 100.0
COUPON_ROUNDING = 0.00125          # 1/8 pct


@lru_cache(maxsize=1)
def prepayment_config() -> dict:
    return load_yaml("prepayment.yaml")


def load_assumptions(path=None, prepay_model: str | None = None) -> pd.DataFrame:
    """Universe assumptions. prepay_model overrides config/prepayment.yaml ('rate_dependent' | 'constant')."""
    df = pd.read_csv(path or CONFIG_DIR / "universe_assumptions.csv", comment="#")
    df["maturity"] = df["maturity"].astype(int)
    df["amort_term"] = df["amort_term"].fillna(df["maturity"]).astype(int)
    df["float_margin"] = df["float_margin"].fillna(df["spread"])
    cfg = prepayment_config()
    model = prepay_model or cfg["model"]
    df["prepay_model"] = np.where(df.bucket.isin(cfg["apply_to_buckets"]), model, "constant")
    return df


def effective_cpr(sec, curve: SpotCurve, base_curve: SpotCurve) -> float:
    """CPR used on `curve`: constant, or the S-curve response to the 10Y rate change vs base."""
    if getattr(sec, "prepay_model", "constant") != "rate_dependent":
        return sec.cpr
    cfg = prepayment_config()
    dy = float(curve.zero(cfg["rate_tenor"]) - base_curve.zero(cfg["rate_tenor"]))
    return rate_dependent_cpr(sec.cpr, dy, cfg["a"], cfg["b"], cfg["cpr_min"], cfg["cpr_max"])


def schedule(sec, coupon: float, curve: SpotCurve, base_curve: SpotCurve) -> dict:
    """Promised cash flow schedule per 100 face. `curve` drives floating coupons;
    `base_curve` sets the first (already fixed) floating coupon."""
    m = sec.cf_model
    if m in ("bullet", "zero"):
        return bullet_schedule(FACE, coupon, sec.maturity)
    if m == "amortizing":
        return amortizing_schedule(FACE, coupon, sec.amort_term, sec.maturity,
                                   effective_cpr(sec, curve, base_curve))
    if m == "floater":
        return floating_schedule(FACE, sec.float_margin, sec.maturity, curve,
                                 first_fixing=base_curve.zero(1.0))
    raise ValueError(f"unknown cf_model {m!r}")


def price(sec, coupon: float, curve: SpotCurve, base_curve: SpotCurve, dspread: float = 0.0) -> float:
    """PV of promised cash flows (re-projected on `curve`) at z + spread + dspread."""
    s = schedule(sec, coupon, curve, base_curve)
    return float(np.sum(s["total"] * curve.discount_factor(s["t"], sec.spread + dspread)))


def solve_coupon(sec, base_curve: SpotCurve) -> float:
    """Fixed coupon pricing the security at 100, rounded to 1/8 pct (blank CSV coupon)."""
    if not np.isnan(sec.coupon):
        return float(sec.coupon)
    if sec.cf_model == "floater":
        return float("nan")                                   # coupon comes from forwards + margin
    c = brentq(lambda c: price(sec, c, base_curve, base_curve) - FACE, -0.05, 0.30)
    return round(c / COUPON_ROUNDING) * COUPON_ROUNDING


def price_fn(sec, coupon: float, base_curve: SpotCurve):
    """SpotCurve -> price, re-projecting curve-dependent cash flows."""
    return lambda curve: price(sec, coupon, curve, base_curve)


def spread_duration(sec, coupon: float, base_curve: SpotCurve, bump: float = BP) -> float:
    """-(1/P) dP/ds: bump the discount spread (DM) by +/-1bp, Treasury curve fixed."""
    p0 = price(sec, coupon, base_curve, base_curve)
    p_up = price(sec, coupon, base_curve, base_curve, bump)
    p_dn = price(sec, coupon, base_curve, base_curve, -bump)
    return (p_dn - p_up) / (2.0 * p0 * bump)


def security_analytics(sec, base_curve: SpotCurve, key_tenors, eff_bump: float = 0.0025) -> dict:
    """All per-security measures (per 100 face)."""
    coupon = solve_coupon(sec, base_curve)
    s = schedule(sec, coupon, base_curve, base_curve)
    pf = price_fn(sec, coupon, base_curve)
    p = pf(base_curve)
    y = ytm(s["t"], s["total"], p)
    krd = key_rate_durations(pf, base_curve, key_tenors)
    krdv01 = key_rate_dv01(pf, base_curve, key_tenors)
    row = {"id": sec.id, "coupon": coupon, "price": p, "ytm": y, "wal": wal(s),
           "macaulay": macaulay_duration(s["t"], s["total"], y),
           "modified": modified_duration(s["t"], s["total"], y),
           "effective_dur": effective_duration(pf, base_curve, eff_bump),
           "convexity": convexity(s["t"], s["total"], y),
           "effective_conv": effective_convexity(pf, base_curve, eff_bump),
           "dv01": float(krdv01.sum()),
           "spread_dur": spread_duration(sec, coupon, base_curve),
           "el_annual": sec.pd * (1.0 - sec.recovery),
           "net_spread": sec.spread - sec.pd * (1.0 - sec.recovery)}
    row.update({f"krd_{int(k)}y": v for k, v in zip(key_tenors, krd)})
    row.update({f"krdv01_{int(k)}y": v for k, v in zip(key_tenors, krdv01)})
    return row


def build_universe(base_curve: SpotCurve, key_tenors, assumptions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assumptions + computed analytics, one row per security."""
    a = assumptions if assumptions is not None else load_assumptions()
    rows = [security_analytics(sec, base_curve, key_tenors) for sec in a.itertuples(index=False)]
    out = a.drop(columns="coupon").merge(pd.DataFrame(rows), on="id")
    return out


def cashflow_table(universe: pd.DataFrame, base_curve: SpotCurve, horizon: int = 30) -> pd.DataFrame:
    """Long table of promised and expected cash flows per 100 face, years 1..horizon."""
    frames = []
    for sec in universe.itertuples(index=False):
        s = schedule(sec, sec.coupon, base_curve, base_curve)
        e = expected_cashflows(s, sec.pd, sec.recovery)
        frames.append(pd.DataFrame({
            "id": sec.id, "t": s["t"].astype(int), "balance_start": s["balance_start"],
            "interest": s["interest"], "scheduled_principal": s["scheduled_principal"],
            "prepayment": s["prepayment"], "promised": e["promised"], "recovery": e["recovery"],
            "expected": e["expected"], "credit_shortfall": e["credit_shortfall"]}))
    df = pd.concat(frames, ignore_index=True)
    return df[df.t <= horizon]


def cashflow_matrix(cf_table: pd.DataFrame, column: str = "expected", horizon: int = 30) -> pd.DataFrame:
    """Years (rows 1..horizon) x securities (columns), per 100 face."""
    m = cf_table.pivot(index="t", columns="id", values=column)
    return m.reindex(range(1, horizon + 1)).fillna(0.0)
