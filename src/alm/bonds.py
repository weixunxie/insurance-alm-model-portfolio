"""Fixed-rate bond analytics: cash flows, pricing, YTM, duration, convexity, DV01.

Units: cash flows per 100 face (or USD millions for positions); times in years.
Annual coupons and annual compounding throughout.

Two families of measures:
- Yield-based (Macaulay, modified, analytic convexity, analytic DV01): computed
  from the bond's own YTM, assuming a flat shift of that single yield.
- Curve-based (effective duration / convexity, finite-difference DV01): computed
  by shifting the spot curve and repricing through `price_fn(curve)`. Because
  `price_fn` receives the shifted curve, rate-dependent cash flows (MBS
  prepayments, floating coupons) can be re-projected inside it.
"""

from typing import Callable

import numpy as np
from scipy.optimize import brentq

from alm.curve import SpotCurve, parallel_shift

PriceFn = Callable[[SpotCurve], float]
BP = 1e-4


# ---------------------------------------------------------------- cash flows

def fixed_cashflows(face: float, coupon: float, maturity: int):
    """Annual-pay bullet bond: coupon * face each year, face repaid at maturity."""
    times = np.arange(1, maturity + 1, dtype=float)
    cfs = np.full(maturity, coupon * face)
    cfs[-1] += face
    return times, cfs


def zero_cashflows(face: float, maturity: float):
    """Zero-coupon bond (e.g. Treasury STRIPS): single payment of face at maturity."""
    return np.array([float(maturity)]), np.array([float(face)])


def annuity_cashflows(payment: float, n_years: int):
    """Level annual payment for n years (no principal at the end)."""
    return np.arange(1, n_years + 1, dtype=float), np.full(n_years, float(payment))


# ------------------------------------------------------------------- pricing

def pv_on_curve(times, cfs, curve: SpotCurve, spread: float = 0.0) -> float:
    """PV = sum CF_t * (1 + z(t) + spread)^(-t)."""
    return float(np.sum(cfs * curve.discount_factor(times, spread)))


def price_at_yield(times, cfs, y: float) -> float:
    """PV discounting every cash flow at a single yield y."""
    return float(np.sum(cfs * (1.0 + y) ** (-np.asarray(times))))


def ytm(times, cfs, price: float, lo: float = -0.5, hi: float = 1.0) -> float:
    """Yield to maturity: the y solving price_at_yield(y) = price (IRR)."""
    return brentq(lambda y: price_at_yield(times, cfs, y) - price, lo, hi, xtol=1e-12)


def par_coupon(curve: SpotCurve, maturity: int, spread: float = 0.0) -> float:
    """Annual coupon that prices a bullet bond at 100: (1 - DF_T) / sum DF_t."""
    df = curve.discount_factor(np.arange(1, maturity + 1, dtype=float), spread)
    return float((1.0 - df[-1]) / df.sum())


def fixed_price_fn(times, cfs, spread: float = 0.0) -> PriceFn:
    """Price function for contractual (rate-independent) cash flows."""
    return lambda curve: pv_on_curve(times, cfs, curve, spread)


# ------------------------------------------------------- yield-based measures

def macaulay_duration(times, cfs, y: float) -> float:
    """PV-weighted average time to cash flow (years)."""
    pv = cfs * (1.0 + y) ** (-np.asarray(times))
    return float(np.sum(times * pv) / np.sum(pv))


def modified_duration(times, cfs, y: float) -> float:
    """-(1/P) dP/dy = Macaulay / (1 + y) under annual compounding."""
    return macaulay_duration(times, cfs, y) / (1.0 + y)


def convexity(times, cfs, y: float) -> float:
    """(1/P) d2P/dy2 = sum t(t+1) CF (1+y)^-(t+2) / P (annual compounding)."""
    times = np.asarray(times)
    price = price_at_yield(times, cfs, y)
    return float(np.sum(times * (times + 1) * cfs * (1.0 + y) ** (-(times + 2))) / price)


def curve_modified_duration(times, cfs, curve: SpotCurve, spread: float = 0.0) -> float:
    """Analytic -(1/P) dP/ds for FIXED cash flows discounted at z(t) + s:
    sum t CF (1 + z + s)^-(t+1) / P. This is the exact spread duration when the
    spread bump leaves projected cash flows unchanged (e.g. floater DM bump)."""
    times = np.asarray(times, dtype=float)
    r = 1.0 + curve.zero(times) + spread
    pv = cfs * r ** (-times)
    return float(np.sum(times * pv / r) / np.sum(pv))


def dv01(times, cfs, y: float) -> float:
    """Analytic DV01 = modified duration * price * 1bp (value lost per +1bp)."""
    return modified_duration(times, cfs, y) * price_at_yield(times, cfs, y) * BP


# ------------------------------------------------------- curve-based measures

def dv01_fd(price_fn: PriceFn, curve: SpotCurve, bump: float = BP) -> float:
    """Central-difference DV01 from a parallel spot curve shift, scaled to 1bp."""
    p_up = price_fn(parallel_shift(curve, bump))
    p_dn = price_fn(parallel_shift(curve, -bump))
    return (p_dn - p_up) / 2.0 * (BP / bump)


def effective_duration(price_fn: PriceFn, curve: SpotCurve, bump: float = 0.0025) -> float:
    """(P(-dy) - P(+dy)) / (2 P0 dy), repricing on a parallel-shifted curve."""
    p0 = price_fn(curve)
    p_up = price_fn(parallel_shift(curve, bump))
    p_dn = price_fn(parallel_shift(curve, -bump))
    return (p_dn - p_up) / (2.0 * p0 * bump)


def effective_convexity(price_fn: PriceFn, curve: SpotCurve, bump: float = 0.0025) -> float:
    """(P(-dy) + P(+dy) - 2 P0) / (P0 dy^2)."""
    p0 = price_fn(curve)
    p_up = price_fn(parallel_shift(curve, bump))
    p_dn = price_fn(parallel_shift(curve, -bump))
    return (p_dn + p_up - 2.0 * p0) / (p0 * bump ** 2)
