"""Spot (zero) curve: interpolation, discount factors, forwards and curve shifts.

Conventions (see docs/methodology.md):
- Spot rates z(t) are annually compounded decimals; DF(t) = (1 + z(t) + s)^(-t).
- Linear interpolation on zero rates, flat extrapolation beyond the end nodes.
- A curve shift is any function dz(t) added to z(t). All shifts used in this
  project are piecewise linear with kinks only at key tenors, so evaluating
  the shift on the union of curve nodes and key tenors represents it exactly.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from alm.config import load_yaml


@dataclass(frozen=True)
class SpotCurve:
    """Annually compounded spot curve defined on a tenor grid (years)."""

    tenors: np.ndarray
    rates: np.ndarray

    def __post_init__(self):
        tenors = np.asarray(self.tenors, dtype=float)
        rates = np.asarray(self.rates, dtype=float)
        if tenors.shape != rates.shape or tenors.ndim != 1:
            raise ValueError("tenors and rates must be 1-D arrays of equal length")
        if np.any(np.diff(tenors) <= 0):
            raise ValueError("tenors must be strictly increasing")
        object.__setattr__(self, "tenors", tenors)
        object.__setattr__(self, "rates", rates)

    def zero(self, t):
        """Spot rate z(t); linear between nodes, flat outside."""
        return np.interp(t, self.tenors, self.rates)

    def discount_factor(self, t, spread: float = 0.0):
        """DF(t) = (1 + z(t) + spread)^(-t)."""
        t = np.asarray(t, dtype=float)
        return (1.0 + self.zero(t) + spread) ** (-t)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Annually compounded forward rate between t1 and t2 (t2 > t1)."""
        ratio = self.discount_factor(t1) / self.discount_factor(t2)
        return float(ratio ** (1.0 / (t2 - t1)) - 1.0)

    def shifted(self, shift_fn: Callable, extra_nodes=()) -> "SpotCurve":
        """New curve with z'(t) = z(t) + shift_fn(t), evaluated on curve nodes + extra_nodes."""
        nodes = np.union1d(self.tenors, np.asarray(extra_nodes, dtype=float))
        return SpotCurve(nodes, self.zero(nodes) + shift_fn(nodes))


def load_curve(cfg: dict | None = None) -> SpotCurve:
    """Base synthetic Treasury spot curve from config/curve.yaml."""
    cfg = cfg or load_yaml("curve.yaml")
    spot = cfg["spot_curve"]
    return SpotCurve(spot["tenors"], spot["rates"])


def load_key_tenors(cfg: dict | None = None) -> np.ndarray:
    cfg = cfg or load_yaml("curve.yaml")
    return np.asarray(cfg["key_tenors"], dtype=float)


def flat_curve(rate: float, tenors=(1.0, 30.0)) -> SpotCurve:
    """Flat spot curve (used in textbook tests)."""
    return SpotCurve(tenors, [rate] * len(tenors))


def parallel_shift(curve: SpotCurve, dz: float) -> SpotCurve:
    """Shift every spot rate by dz (decimal)."""
    return curve.shifted(lambda t: np.full_like(t, dz))


def key_rate_shape(t, key_tenors, k: int):
    """Triangular bump weight for key tenor index k at maturity t.

    Equals 1 at key_tenors[k], falls linearly to 0 at the neighbouring key
    tenors; the first and last buckets extend flat to the ends. The weights
    of all key tenors sum to 1 at every t (so the sum of KRDs = parallel).
    """
    unit = np.zeros(len(key_tenors))
    unit[k] = 1.0
    return np.interp(t, key_tenors, unit)


def key_rate_shocks(curve: SpotCurve, key_tenors, shocks) -> SpotCurve:
    """Apply a shock dz_k (decimal) at each key tenor, interpolated triangularly."""
    key_tenors = np.asarray(key_tenors, dtype=float)
    shocks = np.asarray(shocks, dtype=float)
    return curve.shifted(lambda t: np.interp(t, key_tenors, shocks), extra_nodes=key_tenors)


def linear_twist(curve: SpotCurve, dz_short: float, dz_long: float,
                 t_short: float = 1.0, t_long: float = 30.0) -> SpotCurve:
    """Shift linear in maturity from dz_short at t_short to dz_long at t_long, flat outside.

    Steepener: dz_short < dz_long. Flattener: dz_short > dz_long.
    """
    return curve.shifted(lambda t: np.interp(t, [t_short, t_long], [dz_short, dz_long]),
                         extra_nodes=[t_short, t_long])
