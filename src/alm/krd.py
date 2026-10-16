"""Key rate DV01 and key rate duration (KRD) via triangular zero-curve bumps.

Sign convention: key rate DV01 is the value LOST when the key rate rises 1bp,
so a normal long bond position has positive key rate DV01.
"""

import numpy as np

from alm.bonds import BP, PriceFn
from alm.curve import SpotCurve, key_rate_shape


def _bumped(curve: SpotCurve, key_tenors, k: int, dz: float) -> SpotCurve:
    return curve.shifted(lambda t: dz * key_rate_shape(t, key_tenors, k), extra_nodes=key_tenors)


def key_rate_dv01(price_fn: PriceFn, curve: SpotCurve, key_tenors, bump: float = BP) -> np.ndarray:
    """Key rate DV01 per key tenor ($ per 1bp), by central differences."""
    key_tenors = np.asarray(key_tenors, dtype=float)
    out = np.empty(len(key_tenors))
    for k in range(len(key_tenors)):
        p_up = price_fn(_bumped(curve, key_tenors, k, bump))
        p_dn = price_fn(_bumped(curve, key_tenors, k, -bump))
        out[k] = (p_dn - p_up) / 2.0 * (BP / bump)
    return out


def key_rate_durations(price_fn: PriceFn, curve: SpotCurve, key_tenors, bump: float = BP) -> np.ndarray:
    """KRD_k = key rate DV01_k / (P * 1bp), in years."""
    return key_rate_dv01(price_fn, curve, key_tenors, bump) / (price_fn(curve) * BP)


def krd_pnl(krdv01, shocks_bp) -> float:
    """First-order P&L of a key rate shock: -sum_k KRDV01_k * shock_k (shocks in bp)."""
    return float(-np.dot(krdv01, shocks_bp))
