"""Receive-fixed swap overlay (see config/overlay.yaml for every assumption).

Notional is in USD millions. A receive-fixed swap's value per 100 notional is
    V(curve) = PV(par fixed bond at rate K) - PV(floater, margin 0, first fixing at base 1Y)
with K the base-curve par rate, so V(base) = 0 and the swap adds duration without using cash.
"""

from dataclasses import dataclass, replace

import numpy as np

from alm.bonds import fixed_cashflows, par_coupon
from alm.curve import SpotCurve
from alm.krd import key_rate_dv01
from alm.optimizer import ProblemData
from alm.structured import floating_schedule


def swap_value_per_100(tenor: int, curve: SpotCurve, base: SpotCurve) -> float:
    k = par_coupon(base, tenor)
    t, cf = fixed_cashflows(100.0, k, tenor)
    fixed = float(np.sum(cf * curve.discount_factor(t)))
    fl = floating_schedule(100.0, 0.0, tenor, curve, first_fixing=base.zero(1.0))
    floating = float(np.sum(fl["total"] * curve.discount_factor(fl["t"])))
    return fixed - floating


def swap_krdv01_per_musd(tenor: int, base: SpotCurve, keys) -> np.ndarray:
    """Key rate DV01 (USD thousands per bp) per USD 1M notional of a receive-fixed swap."""
    per100 = key_rate_dv01(lambda c: swap_value_per_100(tenor, c, base), base, keys)
    return per100 / 100.0 * 1000.0


@dataclass
class SwapOverlay:
    notional: dict          # tenor -> USD millions
    base: SpotCurve

    def value(self, curve: SpotCurve) -> float:
        """USD millions."""
        return sum(n / 100.0 * swap_value_per_100(t, curve, self.base) for t, n in self.notional.items())

    def krdv01(self, curve: SpotCurve, keys) -> np.ndarray:
        """USD millions per bp."""
        return sum(n / 100.0 * key_rate_dv01(lambda c, t=t: swap_value_per_100(t, c, self.base), curve, keys)
                   for t, n in self.notional.items())


def add_overlay(d: ProblemData, base: SpotCurve, cfg: dict) -> ProblemData:
    """Append one decision variable per swap tenor: zero budget, zero C-1, zero funding cash flows."""
    tenors = cfg["tenors"]
    krd = np.array([swap_krdv01_per_musd(t, base, d.keys) for t in tenors])
    n = len(tenors)
    overlay_flag = np.concatenate([d.extra.get("is_overlay", np.zeros(len(d.ids), bool)), np.ones(n, bool)])
    return replace(
        d, ids=d.ids + [f"SWAP_RECV_{t}Y" for t in tenors],
        net_spread=np.concatenate([d.net_spread, np.full(n, cfg["carry_bp_per_year"] * 1e-4)]),
        cap_factor=np.concatenate([d.cap_factor, np.zeros(n)]),
        krd=np.vstack([d.krd, krd]), cf=np.hstack([d.cf, np.zeros((d.cf.shape[0], n))]),
        bucket=np.concatenate([d.bucket, np.full(n, "SWAP")]),
        designation=np.concatenate([d.designation, np.full(n, "EXEMPT")]),
        issuer=np.concatenate([d.issuer, np.full(n, "SWAP_OVERLAY")]),
        extra={**d.extra, "is_overlay": overlay_flag, "max_total_notional": cfg["max_total_notional"]})
