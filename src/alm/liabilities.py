"""Synthetic liability cash flows: hump-shaped benefits, surrenders and expenses.

Projection (annual, end-of-year payments, per unit of initial in-force):
    B_t        = t^k * exp(-t / theta)                    benefit per unit in force
    V_t        = sum_{s>t} B_s DF(s) / DF(t)              reserve per unit after paying B_t
    inforce_t  = inforce_{t-1} * (1 - lapse_t),  inforce_0 = 1
    benefits_t = inforce_{t-1} * B_t
    surrender_t= inforce_{t-1} * lapse_t * V_t           lapsers receive the reserve
    expenses_t = expense_pct * benefits_t
All cash flows are then scaled so PV(L) on the base curve equals target_pv.
The cash flows are projected once on the base curve and held fixed under rate shocks.
"""

import numpy as np
import pandas as pd

from alm.bonds import fixed_price_fn
from alm.config import load_yaml
from alm.curve import SpotCurve


def lapse_rates(t, cfg: dict) -> np.ndarray:
    return np.interp(t, [cfg["decline_start"], cfg["decline_end"]],
                     [cfg["base_rate"], cfg["ultimate_rate"]])


def project_liabilities(base_curve: SpotCurve, cfg: dict | None = None, lapse_multiplier=None,
                        scale: float | None = None) -> pd.DataFrame:
    """Annual liability cash flow table (USD millions), scaled to PV = target_pv.

    lapse_multiplier: optional array (one value per year) applied to the base lapse rates (stress).
    scale: reuse the base-case scale so a stressed block is the SAME block, not re-normalized to target_pv.
    """
    cfg = cfg or load_yaml("liabilities.yaml")
    t = np.arange(1, cfg["horizon_years"] + 1, dtype=float)
    df = base_curve.discount_factor(t)

    unit_benefit = t ** cfg["benefits"]["k"] * np.exp(-t / cfg["benefits"]["theta"])
    reserve = np.array([np.sum(unit_benefit[i + 1:] * df[i + 1:]) / df[i] for i in range(len(t))])
    lapse = lapse_rates(t, cfg["lapse"])
    if lapse_multiplier is not None:
        lapse = np.clip(lapse * np.asarray(lapse_multiplier), 0.0, 1.0)
    inforce = np.cumprod(1.0 - lapse)
    inforce_prev = np.concatenate([[1.0], inforce[:-1]])

    benefits = inforce_prev * unit_benefit
    surrenders = inforce_prev * lapse * reserve
    expenses = cfg["expense_pct_of_benefits"] * benefits
    total = benefits + surrenders + expenses

    scale = scale if scale is not None else cfg["target_pv"] / np.sum(total * df)
    out = pd.DataFrame({"t": t.astype(int), "lapse_rate": lapse, "inforce_end": inforce,
                         "benefits": scale * benefits, "surrenders": scale * surrenders,
                         "expenses": scale * expenses, "total": scale * total, "df": df,
                         "pv": scale * total * df})
    out.attrs["scale"] = float(scale)
    return out


def liability_price_fn(liab: pd.DataFrame):
    """SpotCurve -> PV(L), discounting the fixed projected cash flows at the Treasury curve."""
    return fixed_price_fn(liab["t"].to_numpy(float), liab["total"].to_numpy())


def truncation_diagnostics(base_curve: SpotCurve, extended_horizon: int = 60, cfg: dict | None = None) -> dict:
    """How much the closed 30-year block cuts off, vs the same benefit shape run to `extended_horizon`.

    Both projections are unscaled (scale = 1) so they describe the same policies; the curve is flat beyond 30Y.
    """
    cfg = cfg or load_yaml("liabilities.yaml")
    n = cfg["horizon_years"]
    short = project_liabilities(base_curve, cfg, scale=1.0)
    long = project_liabilities(base_curve, {**cfg, "horizon_years": extended_horizon}, scale=1.0)
    late = slice(n - 10, n)                                          # last ten years of the closed block
    return {"horizon_years": n, "extended_horizon": extended_horizon,
            "pv_share_beyond_horizon": float(long.pv.iloc[n:].sum() / long.pv.sum()),
            "late_surrender_ratio": float(short.surrenders.iloc[late].sum() / long.surrenders.iloc[late].sum())}
