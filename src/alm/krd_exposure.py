"""Module B: portfolio and liability KRD exposure, duration gap, surplus scenario P&L.

Units: holdings in USD millions of market value; key rate DV01 in USD millions
per bp (multiply by 1000 for $k per bp). Positive DV01 = value lost when rates rise.
Economic surplus S = A - L, with L the PV of fixed liability cash flows on the
Treasury curve.
"""

import numpy as np
import pandas as pd

from alm.curve import SpotCurve, linear_twist, parallel_shift
from alm.krd import key_rate_dv01
from alm.universe import price

BP = 1e-4


def krd_columns(keys) -> list[str]:
    return [f"krdv01_{int(k)}y" for k in keys]


def position_krdv01(holdings: pd.Series, universe: pd.DataFrame, keys) -> pd.DataFrame:
    """Security x tenor key rate DV01 of each position (USD millions per bp)."""
    u = universe.set_index("id").loc[holdings.index]
    units = holdings / u["price"]                         # hundreds of face per USD 1 MV
    out = u[krd_columns(keys)].mul(units, axis=0)
    out.columns = [f"{int(k)}y" for k in keys]
    return out


def bucket_krdv01(holdings: pd.Series, universe: pd.DataFrame, keys) -> pd.DataFrame:
    """Bucket x tenor key rate DV01 (USD millions per bp)."""
    pos = position_krdv01(holdings, universe, keys)
    buckets = universe.set_index("id").loc[holdings.index, "bucket"]
    return pos.groupby(buckets, sort=False).sum()


def krd_exposure_table(holdings, universe, liability_pf, curve: SpotCurve, keys) -> pd.DataFrame:
    """Asset vs liability key rate DV01 and net mismatch by tenor (USD millions per bp)."""
    a = position_krdv01(holdings, universe, keys).sum().to_numpy()
    l = key_rate_dv01(liability_pf, curve, keys)
    pv_a, pv_l = float(holdings.sum()), liability_pf(curve)
    return pd.DataFrame({"tenor": keys, "asset_krdv01": a, "liability_krdv01": l, "net_krdv01": a - l,
                         "asset_krd": a / (pv_a * BP), "liability_krd": l / (pv_l * BP)})


def duration_gap(exposure: pd.DataFrame, assets: float, liabilities: float) -> dict:
    """Duration gap D_A - (L/A) D_L and dollar measures, from summed key rate DV01s."""
    dv01_a, dv01_l = exposure.asset_krdv01.sum(), exposure.liability_krdv01.sum()
    d_a, d_l = dv01_a / (assets * BP), dv01_l / (liabilities * BP)
    return {"asset_duration": d_a, "liability_duration": d_l,
            "duration_gap": d_a - liabilities / assets * d_l,
            "asset_dv01": dv01_a, "liability_dv01": dv01_l, "surplus_dv01": dv01_a - dv01_l}


# ------------------------------------------------------------------ scenarios

def scenario_curve(curve: SpotCurve, spec: dict) -> SpotCurve:
    if spec["type"] == "parallel":
        return parallel_shift(curve, spec["bp"] * BP)
    if spec["type"] == "twist":
        return linear_twist(curve, spec["short_bp"] * BP, spec["long_bp"] * BP,
                            spec["t_short"], spec["t_long"])
    raise ValueError(f"unknown scenario type {spec['type']!r}")


def scenario_key_shocks_bp(base: SpotCurve, shocked: SpotCurve, keys) -> np.ndarray:
    """Shift at each key tenor (bp). Exact triangular decomposition for shifts with kinks at key tenors."""
    keys = np.asarray(keys, dtype=float)
    return (shocked.zero(keys) - base.zero(keys)) / BP


def portfolio_value(holdings: pd.Series, universe: pd.DataFrame, curve: SpotCurve, base: SpotCurve) -> float:
    """Full repricing of the portfolio on `curve` (cash flows re-projected where curve-dependent)."""
    u = universe.set_index("id", drop=False).loc[holdings.index]
    units = holdings / u["price"]
    prices = [price(sec, sec.coupon, curve, base) for sec in u.itertuples(index=False)]
    return float(np.dot(units, prices))


def surplus_scenarios(holdings, universe, liability_pf, base: SpotCurve, keys, scenarios: dict,
                      exposure: pd.DataFrame) -> pd.DataFrame:
    """Delta A, Delta L, Delta S by full repricing vs the first-order KRD approximation."""
    a0, l0 = portfolio_value(holdings, universe, base, base), liability_pf(base)
    rows = []
    for name, spec in scenarios.items():
        shocked = scenario_curve(base, spec)
        d_a = portfolio_value(holdings, universe, shocked, base) - a0
        d_l = liability_pf(shocked) - l0
        shocks = scenario_key_shocks_bp(base, shocked, keys)
        d_a_krd = -float(exposure.asset_krdv01 @ shocks)
        d_l_krd = -float(exposure.liability_krdv01 @ shocks)
        rows.append({"scenario": name, "dA_full": d_a, "dL_full": d_l, "dS_full": d_a - d_l,
                     "dA_krd": d_a_krd, "dL_krd": d_l_krd, "dS_krd": d_a_krd - d_l_krd,
                     "dS_approx_error": (d_a_krd - d_l_krd) - (d_a - d_l)})
    return pd.DataFrame(rows)
