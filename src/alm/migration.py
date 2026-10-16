"""Rating (NAIC designation) migration: notch downgrades and their capital / income / value effects.

See config/credit_migration.yaml for the generic spread and PD levels (synthetic) and the rules.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from alm.capital import capital_factor, load_bond_factors, load_clo_factors, load_thin_tranche
from alm.config import load_yaml
from alm.optimizer import is_hy

BP = 1e-4
BBB_FAMILY = ("2.A", "2.B", "2.C")


@lru_cache(maxsize=1)
def migration_config() -> dict:
    cfg = load_yaml("credit_migration.yaml")
    cfg["notch_order"] = [str(d) for d in cfg["notch_order"]]
    cfg["spread_bp"] = {str(k): v for k, v in cfg["spread_bp"].items()}
    cfg["pd"] = {str(k): v for k, v in cfg["pd"].items()}
    return cfg


def notch_down(designation: str, notches: int) -> str:
    order = migration_config()["notch_order"]
    if designation not in order:
        return designation                                    # EXEMPT stays exempt
    return order[min(order.index(designation) + notches, len(order) - 1)]


def migrate(u: pd.DataFrame, new_designation: pd.Series, regime: str = "new_2026") -> tuple[pd.DataFrame, np.ndarray]:
    """Apply new designations to the rows of u (indexed like new_designation).

    Returns (migrated frame with naic_designation, cap_factor, pd updated; spread widening in decimal).
    """
    cfg = migration_config()
    out = u.copy()
    old = u["naic_designation"].astype(str)
    new = new_designation.reindex(u.index).fillna(old).astype(str)
    out["naic_designation"] = new
    bond, clo, thin = load_bond_factors(), load_clo_factors(), load_thin_tranche()
    out["cap_factor"] = [capital_factor(s, regime, bond, clo, thin) for s in out.itertuples(index=False)]
    changed = (new != old).to_numpy()
    ratio = np.array([cfg["pd"].get(n, 1.0) / cfg["pd"].get(o, 1.0) if c else 1.0
                      for o, n, c in zip(old, new, changed)])
    out["pd"] = u["pd"].to_numpy() * ratio
    out["el_annual"] = out["pd"] * (1.0 - out["recovery"])
    out["net_spread"] = out["spread"] - out["el_annual"]
    widening = np.array([(cfg["spread_bp"][n] - cfg["spread_bp"][o]) * BP if c else 0.0
                         for o, n, c in zip(old, new, changed)])
    return out, widening


def eligible(u: pd.DataFrame) -> np.ndarray:
    return ~u["bucket"].isin(migration_config()["downgrade_exempt_buckets"]).to_numpy()


def downgrade_rule(u: pd.DataFrame, rule: dict) -> pd.Series:
    """New designations for a scenario rule: {notches: n, buckets: [...]?} or {bbb_family: one_notch | to: '3.A'}."""
    d = u["naic_designation"].astype(str)
    mask = eligible(u)
    if "buckets" in rule:
        mask &= u["bucket"].isin(rule["buckets"]).to_numpy()
    if "bbb_family" in rule:
        mask &= d.isin(BBB_FAMILY).to_numpy()
        if rule["bbb_family"] == "one_notch":
            return pd.Series(np.where(mask, [notch_down(x, 1) for x in d], d), index=u.index)
        return pd.Series(np.where(mask, rule["bbb_family"], d), index=u.index)     # e.g. '3.A' fallen angels
    n = rule["notches"]
    return pd.Series(np.where(mask, [notch_down(x, n) for x in d], d), index=u.index)


def downgrade_sensitivity(holdings: pd.Series, universe: pd.DataFrame, hy_limit: float) -> pd.DataFrame:
    """Bucket x {1, 2}-notch downgrade: change in capital charge, expected net income and MV (first order)."""
    u = universe.set_index("id", drop=False).loc[holdings.index]
    rows = []
    for bucket in ["ALL"] + sorted(set(u.bucket[eligible(u)])):
        for n in (1, 2):
            rule = {"notches": n} if bucket == "ALL" else {"notches": n, "buckets": [bucket]}
            new_u, widening = migrate(u, downgrade_rule(u, rule))
            rows.append({"bucket": bucket, "notches": n,
                         "d_capital": float(holdings @ (new_u.cap_factor - u.cap_factor)),
                         "d_net_income": float(holdings @ (new_u.net_spread - u.net_spread)),
                         "d_mv_first_order": float(-(holdings * u.spread_dur * widening).sum()),
                         "hy_after": float(holdings[is_hy(new_u.naic_designation)].sum()),
                         "hy_limit_breached": bool(holdings[is_hy(new_u.naic_designation)].sum() > hy_limit + 1e-9)})
    return pd.DataFrame(rows)
