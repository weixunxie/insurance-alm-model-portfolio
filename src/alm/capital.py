"""Module C: simplified NAIC Life RBC C-1 bond capital charges.

Modeled charge = market value x pre-tax factor. SIMPLIFICATIONS (see docs/assumptions.md):
statutory RBC uses book / adjusted carrying value; no bond size factor, no tax
effect, no covariance across C-0..C-4, no RBC ratio. This is NOT a statutory result.

Factor rules:
- EXEMPT (Treasuries, GNMA): 0.
- CLO tranches under regime 'new_2026': 12/31/2026 CLO table (LR002 form), plus the
  thin-tranche surcharge (config/clo_thin_tranche.yaml) for BSL tranches designated 2.C or
  below with thickness <= 4%, modeled as an add-on (the form is ambiguous on add-on vs replacement).
- Everything else (and CLOs under regime 'old_bond'): corporate bond factor by designation.
"""

import numpy as np
import pandas as pd

from alm.config import CONFIG_DIR, load_yaml

REGIMES = ("new_2026", "old_bond")
DESIGNATION_ORDER = ["1.A", "1.B", "1.C", "1.D", "1.E", "1.F", "1.G", "2.A", "2.B", "2.C",
                     "3.A", "3.B", "3.C", "4.A", "4.B", "4.C", "5.A", "5.B", "5.C", "6"]


def load_bond_factors() -> pd.Series:
    df = pd.read_csv(CONFIG_DIR / "naic_life_rbc_bond_factors.csv", comment="#", dtype={"designation": str})
    return df.set_index("designation")["factor_pre_tax"]


def load_clo_factors() -> pd.DataFrame:
    df = pd.read_csv(CONFIG_DIR / "naic_life_rbc_clo_factors.csv", comment="#", dtype={"designation": str})
    return df.set_index("designation")


def load_thin_tranche() -> dict:
    cfg = load_yaml("clo_thin_tranche.yaml")
    if cfg["mode"] != "add_on":
        raise ValueError("only the add_on reading of the thin-tranche rule is implemented")
    return cfg


def thin_tranche_surcharge(designation: str, thickness: float, collateral: str, cfg: dict | None = None) -> float:
    """Surcharge added to the CLO designation factor (0 if the tranche does not qualify)."""
    cfg = cfg or load_thin_tranche()
    d = str(designation)
    if d not in DESIGNATION_ORDER:
        return 0.0
    qualifies = (collateral == cfg["collateral"] and thickness <= cfg["max_thickness"]
                 and DESIGNATION_ORDER.index(d) >= DESIGNATION_ORDER.index(cfg["min_designation"]))
    return float(cfg["surcharge"]) if qualifies else 0.0


def capital_factor(sec, regime: str = "new_2026", bond=None, clo=None, thin=None) -> float:
    """Pre-tax capital factor for one security (row with designation, bucket, thickness, collateral)."""
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}")
    bond = load_bond_factors() if bond is None else bond
    if sec.bucket == "CLO" and regime == "new_2026":
        clo = load_clo_factors() if clo is None else clo
        return float(clo.loc[str(sec.naic_designation), "factor"]) + thin_tranche_surcharge(
            sec.naic_designation, sec.tranche_thickness, sec.clo_collateral, thin)
    return float(bond.loc[sec.naic_designation])


def add_capital_factors(universe: pd.DataFrame, regime: str = "new_2026") -> pd.DataFrame:
    """Universe with cap_factor and capital efficiency = net spread / factor (inf if exempt)."""
    bond, clo, thin = load_bond_factors(), load_clo_factors(), load_thin_tranche()
    out = universe.copy()
    out["cap_factor"] = [capital_factor(s, regime, bond, clo, thin) for s in out.itertuples(index=False)]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["capital_efficiency"] = np.where(out.cap_factor > 0, out.net_spread / out.cap_factor, np.inf)
    return out


def capital_charge(market_value: float, factor: float) -> float:
    return market_value * factor


def capital_report(holdings: pd.Series, universe: pd.DataFrame, budget: float) -> tuple[pd.DataFrame, dict]:
    """Per-security charges plus portfolio totals. universe must contain cap_factor."""
    u = universe.set_index("id").loc[holdings.index]
    rows = pd.DataFrame({"bucket": u.bucket, "designation": u.naic_designation, "market_value": holdings,
                         "factor": u.cap_factor, "charge": holdings * u.cap_factor,
                         "net_income": holdings * u.net_spread})
    total = float(rows.charge.sum())
    rows["contribution"] = rows.charge / total if total > 0 else 0.0
    summary = {"capital_charge": total, "budget": budget, "utilization": total / budget,
               "net_spread_income": float(rows.net_income.sum()),
               "capital_efficiency": float(rows.net_income.sum() / total) if total > 0 else np.inf}
    return rows, summary


def bucket_capital(report: pd.DataFrame) -> pd.DataFrame:
    return report.groupby("bucket", sort=False)[["market_value", "charge", "net_income"]].sum()


def clo_regime_impact(holdings: pd.Series, universe: pd.DataFrame) -> pd.DataFrame:
    """CLO capital charge under old bond factors vs new 12/31/2026 CLO factors."""
    rows = []
    for regime in REGIMES:
        u = add_capital_factors(universe, regime).set_index("id").loc[holdings.index]
        clo = u.bucket == "CLO"
        rows.append({"regime": regime, "clo_charge": float((holdings[clo] * u.cap_factor[clo]).sum()),
                     "total_charge": float((holdings * u.cap_factor).sum())})
    return pd.DataFrame(rows)


def ig_to_hy_reallocation(universe: pd.DataFrame, from_id: str, amount: float = 1.0) -> pd.DataFrame:
    """Incremental charge and net income from moving `amount` USD millions from one IG bond into each HY bond."""
    u = universe.set_index("id")
    src = u.loc[from_id]
    hy = u[u.bucket == "HY"]
    return pd.DataFrame({"from": from_id, "to": hy.index,
                         "d_capital": amount * (hy.cap_factor - src.cap_factor),
                         "d_net_income": amount * (hy.net_spread - src.net_spread)}).reset_index(drop=True)
