"""Read model outputs and hold the base model in memory.

Static results come from outputs/metrics.json and outputs/tables/*.csv (paths relative to the
repo root, not the working directory). The base model (universe, liabilities, cash flows) is
built once per process by warm(); interactive pages re-solve only the LP on top of it.
"""

import json
from functools import lru_cache

import pandas as pd

import dashboard  # noqa: F401  (puts src/ on sys.path)
from alm.capital import clo_regime_impact
from alm.config import CONFIG_DIR, OUTPUT_DIR, REPO_ROOT
from alm.pipeline import Base, build_base, portfolio_analysis
from alm.portfolios import load_rule_based
from alm.report import lookup

TABLES = OUTPUT_DIR / "tables"
DOCS = REPO_ROOT / "docs"

_state = {"ready": False}


@lru_cache(maxsize=1)
def metrics() -> dict:
    return json.loads((OUTPUT_DIR / "metrics.json").read_text())


def m(path: str):
    """Value at a dotted path in metrics.json, e.g. m('phase5.portfolios.C.net_spread_income')."""
    return lookup(metrics(), path)


@lru_cache(maxsize=None)
def _table(name: str, index_col) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv", index_col=index_col)


def table(name: str, index_col=None) -> pd.DataFrame:
    """outputs/tables/<name>.csv (a copy, so callers may modify it)."""
    return _table(name, index_col).copy()


def config_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(CONFIG_DIR / name, comment="#")


def config_csv_header(name: str) -> list[str]:
    """The '#' provenance lines at the top of a config CSV, verbatim (without the '# ')."""
    lines = (CONFIG_DIR / name).read_text().splitlines()
    return [ln.lstrip("#").strip() for ln in lines if ln.startswith("#")]


@lru_cache(maxsize=1)
def base() -> Base:
    """Universe, liabilities and expected cash flows, built once per process."""
    return build_base()


RULE_BASED = {"A": "A_conservative_ig", "B": "B_yield_seeking"}


@lru_cache(maxsize=3)
def holdings(p: str) -> pd.Series:
    """USD millions by security: A and B from config/portfolios.yaml, C = saved LP solution."""
    if p == "C":
        return _table("portfolio_C_holdings", 0)["market_value"].rename(None)
    return load_rule_based()[RULE_BASED[p]]


@lru_cache(maxsize=3)
def analysis(p: str) -> dict:
    """Cash flow gaps, KRD exposure, surplus scenarios and capital for portfolio p (same code as run_all)."""
    b = base()
    return portfolio_analysis(holdings(p), b.curve, b.keys, b.universe, b.liab, b.per_dollar, b.policy)


@lru_cache(maxsize=3)
def clo_regime(p: str) -> pd.DataFrame:
    """Same holdings, CLO capital under old bond factors vs 12/31/2026 CLO factors."""
    return clo_regime_impact(holdings(p), base().universe)


def warm() -> None:
    """Load outputs, build the base model and per-portfolio analytics; /health reports ready afterwards."""
    metrics()
    base()
    for p in ("A", "B", "C"):
        analysis(p)
    _state["ready"] = True


def is_ready() -> bool:
    return _state["ready"]
