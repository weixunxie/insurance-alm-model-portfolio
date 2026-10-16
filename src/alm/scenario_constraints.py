"""Extra LP rows that let the optimizer see convexity and downgrade risk (both linear in holdings).

1. Scenario surplus constraints. For each scenario s, full-repricing value change per USD 1 of
   market value (per USD 1 of notional for swaps), with cash flows re-projected on the shocked
   curve (so Agency MBS negative convexity is captured):
       dP_i,s = P_i(shocked) / P_i(base) - 1
   Constraint: sum_i x_i dP_i,s - dL_s >= -max_loss_s.  Exact for the scenario (no Taylor
   approximation), linear because holdings are fixed in face per USD of market value.
2. Downgrade buffer. Capital after a designation migration rule (default: every BBB-family
   holding down one notch), at t0 market value: sum_i x_i c'_i <= multiple * K.
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from alm.curve import SpotCurve
from alm.krd_exposure import scenario_curve
from alm.liabilities import liability_price_fn
from alm.migration import downgrade_rule, migrate
from alm.optimizer import ProblemData, is_hy
from alm.overlay import swap_value_per_100
from alm.universe import price


def scenario_dp(ids, universe: pd.DataFrame, base: SpotCurve, shocked: SpotCurve) -> np.ndarray:
    """Value change per USD 1 MV (securities) or per USD 1 notional (SWAP_RECV_<T>Y)."""
    u = universe.set_index("id", drop=False)
    out = []
    for i in ids:
        if i.startswith("SWAP_RECV_"):
            tenor = int(i.removeprefix("SWAP_RECV_").removesuffix("Y"))
            out.append(swap_value_per_100(tenor, shocked, base) / 100.0)
        else:
            s = next(u.loc[[i]].itertuples(index=False))
            out.append(price(s, s.coupon, shocked, base) / s.price - 1.0)
    return np.array(out)


def add_scenario_constraints(d: ProblemData, universe, liab, base: SpotCurve, specs: dict,
                             max_loss) -> ProblemData:
    """specs: {name: curve spec}; max_loss: scalar or {name: value} in USD millions."""
    lpf = liability_price_fn(liab)
    l0 = lpf(base)
    names, dps, dls, losses = [], [], [], []
    for name, spec in specs.items():
        shocked = scenario_curve(base, spec)
        names.append(name)
        dps.append(scenario_dp(d.ids, universe, base, shocked))
        dls.append(lpf(shocked) - l0)
        losses.append(max_loss[name] if isinstance(max_loss, dict) else max_loss)
    return replace(d, extra={**d.extra, "scenario_names": names, "scenario_dp": dps,
                             "scenario_dl": dls, "scenario_max_loss": losses})


def add_downgrade_buffer(d: ProblemData, universe, rule: dict, multiple: float,
                         regime: str = "new_2026") -> ProblemData:
    u = universe.set_index("id", drop=False)
    held = [i for i in d.ids if i in u.index]
    migrated, _ = migrate(u.loc[held], downgrade_rule(u.loc[held], rule), regime)
    factor = migrated["cap_factor"]
    c_down = np.array([factor[i] if i in factor.index else 0.0 for i in d.ids])
    return replace(d, extra={**d.extra, "cap_factor_downgraded": c_down,
                             "downgrade_buffer_multiple": multiple})


def add_downgrade_hy_limit(d: ProblemData, universe, rule: dict, limit: float) -> ProblemData:
    """NAIC 3+ holdings after the downgrade rule <= limit (indicator of post-downgrade designation)."""
    u = universe.set_index("id", drop=False)
    held = [i for i in d.ids if i in u.index]
    migrated, _ = migrate(u.loc[held], downgrade_rule(u.loc[held], rule))
    hy_after = pd.Series(is_hy(migrated["naic_designation"]), index=migrated.index)
    ind = np.array([float(hy_after.get(i, False)) for i in d.ids])
    return replace(d, extra={**d.extra, "hy_after_downgrade": ind, "hy_after_downgrade_limit": limit})


def apply_policy_constraints(d: ProblemData, universe, liab, base: SpotCurve, policy: dict, scenarios_cfg: dict,
                             regime: str = "new_2026", scenario_on=None, buffer_on=None,
                             hy_after_on=None) -> ProblemData:
    """Add whichever extra rows policy.yaml switches on (overridable for before/after comparisons)."""
    sc, db = policy["scenario_constraints"], policy["downgrade_buffer"]
    if sc["enabled"] if scenario_on is None else scenario_on:
        specs = {n: scenarios_cfg["rate_scenarios"][n] for n in sc["scenarios"]}
        d = add_scenario_constraints(d, universe, liab, base, specs, sc["max_loss"])
    if db["enabled"] if buffer_on is None else buffer_on:
        d = add_downgrade_buffer(d, universe, db["rule"], db["multiple"], regime)
    hl = db.get("hy_limit_after_downgrade", {})
    if hl.get("enabled", False) if hy_after_on is None else hy_after_on:
        d = add_downgrade_hy_limit(d, universe, db["rule"], policy["limits"]["hy_max"] + hl.get("allowance", 0.0))
    return d
