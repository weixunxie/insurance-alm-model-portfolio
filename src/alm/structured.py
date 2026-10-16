"""Cash flow schedules for amortizing / prepaying / floating securities, credit
adjustment, and a one-deal sequential-pay waterfall demo.

All schedules are annual, per `balance` of face, and return a dict of arrays
indexed by year t = 1..n:
    t, balance_start, interest, scheduled_principal, prepayment, total

SIMPLIFICATIONS (documented in docs/assumptions.md):
- Annual periods; real MBS / ABS / CLO pay monthly or quarterly.
- Constant CPR (rate-dependent prepayment is a Phase 6 extension).
- Pass-through coupon = collateral note rate (no servicing / guarantee fee strip).
- Credit losses use one bond-level annual default rate `pd` (CDR for structured
  bonds) and `recovery`; deal-level collateral modeling is out of scope except
  for the illustrative waterfall below.
"""

import numpy as np

from alm.curve import SpotCurve


def _schedule(t, bal, interest, sched, prepay) -> dict:
    return {"t": t, "balance_start": bal, "interest": interest, "scheduled_principal": sched,
            "prepayment": prepay, "total": interest + sched + prepay}


def bullet_schedule(balance: float, rate: float, maturity: int) -> dict:
    """Fixed coupon, full principal at maturity (also used for zero coupon: rate = 0)."""
    t = np.arange(1, maturity + 1, dtype=float)
    sched = np.zeros(maturity)
    sched[-1] = balance
    return _schedule(t, np.full(maturity, balance), np.full(maturity, rate * balance),
                     sched, np.zeros(maturity))


def amortizing_schedule(balance: float, rate: float, amort_term: int, maturity: int,
                        cpr: float = 0.0) -> dict:
    """Level-pay amortization over `amort_term` years with constant annual CPR.

    Each year: interest = rate * B; scheduled principal from the level payment
    on the remaining amortization term; prepayment = CPR * (B - scheduled).
    If maturity < amort_term the remaining balance is paid as a balloon.
    """
    n = maturity
    bal = np.zeros(n)
    interest, sched, prepay = np.zeros(n), np.zeros(n), np.zeros(n)
    b = balance
    for i in range(n):
        bal[i] = b
        remaining = amort_term - i
        if rate == 0:
            payment = b / remaining
        else:
            payment = b * rate / (1.0 - (1.0 + rate) ** (-remaining))
        interest[i] = rate * b
        sched[i] = payment - interest[i]
        prepay[i] = cpr * (b - sched[i])
        b -= sched[i] + prepay[i]
    sched[-1] += b                                  # balloon (0 if fully amortized)
    return _schedule(np.arange(1, n + 1, dtype=float), bal, interest, sched, prepay)


def rate_dependent_cpr(cpr_base: float, dy: float, a: float, b: float,
                       cpr_min: float, cpr_max: float) -> float:
    """S-curve CPR: CPR_base + a * (logistic(-b * dy) - 0.5), clipped. dy = change in mortgage-rate proxy."""
    logistic = 1.0 / (1.0 + np.exp(b * dy))          # = logistic(-b * dy)
    return float(np.clip(cpr_base + a * (logistic - 0.5), cpr_min, cpr_max))


def floating_schedule(balance: float, margin: float, maturity: int, curve: SpotCurve,
                      first_fixing: float) -> dict:
    """Floater with annual reset: coupon_t = fwd(t-1, t) + margin, bullet principal.

    The first coupon was fixed at the valuation date (`first_fixing`, from the
    base curve) and does NOT move when `curve` is shocked; later coupons are
    re-projected from `curve`. This is why rate duration ~ time to next reset.
    """
    t = np.arange(1, maturity + 1, dtype=float)
    fwd = np.array([curve.forward_rate(x - 1, x) for x in t])
    fwd[0] = first_fixing
    sched = np.zeros(maturity)
    sched[-1] = balance
    return _schedule(t, np.full(maturity, balance), (fwd + margin) * balance,
                     sched, np.zeros(maturity))


def expected_cashflows(schedule: dict, pd: float, recovery: float) -> dict:
    """Credit-adjusted cash flows: survivors get promised CFs, defaulters recover.

    S_t = (1 - pd)^t. Expected CF_t = S_t * promised_t + (S_{t-1} - S_t) * recovery * B_{t-1},
    with recovery received at the end of the default year.
    """
    t = schedule["t"]
    surv = (1.0 - pd) ** t
    surv_prev = (1.0 - pd) ** (t - 1)
    recov = recovery * (surv_prev - surv) * schedule["balance_start"]
    expected = surv * schedule["total"] + recov
    return {"t": t, "promised": schedule["total"], "survival": surv, "recovery": recov,
            "expected": expected, "credit_shortfall": schedule["total"] - expected}


def wal(schedule: dict) -> float:
    """Weighted average life = sum t * principal_t / sum principal_t."""
    principal = schedule["scheduled_principal"] + schedule["prepayment"]
    return float(np.sum(schedule["t"] * principal) / np.sum(principal))


# ------------------------------------------------------------- waterfall demo

def sequential_waterfall(collateral: float, coupon: float, cdr: float, recovery: float,
                         repay_rate: float, maturity: int, tranches: list[dict]) -> dict:
    """One illustrative deal, annual, sequential pay. NOT a full deal model.

    Each year on the performing collateral balance B:
      defaults = cdr * B; loss = (1 - recovery) * defaults
      interest collected = coupon * (B - defaults)
      principal collected = recovery * defaults + repay_rate * (B - defaults)
                            (everything left is repaid at maturity)
    Interest pays tranche coupons top-down; any excess goes to the most junior
    tranche (equity residual). Principal pays tranches top-down (sequential).
    Losses write down tranche balances bottom-up (subordination).

    tranches: list of {"name", "size", "coupon"} from senior to junior; the last
    one is the equity / residual tranche.
    """
    names = [tr["name"] for tr in tranches]
    bal = np.array([tr["size"] for tr in tranches], dtype=float)
    cpn = np.array([tr["coupon"] for tr in tranches], dtype=float)
    interest_paid = np.zeros(len(tranches))
    principal_paid = np.zeros(len(tranches))
    written_down = np.zeros(len(tranches))
    b = collateral
    for year in range(1, maturity + 1):
        defaults = cdr * b
        loss = (1.0 - recovery) * defaults
        performing = b - defaults
        interest = coupon * performing
        repaid = performing if year == maturity else repay_rate * performing
        principal = recovery * defaults + repaid
        b = performing - repaid

        due = cpn * bal                                    # interest on start-of-year balance
        for i in range(len(bal)):                          # interest: senior first
            pay = min(due[i], interest)
            interest_paid[i] += pay
            interest -= pay
        interest_paid[-1] += interest                      # excess spread to equity

        for i in reversed(range(len(bal))):                # losses: junior first
            hit = min(loss, bal[i])
            bal[i] -= hit
            written_down[i] += hit
            loss -= hit

        for i in range(len(bal)):                          # principal: senior first
            pay = min(principal, bal[i])
            bal[i] -= pay
            principal_paid[i] += pay
            principal -= pay
        principal_paid[-1] += principal                    # leftover to residual

    sizes = np.array([tr["size"] for tr in tranches], dtype=float)
    return {"tranche": names, "size": sizes, "interest_paid": interest_paid,
            "principal_paid": principal_paid, "written_down": written_down,
            "loss_pct": written_down / sizes}
