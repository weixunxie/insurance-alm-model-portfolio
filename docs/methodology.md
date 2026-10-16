<!-- GENERATED FILE: edit docs/templates/methodology.md.tmpl, then run scripts/run_all.py. -->
# Methodology

Notes on modeling choices. Every number in this project is illustrative and based on synthetic data.

## Spread duration vs WAL

Spread duration is below WAL because it is a PV-weighted average time, not a principal-weighted one: coupons arrive before principal and pull the weighted time forward, and dividing by (1 + z + s) under annual discounting lowers it further, so the gap widens as the coupon or spread rises.

## Cash flow matching vs duration matching vs KRD matching

- **Cash flow matching** asks whether asset inflows cover liability outflows year by year (gap, cumulative funding balance). It controls liquidity and reinvestment risk, but says nothing directly about market value sensitivity.
- **Duration matching** sets asset dollar duration (DV01) equal to liability DV01, so the economic surplus is immune to small *parallel* moves only.
- **KRD matching** sets key rate DV01 equal at every key tenor, so the surplus is also immune to small *non-parallel* moves (steepeners, flatteners, twists).

Portfolio A illustrates the difference. Its duration gap is only +0.53 years (surplus DV01 +5.3k per bp, USD), yet its net key rate DV01 is +18.2k at 30Y and -9.9k at 20Y. A 1Y-to-30Y steepener (−50bp / +50bp) changes its surplus by -0.69 (USD millions) under full repricing. Its cumulative funding balance stays positive in years 1–10, but without reinvestment it turns negative in year 19.

## Duration gap definition

All durations are dollar-based effective durations built from summed key rate DV01s (USD millions per bp):

- D_A = DV01_A / (A × 1bp), D_L = DV01_L / (L × 1bp), with A = asset market value and L = PV of liabilities on the Treasury curve.
- **Duration gap = D_A − (L/A) · D_L** (years).
- **Surplus DV01 = DV01_A − DV01_L = (D_A · A − D_L · L) · 1bp = duration gap × A × 1bp.**

Example (Portfolio A): D_A = 9.102, D_L = 9.520, L/A = 0.90, so the gap is D_A − (L/A)·D_L = +0.534 years, and surplus DV01 = gap × A × 1bp = +5.34k per bp (USD). A positive gap means surplus falls when rates rise.

## Integrated optimization (LP)

max Σ x_i (s_i − EL_i) over x_i ≥ 0 (USD millions of market value), subject to: budget Σ x_i = 100; capital Σ x_i c_i ≤ K; per key tenor |net key rate DV01_k| ≤ max($5k, 20% × liability key rate DV01_k); |surplus DV01| ≤ $5k per bp; cumulative funding balance C = M (CF x − L) ≥ 0 for years 1–10 (M compounds positive balances at 1Y forwards, so the constraint stays linear); HY (NAIC 3+) ≤ 10; non-agency structured ≤ 30; Agency MBS ≤ 20; CLO ≤ 10; single non-government issuer ≤ 5; Treasuries + Agency MBS ≥ 10. Scenario surplus limits and the downgrade buffer (below) are added when switched on in policy.yaml. With the overlay (Phase 6), swap notional ≤ 50. Solved with HiGHS. Every constraint is a linear row a·x ≤ b, so the dual of the capital row is the shadow price of capital: the extra net spread income per year from one more dollar of capital budget (checked against a finite-difference re-solve in the tests). If the hard problem is infeasible, it is re-solved with penalized slack on every inequality, and the rows that need slack are reported as the conflicting constraints.

## Stress classification and capital

- **HY.** HY spread and default shocks select NAIC 3.A and below in any bucket (CLO_BB included), the same rule as the HY limit, so a scenario and the limit it threatens always cover the same assets.
- **Capital.** Stress capital defaults to a book-value proxy (t0 market value × factor). Statutory RBC uses book / adjusted carrying value, which a market move does not change, so only a designation change (downgrade) moves the charge. Mark-to-market capital is kept as a sensitivity.

## Rate-dependent prepayment and negative convexity

The Agency MBS CPR responds to the change in the 10Y rate through a centered S-curve. When rates fall, prepayments speed up and principal comes back at par just when the bond is worth more (price compression). When rates rise, prepayments slow and the bond extends just when it is worth less. Both effects work against the holder, which is negative effective convexity. Because the base case is unchanged, modified duration (fixed cash flows) is unchanged, but effective duration and convexity (re-projected cash flows) are not. The LP's rate constraints are linear, so they miss this; the full-repricing stress shows it.

## Optimizer's curse

Each robustness draw perturbs spreads and PDs and re-solves the LP. The re-optimized in-sample income is max over x of (noisy coefficients · x), which is biased upward. Evaluating fixed base C on the same draws gives an unbiased reference. Evaluating each re-optimized portfolio on an independent draw shows that chasing noise does not beat fixed C out-of-sample.

## Swap overlay

A receive-fixed swap is a long par fixed bond minus a floater, so it adds key rate DV01 at its tenor (and a small negative 1Y DV01 from the floating leg) without using cash. In the LP it is a variable with zero budget weight, zero C-1, zero funding cash flow, an explicit carry cost and a notional cap. The optimizer can then hold spread assets and get duration from swaps instead of zero-spread Treasuries.

## Scenario surplus constraints and downgrade buffer

Duration and KRD rows are first-order, so an LP built only on them happily buys negatively convex assets whose linear duration helps. For each curve scenario s, the model therefore computes the full-repricing value change per dollar of market value for every security, with cash flows re-projected on the shocked curve: dP_i,s = P_i(shocked) / P_i(base) − 1. For swaps it uses the value change per dollar of notional. The row is

ΔA_s(x) − ΔL_s = Σ_i x_i dP_i,s − ΔL_s ≥ −L_max,s.

The row is linear because holdings are fixed in face per dollar of market value, and it is exact for that scenario (no Taylor expansion); between scenarios there is no guarantee. With L_max = 0.50, the binding row is −100bp (dual 0.0426: income per year per extra dollar of allowed loss).

The downgrade buffer is one more linear row: Σ_i x_i c'_i ≤ 1.10·K, where c'_i is the capital factor after every BBB-family holding moves down one notch. Because K now appears in two rows, the shadow price of the budget is dual_capital + 1.10 × dual_buffer = 0.1069.
