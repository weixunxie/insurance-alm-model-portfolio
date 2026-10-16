# Assumptions register

Every input and modeling choice, labeled as one of:

- **RULE**: a regulatory rule or published factor, with its source
- **SYNTHETIC**: an illustrative assumption made for this project, not market data
- **SIMPLIFICATION**: a deliberate departure from production or statutory practice

All amounts are in USD millions unless noted. The valuation date is 2026-12-31.

## Market data and pricing

| Item | Label | Detail |
|---|---|---|
| Treasury spot curve (1Y 4.00% … 30Y 4.55%) | SYNTHETIC | `config/curve.yaml`. An illustrative shape, not a market observation. The 20Y→30Y segment is slightly inverted. |
| Interpolation | SIMPLIFICATION | Linear in zero rates, flat outside 1Y–30Y, annual compounding. |
| Credit spreads, PDs, recoveries | SYNTHETIC | `config/universe_assumptions.csv`. Only the ordering by rating (AAA < … < B) is meant to be plausible. |
| Pricing | SIMPLIFICATION | Promised cash flows are discounted at z(t) + constant spread (Z-spread / discount margin). There is no OAS model. |
| Coupons | SYNTHETIC | Blank coupons are solved so that price ≈ 100, then rounded to 1/8%. |
| Payment frequency | SIMPLIFICATION | Annual for everything. Real Treasuries pay semiannually; MBS, ABS and CLOs pay monthly or quarterly. |

## Securities

| Item | Label | Detail |
|---|---|---|
| Issuers | SYNTHETIC | All corporate and structured issuers are fictional codes (ISS_*). |
| Credit loss model | SIMPLIFICATION | A single bond-level annual default rate on the outstanding balance (a CDR for structured bonds). Recovery is paid at the end of the default year. There is no deal-level collateral model, except in the waterfall demo. |
| Agency MBS prepayment | SYNTHETIC + SIMPLIFICATION | Base CPR of 7–10% plus an S-curve response to the change in the 10Y spot rate: CPR = clip(CPR_base + a·(logistic(−b·Δy) − 0.5), 2%, 60%), with a = 0.15 and b = 100 (`config/prepayment.yaml`). The curve is centered so that base-case cash flows are unchanged. **a = 0.15 and b = 100 are judgment-calibrated to a target effective convexity range (about −160 to −250 for these pools); they are not fitted to data.** There is no burnout and no path dependence. The pass-through coupon equals the note rate. Rate-dependent CPR is the default; constant-CPR results are kept as a sensitivity (`outputs/tables/prepayment_sensitivity.csv`, `mbs_prepayment_comparison.csv`). |
| CMBS | SYNTHETIC | 30Y amortization, 10Y balloon, full prepayment lockout. |
| CLO tranches | SIMPLIFICATION | Annual reset with coupon = 1Y forward + margin. The first coupon is fixed at the valuation date. Principal is paid as a bullet at the expected WAL. |
| Waterfall demo | SIMPLIFICATION | One illustrative sequential-pay deal (70 / 20 / 10). Losses are written down from the bottom up. There are no OC/IC triggers. |
| NAIC designations of RMBS / CMBS / CLO | SYNTHETIC | Assigned for illustration only. In practice, NAIC SSG modeling or the filing rules determine them, and they **do not** automatically follow the credit rating. |
| Bond definition | RULE (assumed satisfied) | Since 1/1/2025, the principles-based bond definition decides whether an ABS can be reported on Schedule D. All structured securities here are assumed to qualify. |

## Liabilities

| Item | Label | Detail |
|---|---|---|
| Liability block | SYNTHETIC | Hump-shaped benefits B_t = t^1.5 · e^(−t/8). The lapse rate is 4% in years 1–10 and declines to 2% by year 15. Expenses are 3% of benefits. Everything is scaled to PV(L) = 90. |
| Discounting | SIMPLIFICATION | Economic view on the Treasury curve, with no spread. Statutory reserves use prescribed valuation rates. |
| Closed block | SIMPLIFICATION | The liability block terminates at year 30 (the project's annual grid). The benefit shape's tail beyond year 30 is dropped, and because the reserve V_t only includes benefits to year 30, late surrenders shrink toward zero. Both effects are quantified in the README Limitations (`metrics.json` → `phase2.liability_truncation`). |
| Surrender value | SIMPLIFICATION | Equal to the economic reserve (PV of remaining benefits), with no surrender charge. |
| Rate sensitivity of liabilities | SIMPLIFICATION | Cash flows are fixed under rate shocks (no dynamic lapse). This **understates** liability risk when rates rise; dynamic lapse is in the Phase 6 backlog. |

## Capital

| Item | Label | Detail |
|---|---|---|
| Life RBC C-1 bond factors (20 designations) | RULE | C-1 bond factors (current, 20 designations): 1.A, 2.B and 3.A checked against the Proposal 2026-12-IRE LR002 form (Column 1); the other 17 from Mayer Brown (July 2026, “Current” column), consistent with the 2021 Moody's Analytics base factors. Not yet checked line by line against the full LR002 instructions. |
| CLO factors effective 12/31/2026 | RULE | CLO factors: all 20 designations and the 45% residual checked against the adopted Proposal 2026-12-IRE LR002 form. Adopted by the RBC IRE WG (6/23/2026) and the CATF (6/30/2026). Background: Mayer Brown and Sidley (July 2026). |
| CLO thin-tranche rule | RULE + MODELING CHOICE | A separate surcharge of 11.77% for BSL tranches designated 2.C or below with thickness ≤ 4% (`config/clo_thin_tranche.yaml`). **The form's treatment (line 7.2) is ambiguous between an add-on to the designation factor and a replacement factor; the model uses the add-on reading** (2.C: 3.28% + 11.77% = 15.05%). No universe tranche is ≤ 4% thick, so the rule is exercised only in a unit test. |
| Treasuries and GNMA = 0% | RULE (to verify) | Treated as exempt U.S. government obligations. Verify in LR002. |
| FNMA / FHLMC MBS = NAIC 1.A | SYNTHETIC | A modeling assumption that still needs verification. |
| Charge = market value × factor | SIMPLIFICATION | Statutory RBC uses book / adjusted carrying value, so under stress, charges move with market value here but would not move statutorily. |
| Not modeled | SIMPLIFICATION | Bond size factor, tax effect, covariance across C-0 to C-4, C-3 interest rate risk, ACL and the RBC ratio. **Outputs are not an RBC ratio or a statutory result.** |

## Policy (config/policy.yaml)

| Item | Label | Value |
|---|---|---|
| Capital budget K | SYNTHETIC | $1.20M (frontier sweep from $0.40M to $3.00M) |
| HY (NAIC 3 or below, CLO tranches included) | SYNTHETIC | ≤ 10% |
| Non-agency structured (RMBS, CMBS, ABS, CLO) | SYNTHETIC | ≤ 30% (Agency MBS excluded) |
| Agency MBS | SYNTHETIC | ≤ 20% (negative convexity / prepayment risk) |
| CLO | SYNTHETIC | ≤ 10% |
| Single issuer | SYNTHETIC | ≤ 5% (US Treasury, GNMA, FNMA and FHLMC are exempt) |
| Liquidity (Treasuries + Agency MBS) | SYNTHETIC | ≥ 10% |
| Surplus DV01 | SYNTHETIC | \|DV01_A − DV01_L\| ≤ $5k per bp |
| Key rate DV01 per tenor | SYNTHETIC | \|net\| ≤ max($5k, 20% × liability key rate DV01 at that tenor) |
| Cash flow funding | SYNTHETIC | Cumulative balance ≥ 0 in years 1–10. Positive balances are reinvested at the 1Y forward; negative balances are funded at the forward + 100bp. |

## Optimization and stress

| Item | Label | Detail |
|---|---|---|
| Objective | SIMPLIFICATION | A linear objective: expected net spread income Σ x(s − PD·LGD). There is no risk penalty (Phase 6 option). |
| LP corner solutions | SIMPLIFICATION | Many positions sit exactly at the 5% issuer cap. See the robustness simulation (`robustness.yaml`): spreads × N(1, 0.10) and PDs × a mean-one lognormal (σ = 0.25). Prices, KRDs and expected cash flows are held at base values. Base C held fixed is evaluated on the same draws, and each re-optimized portfolio on an independent draw (out-of-sample), to measure the optimizer's curse. |
| Scenario surplus constraints | SYNTHETIC (policy) | For +100bp, −100bp, steepener and flattener: ΔA_s(x) − ΔL_s ≥ −$0.50M each (`policy.yaml` → `scenario_constraints`, on by default). ΔA uses full-repricing value change per dollar, with cash flows re-projected, so negative convexity is captured. The constraint is exact for each scenario; between scenarios there is no guarantee. |
| Downgrade buffer | SYNTHETIC (policy) | Capital after every BBB-family holding moves down one notch ≤ 1.10 × K (`policy.yaml` → `downgrade_buffer`, on by default), at t0 market value. It does not control the HY-limit count after the downgrade. |
| HY count after downgrade (optional) | SYNTHETIC (policy) | NAIC 3+ holdings after the one-notch BBB downgrade ≤ HY limit + allowance (`downgrade_buffer.hy_limit_after_downgrade`, off by default). Its income cost is reported. |
| Spread DV01 | DEFINITION | Treasuries / STRIPS are excluded (no credit spread). Agency MBS spread DV01 is reported separately as OAS / basis risk, not credit. |
| Budget shadow price | DEFINITION | K enters two rows (capital ≤ K, post-downgrade capital ≤ 1.10·K), so the reported shadow price is dual_capital + 1.10 × dual_buffer, checked against a finite-difference re-solve. |
| Spread shocks | SIMPLIFICATION | Mark-to-market only; PDs and expected cash flows do not change. **HY (NAIC 3+) shocks select by designation 3.A and below in any bucket, including CLO_BB (3.B), the same rule as the HY limit.** The IG +50bp scenario selects the IG corporate bucket, because structured spreads have their own scenario. |
| Stress capital | SIMPLIFICATION | **Default: a book-value proxy (t0 market value × factor)**, closer to statutory BACV, so pure market moves leave the charge unchanged. Downgrades change the factor. Mark-to-market capital (shocked MV × factor) is reported as a sensitivity column. |
| Credit migration | SYNTHETIC | `config/credit_migration.yaml` sets generic spread and PD levels by designation (monotone, illustrative). On a downgrade: the factor moves to the new designation (CLO table for CLOs), the PD scales by the generic ratio, and the spread widens by the generic difference. Treasuries and Agency MBS are not downgraded. Notches follow the 20-category order (2.C → 3.A crosses into NAIC 3). |
| Default stress | SIMPLIFICATION | PD × 2 and recovery −10 points for NAIC 3+ holdings. It changes income and expected cash flows; prices are unchanged because spreads are held. |
| Prepayment stress | SIMPLIFICATION | CPR × 1.5 or × 0.5 for every prepaying security (Agency MBS, RMBS, auto ABS), on top of the S-curve. |
| Surrender stress | SIMPLIFICATION | Lapse × 1.5 in years 1–5 on the same liability block (base scale, no re-normalization). Surrender is paid at the reserve value, so liability PV barely moves and the effect is on liquidity. |
| Swap overlay | SYNTHETIC + SIMPLIFICATION | Results depend on zero swap spread, no collateral and the −1bp carry. Receive-fixed swaps at 10Y / 20Y / 30Y. The swap curve equals the Treasury curve (swap spread 0), struck at par (zero value at t0). They use no asset budget and carry zero modeled C-1. Carry is −1bp a year per USD of notional (bid-ask and collateral cost; expected carry under forwards is 0). Total notional is capped at $50M. Settlements are excluded from the funding test. **Not modeled:** derivative counterparty RBC, collateral and margin liquidity, hedge accounting. |
| Rule-based portfolios A and B | SYNTHETIC | Hand-set weights in `config/portfolios.yaml`. B ignores the capital budget and the rate limits by design. |
