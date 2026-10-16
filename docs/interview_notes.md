<!-- GENERATED FILE: edit docs/templates/interview_notes.md.tmpl, then run scripts/run_all.py. Numbers come from outputs/metrics.json. -->
# Interview Notes

*Educational project on illustrative, synthetic data. Every model number below is read from `outputs/metrics.json`. Amounts are in USD millions unless noted. Lines marked as textbook examples are unit-test fixtures, not model outputs.*

## 2-minute pitch

I built a reproducible Python model that designs a fixed income portfolio for a hypothetical U.S. life insurer: $100M of assets backing $90M (PV) of annuity and life liabilities. It answers a portfolio manager's question: how much spread can I earn without breaking regulatory capital, interest rate risk, or liquidity limits?

It has three modules:

- **Cash flow matching** checks that asset cash flows fund liability payments year by year.
- **KRD exposure** compares asset and liability sensitivity tenor by tenor, not just in total.
- **NAIC capital** applies the C-1 bond factors, including the new 12/31/2026 CLO factors.

An LP combines them. It maximizes expected net spread income subject to a capital budget, key rate and surplus DV01 limits, a funding test, bucket and issuer limits, full-repricing scenario loss limits, and a downgrade capital buffer.

The optimized portfolio earns 1.239 a year of net spread versus 0.612 for a conservative IG portfolio. It keeps every key rate tenor inside tolerance, where the conservative portfolio breaks 3. It uses 1.191 of a 1.20 capital budget.

Three findings I'd highlight:

- **A linear LP misses convexity.** With only duration and KRD rows, it held negatively convex Agency MBS at the 20 limit and ran surplus DV01 at its floor. Exact full-repricing scenario rows moved the −100bp loss from -0.73 to -0.50 for 0.0083 a year of income.
- **The new CLO factors reshape allocations.** Under the old factors, the optimizer would hold far more BB CLO and earn 1.332.
- **Re-optimizing on noisy inputs is a trap.** The in-sample gain of 0.0136 disappears out-of-sample.

Everything is tested (70 unit tests), config-driven, and every number in the docs is generated from one metrics file.

## Three versions

**HR / non-technical (30 seconds).** Insurers invest premiums in bonds so they can pay future claims. I built a model that picks the bond mix earning the most while respecting regulatory capital rules, interest rate risk limits and the need for cash in each year. It roughly doubles the spread income of a conservative portfolio (1.239 vs 0.612 a year on $100M) while staying inside every limit, and I documented every assumption.

**Portfolio manager (1 minute).** The binding trade-offs are capital, the HY bucket and the −100bp scenario.

- **Capital.** One more dollar of budget is worth 0.107 a year.
- **HY bucket.** A BBB → BB+ swap looks great on capital efficiency, but the NAIC 3+ limit is full (dual 0.0082), so its reduced cost is zero.
- **−100bp scenario.** Spread assets are shorter than the liabilities, so the book runs short duration up to the 0.50 loss limit.
  - A receive-fixed overlay replaces zero-spread Treasuries and adds +0.050 a year.
  - Tightening the −100bp limit to zero then costs 0.0004 instead of 0.0213.
- **Main tail risk: downgrades.** A one-notch BBB move pushes NAIC 3+ holdings over the HY limit, and the capital buffer covers capital but not that count.

**Quant (2 minutes).**

- **Pricing.** Spot-curve discounting at z(t) + s with annual compounding. Effective measures come from repricing with re-projected cash flows. KRDs use triangular 1bp bumps that form a partition of unity, so they sum to effective duration.
- **Prepayment.** Agency MBS use a centered S-curve in the 10Y rate. That gives effective convexity of -216 for FNMA, against 77 with constant CPR.
- **The LP.** Every constraint is a linear row a·x ≤ b, so value, slack and dual come out uniformly.
  - The funding test stays linear through a lower-triangular compounding matrix.
  - Scenario rows use full-repricing ΔP per dollar, which is exact per scenario. A test checks them against the stress engine.
  - Duals are checked against finite-difference re-solves.
- **Robustness.** 500 draws perturb spreads and PDs.
  - Re-optimized income averages 1.251 in-sample, versus 1.237 for the fixed base portfolio on the same draws, and 1.230 when each re-optimized portfolio is scored on an independent draw.
  - That is the optimizer's curse, measured.

## 15 likely technical questions

**1. Why discount on a spot curve instead of one yield?**
Each cash flow has its own maturity, so it should be discounted at its own spot rate. YTM is just the IRR that matches the price. I use it for Macaulay and modified duration, but all matching uses curve-based dollar measures, because the yield-based and curve-based numbers differ on a non-flat curve.

**2. Modified vs effective duration: when do they differ?**
Modified duration assumes fixed cash flows and one yield. Effective duration reprices on a shifted curve and re-projects cash flows. For Agency MBS with rate-dependent prepayment, FNMA's effective duration is 6.91, against 6.48 with constant CPR.

**3. How do you compute KRDs, and why do they sum to effective duration?**
Each key tenor gets a triangular 1bp bump on the zero curve: 1 at its tenor, falling linearly to 0 at its neighbors, with flat ends. The bumps add up to a parallel shift at every maturity, so the KRDs add up to the parallel (effective) duration. A unit test checks this to within one percent.

**4. Why does a CLO floater have near-zero rate duration but a large spread duration?**
The coupon resets to the forward rate, so only the next fixed coupon is exposed to rates (rate duration below 1). The discount margin is fixed for the life of the bond, so a spread move hits every cash flow. Spread duration equals the modified duration of the projected cash flows, which is below WAL because coupons arrive earlier than principal.

**5. Duration-matched but still exposed: example?**
<!-- static --> Textbook barbell: 62.5 in a 2Y zero and 37.5 in a 10Y zero vs a 5Y zero liability, all at a flat 4%. Both sides have modified duration 4.808, but the KRD gaps are +1.20 (2Y), −4.81 (5Y) and +3.61 (10Y), so a 2Y −50 / 10Y +50 steepener loses about 1.20.
In the model, Portfolio A has a duration gap of only +0.53 years but a net key rate DV01 of +18.2k at 30Y.

**6. How is the cash flow funding test linear?**
With reinvestment at 1Y forwards and no borrowing, the cumulative balance is C = M·gap, where M is lower-triangular with compounding factors. Requiring C ≥ 0 rules out borrowing, so the recursion is exactly linear in holdings.

**7. What is the shadow price of capital, and how did you check it?**
It is the dual of the budget rows: extra net spread income per extra dollar of K, here 0.107. Because K also scales the downgrade buffer, it is dual_capital + 1.10 × dual_buffer. A test compares it to a central finite-difference re-solve. The frontier shows it falling as K grows and reaching zero once other limits bind.

**8. Why doesn't the LP buy more HY when HY earns more spread per unit of capital?**
The NAIC 3+ limit is binding (dual 0.0082). Pricing a BBB → BB+ swap at all binding duals (capital, HY room, rate rows, the BBB name's issuer-cap rent) gives a reduced cost of zero. The "0.55% spread for 1.63% capital" ratio ignores the other resources the trade uses.

**9. What do the 12/31/2026 CLO factors change?**
They lower the charge for AAA to A tranches and raise it sharply for BBB (3.240% vs 1.523%) and BB (25.150% vs 4.537%). C's holdings need 1.191 of capital under the new factors, versus 1.047 under the old ones. Re-optimized under the old factors, the LP would hold much more BB CLO.

**10. Why add scenario constraints on top of KRD limits?**
KRD rows are first-order and cannot see convexity. Full-repricing scenario rows are still linear in holdings and exact for each scenario. They moved C's −100bp result from -0.734 to -0.500 and the combined adverse result from -4.62 to -4.35 (B: -4.53). The cost is 0.0083 a year.

**11. What is negative convexity, and why does an insurer care?**
Homeowners refinance when rates fall, so MBS principal comes back at par just when the bond would be worth more. When rates rise, the MBS extends just when it is worth less. Both directions hurt relative to a linear estimate, which matters because the insurer's liabilities are long and positively convex.

**12. What is the optimizer's curse, and how did you measure it?**
Optimizing on noisy inputs picks the positions whose noise happened to be favorable, so in-sample income is biased upward. I compare three averages over 500 draws:

| | Mean income |
|---|---|
| Re-optimized, in-sample | 1.251 |
| Fixed base portfolio, same draws | 1.237 |
| Re-optimized, scored on an independent draw | 1.230 |

The practical conclusion: recommend the stable bucket mix, not the specific names.

**13. Why a book-value proxy for capital under stress?**
Statutory RBC applies factors to book / adjusted carrying value, which a pure market move does not change. Holding t0 value fixed means only downgrades move the charge. Mark-to-market capital is kept as a sensitivity.

**14. What does the swap overlay do, and what is missing?**
A receive-fixed swap is a long par bond minus a floater. It adds duration at its tenor without using cash or C-1 capital in this model. Here it replaces zero-spread Treasuries and adds +0.050 a year, and it makes tightening the −100bp limit almost free. Missing: counterparty and derivative RBC, collateral and margin liquidity, swap spreads, and hedge accounting.

**15. Biggest limitations, and what would you do next?**
- Synthetic market data and liabilities.
- No dynamic lapse; liabilities are fixed under rate moves.
- Annual time steps and bond-level credit losses instead of deal waterfalls.
- An illustrative prepayment S-curve rather than a fitted model.
- No RBC covariance or RBC ratio.

Next steps would be a real Treasury curve loader, dynamic lapse, an OAS / Monte Carlo rate model for MBS, and a CVaR objective instead of fixed scenario limits.
