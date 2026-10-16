<!-- GENERATED FILE: edit docs/templates/investment_committee_summary.md.tmpl, then run scripts/run_all.py. -->
# Investment Committee Summary: Fixed Income Model Portfolio

*Hypothetical U.S. life insurer. Illustrative, synthetic data. Educational model, not a statutory or RBC result. Figures come from `outputs/metrics.json`; amounts are in USD millions.*

**Recommendation: adopt Portfolio C (Optimized ALM) as the model portfolio for the $100M book backing $90M of liabilities (PV), with its scenario loss limits and downgrade buffer as standing policy, and a decision on the swap overlay.**

## Why C

| | A Conservative IG | B Yield seeking | **C Optimized ALM** |
|---|---|---|---|
| Net spread income (after expected loss), per year | 0.612 | 1.266 | **1.239** |
| Expected credit loss, per year | 0.024 | 0.143 | 0.094 |
| Modeled C-1 charge vs $1.20M budget | 0.435 (36%) | **1.679 (140%)** | 1.191 (99%) |
| Surplus DV01 ($k per bp; limit ± 5) | +5.3 | **-5.5** | -3.0 |
| Key rate tenors outside tolerance | **3** | **2** | 0 |
| Rate limits met (surplus DV01 and every KRD tenor) | **False** | **False** | True |
| Bucket and issuer limits met | True | True | True |

- **C earns 2.0x A's spread income** (+0.628 a year) and is the only portfolio that meets every policy limit at once.
- **B earns only 27k a year more than C** (USD), but needs 40% more capital than the budget and breaks the rate-risk limits. Of B's extra 76k of gross spread over C, 49k is lost to higher expected credit losses.
- **The shadow price of the capital budget is 0.107**: an extra dollar of budget adds about that much net spread a year (counting both the capital row and the downgrade buffer). Income rises from 0.893 at K = $0.40M to 1.332 once capital used reaches $2.22M. It stays flat after that, because the HY, CLO and issuer limits bind.

## What C gives up

- **C is still short duration, but within a loss limit.** Spread-bearing assets here are shorter than the liabilities, and long Treasuries carry zero spread. So the optimizer runs a short position (surplus DV01 -3.0k per bp) up to the −100bp scenario limit of 0.50.
  - A 100bp fall changes C's surplus by **-0.50**, against +0.56 for A. A 100bp rise changes it by +0.07.
  - Holding that limit costs 0.0083 a year of income, versus an unconstrained C at -0.73.
- **C has more credit and spread risk.** NAIC 3+ spreads +200bp change C's surplus by -0.91, against +0.00 for A and -0.94 for B. C holds 10.0 of NAIC 3+ assets (including CLO BB), 10.0 of CLOs and 25.0 of non-agency structured.
- **Capital is sized to a downgrade, not just to today.** Budget utilization is 99%. The binding row is the downgrade buffer: after a one-notch downgrade of every BBB-family holding, capital is 1.320, against a limit of 1.32. The buffer costs 0.0009 a year.
- **The same downgrade breaks the HY limit.** It lifts NAIC 3+ holdings to 15.0 (limit 10), because BBB− becomes 3.A. The buffer controls capital, not the HY count.
- **Negative convexity.** C holds 19.5 of Agency MBS, with effective convexity of about -216 (FNMA). A ±100bp move changes surplus by +0.07 / -0.50, against +0.30 / -0.30 on a linear estimate.
- **Combined adverse stress** (−100bp, wider spreads, BBB downgrades): C -4.35 (it was -4.62 before the scenario limits), B -4.53, A -2.04.
- **C has the liquidity A lacks.** With surrenders +50% in years 1–5, C's minimum funding balance stays at +1.34, while A's falls to -0.55.

## How robust is C

500 of 500 perturbed re-optimizations were feasible (spreads × N(1, 0.10), PDs × lognormal, σ = 0.25).

| Mean net spread income across draws | |
|---|---|
| Base C held fixed | 1.237 (5th–95th percentile: 1.178–1.298) |
| Re-optimized on each draw (in-sample) | 1.251 |
| Re-optimized, evaluated on a fresh draw (out-of-sample) | 1.230 |

- The in-sample gain (+0.0136) is the **optimizer's curse**. Out-of-sample, chasing each draw's noise is worse than holding C (-0.0077).
- Bucket weights are stable: HY runs from 9.3 to 10.0, and CLO, CMBS and RMBS sit at their limits. Individual names are not stable: the Agency MBS issuer and the second HY name change from draw to draw.
- Treat the bucket mix as the recommendation and name selection as an execution decision.

## Swap overlay result

- Receive-fixed swaps (6.8 at 20Y and 6.2 at 30Y notional) replace 8.3 of zero-spread Treasuries / STRIPS and add +0.050 a year.
- The steepener result improves from -0.23 to -0.03.
- **The overlay makes −100bp protection nearly free.** Tightening the −100bp loss limit from 0.50 to 0.00 costs 0.0213 a year with cash bonds only, but 0.0004 with the overlay.
- Not modeled: counterparty RBC, collateral, and margin.

## Asks for the committee

1. Approve the Portfolio C bucket mix as the target allocation.
2. Adopt as policy the scenario surplus loss limit (0.50 per curve scenario) and the downgrade buffer (1.10 × K). Add a BBB− cap if the HY-limit breach on downgrade is not acceptable.
3. Approve the swap overlay once the collateral and counterparty framework is in place, and then tighten the −100bp loss limit, which the overlay makes almost costless.
