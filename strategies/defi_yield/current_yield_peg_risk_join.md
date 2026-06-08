# Current Yield Peg Risk Join

This joins stable-yield candidates with stablecoin peg stress. It is a cross-lane risk screen, not a trade instruction.

| chain | project | symbol | status | apy | base | tvl USD | peg symbol | peg status | price | peg deviation | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| Ethereum | apyx-protocol | APXUSD | paper_yield_depeg_conflict_watch | 12.5457 | 12.5457 | 193226290 | apxUSD | paper_depeg_repeg_watch | 0.943997 | -0.056003 | 56.7390 | yield may be compensation for below-peg, redemption, or issuer risk |
| Ethereum | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 1103651539 | USDY | paper_premium_mean_reversion_watch | 1.132460 | 0.132460 | 53.6800 | yield asset trades above peg, so carry can be offset by premium reversion |
| Stellar | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 525500511 | USDY | paper_premium_mean_reversion_watch | 1.132460 | 0.132460 | 53.6800 | yield asset trades above peg, so carry can be offset by premium reversion |
| BSC | circle-usyc | USYC | paper_yield_premium_conflict_watch | 3.2387 | 3.2387 | 2828928672 | USYC | paper_premium_mean_reversion_watch | 1.127497 | 0.127497 | 53.1144 | yield asset trades above peg, so carry can be offset by premium reversion |
| Ethereum | re | REUSD | paper_yield_premium_conflict_watch | 6.7440 | 6.7440 | 164178619 | reUSD | paper_premium_mean_reversion_watch | 1.081505 | 0.081505 | 51.4515 | yield asset trades above peg, so carry can be offset by premium reversion |
| Sei | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 256841301 | USDY | paper_premium_mean_reversion_watch | 1.132460 | 0.132460 | 48.8168 | yield asset trades above peg, so carry can be offset by premium reversion |
| Solana | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 181136169 | USDY | paper_premium_mean_reversion_watch | 1.132460 | 0.132460 | 47.3027 | yield asset trades above peg, so carry can be offset by premium reversion |
| Ethereum | circle-usyc | USYC | paper_yield_premium_conflict_watch | 3.2387 | 3.2387 | 77947683 | USYC | paper_premium_mean_reversion_watch | 1.127497 | 0.127497 | 44.6733 | yield asset trades above peg, so carry can be offset by premium reversion |
| Mantle | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 29407251 | USDY | paper_premium_mean_reversion_watch | 1.132460 | 0.132460 | 44.2681 | yield asset trades above peg, so carry can be offset by premium reversion |
| Sui | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 23185008 | USDY | paper_premium_mean_reversion_watch | 1.132460 | 0.132460 | 44.1437 | yield asset trades above peg, so carry can be offset by premium reversion |
| Noble | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 14991180 | USDY | paper_premium_mean_reversion_watch | 1.132460 | 0.132460 | 43.9798 | yield asset trades above peg, so carry can be offset by premium reversion |
| Arbitrum | usd-ai | SUSDAI | paper_yield_premium_conflict_watch | 8.0000 | 8.0000 | 289600286 | USDai | paper_premium_mean_reversion_watch | 1.008595 | 0.008595 | 23.5534 | yield asset trades above peg, so carry can be offset by premium reversion |
| Ethereum | mainstreet | MSUSD | yield_supply_stress_watch | 12.0000 | 12.0000 | 81845813 | MSUSD | peg_supply_stress_watch | 0.999355 | -0.000645 | 21.1514 | yield asset has material supply stress even though price is near peg |
| Arbitrum | pendle | SUSDAI | paper_yield_premium_conflict_watch | 9.6078 | 9.6078 | 12284812 | USDai | paper_premium_mean_reversion_watch | 1.008595 | 0.008595 | 21.0110 | yield asset trades above peg, so carry can be offset by premium reversion |
| Ethereum | ember-protocol | USDC | paper_yield_without_peg_stress_watch | 12.4619 | 12.4619 | 37638707 | USDC | watch | 0.999674 | -0.000326 | 19.1749 | yield candidate has no current material peg stress in the peg screen |
| Ethereum | re | REUSDE | yield_context_watch | 12.0003 | 12.0003 | 19696309 | - | - | 0.000000 | 0.000000 | 19.0601 | yield is context but peg linkage is not yet actionable |
| Arbitrum | pendle | SUSDAI | paper_yield_premium_conflict_watch | 8.2673 | 8.0981 | 12284812 | USDai | paper_premium_mean_reversion_watch | 1.008595 | 0.008595 | 18.5052 | yield asset trades above peg, so carry can be offset by premium reversion |
| Arbitrum | pendle | USDAI | paper_yield_premium_conflict_watch | 7.2929 | 7.2929 | 35302888 | USDai | paper_premium_mean_reversion_watch | 1.008595 | 0.008595 | 17.5190 | yield asset trades above peg, so carry can be offset by premium reversion |
| Ethereum | morpho-blue | FAUSDE | yield_context_watch | 11.0022 | 11.0022 | 16498609 | - | - | 0.000000 | 0.000000 | 17.4132 | yield is context but peg linkage is not yet actionable |
| Ethereum | maple | USDC | paper_yield_without_peg_stress_watch | 4.6441 | 4.6441 | 3146160051 | USDC | watch | 0.999674 | -0.000326 | 17.3520 | yield candidate has no current material peg stress in the peg screen |
