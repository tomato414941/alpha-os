# Current Yield Peg Risk Join

This joins stable-yield candidates with stablecoin peg stress. It is a cross-lane risk screen, not a trade instruction.

| chain | project | symbol | status | apy | base | tvl USD | peg symbol | peg status | price | peg deviation | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| Ethereum | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 1103651539 | USDY | paper_premium_mean_reversion_watch | 1.135572 | 0.135572 | 53.6800 | yield asset trades above peg, so carry can be offset by premium reversion |
| Stellar | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 525500511 | USDY | paper_premium_mean_reversion_watch | 1.135572 | 0.135572 | 53.6800 | yield asset trades above peg, so carry can be offset by premium reversion |
| BSC | circle-usyc | USYC | paper_yield_premium_conflict_watch | 3.2387 | 3.2387 | 2828928672 | USYC | paper_premium_mean_reversion_watch | 1.127497 | 0.127497 | 53.1151 | yield asset trades above peg, so carry can be offset by premium reversion |
| Ethereum | re | REUSD | paper_yield_premium_conflict_watch | 6.7440 | 6.7440 | 164178619 | reUSD | paper_premium_mean_reversion_watch | 1.081645 | 0.081645 | 51.4625 | yield asset trades above peg, so carry can be offset by premium reversion |
| Sei | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 256841301 | USDY | paper_premium_mean_reversion_watch | 1.135572 | 0.135572 | 48.8168 | yield asset trades above peg, so carry can be offset by premium reversion |
| Solana | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 181136169 | USDY | paper_premium_mean_reversion_watch | 1.135572 | 0.135572 | 47.3027 | yield asset trades above peg, so carry can be offset by premium reversion |
| Ethereum | apyx-protocol | APXUSD | paper_yield_depeg_conflict_watch | 12.5526 | 12.5526 | 197127718 | apxUSD | paper_depeg_repeg_watch | 0.963584 | -0.036416 | 45.0788 | yield may be compensation for below-peg, redemption, or issuer risk |
| Ethereum | circle-usyc | USYC | paper_yield_premium_conflict_watch | 3.2387 | 3.2387 | 77947683 | USYC | paper_premium_mean_reversion_watch | 1.127497 | 0.127497 | 44.6741 | yield asset trades above peg, so carry can be offset by premium reversion |
| Mantle | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 29407251 | USDY | paper_premium_mean_reversion_watch | 1.135572 | 0.135572 | 44.2681 | yield asset trades above peg, so carry can be offset by premium reversion |
| Sui | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 23185008 | USDY | paper_premium_mean_reversion_watch | 1.135572 | 0.135572 | 44.1437 | yield asset trades above peg, so carry can be offset by premium reversion |
| Noble | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 14991180 | USDY | paper_premium_mean_reversion_watch | 1.135572 | 0.135572 | 43.9798 | yield asset trades above peg, so carry can be offset by premium reversion |
| Ethereum | mainstreet | MSUSD | yield_supply_stress_watch | 12.0000 | 12.0000 | 81845813 | MSUSD | peg_supply_stress_watch | 0.996843 | -0.003157 | 21.4025 | yield asset has material supply stress even though price is near peg |
| Ethereum | ember-protocol | USDC | paper_yield_without_peg_stress_watch | 12.4619 | 12.4619 | 37627564 | USDC | watch | 0.999680 | -0.000320 | 19.1861 | yield candidate has no current material peg stress in the peg screen |
| Ethereum | re | REUSDE | yield_context_watch | 12.0003 | 12.0003 | 19696309 | - | - | 0.000000 | 0.000000 | 19.0738 | yield is context but peg linkage is not yet actionable |
| Arbitrum | usd-ai | SUSDAI | yield_supply_stress_watch | 8.0000 | 8.0000 | 289606526 | USDai | peg_supply_stress_watch | 0.998505 | -0.001495 | 18.5483 | yield asset has material supply stress even though price is near peg |
| Ethereum | morpho-blue | FAUSDE | yield_context_watch | 11.0022 | 11.0022 | 16501018 | - | - | 0.000000 | 0.000000 | 17.4204 | yield is context but peg linkage is not yet actionable |
| Ethereum | maple | USDC | paper_yield_without_peg_stress_watch | 4.6441 | 4.6441 | 3156446647 | USDC | watch | 0.999680 | -0.000320 | 17.3545 | yield candidate has no current material peg stress in the peg screen |
| Ethereum | ethena-usde | SUSDE | yield_supply_stress_watch | 4.5062 | 4.5062 | 1773945044 | USDe | peg_supply_stress_watch | 0.999239 | -0.000761 | 17.2231 | yield asset has material supply stress even though price is near peg |
| BSC | unitas | SUSDU | yield_supply_stress_watch | 9.9579 | 9.9579 | 43500091 | USDU | peg_supply_stress_watch | 0.998806 | -0.001194 | 17.1674 | yield asset has material supply stress even though price is near peg |
| Ethereum | goldfinch | USDC | paper_yield_without_peg_stress_watch | 10.1234 | 10.1234 | 36713347 | USDC | watch | 0.999680 | -0.000320 | 16.8977 | yield candidate has no current material peg stress in the peg screen |
