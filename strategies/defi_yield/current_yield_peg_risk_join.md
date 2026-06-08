# Current Yield Peg Risk Join

This joins stable-yield candidates with stablecoin peg stress. It is a cross-lane risk screen, not a trade instruction.

| chain | project | symbol | status | apy | base | tvl USD | peg symbol | peg status | price | peg deviation | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| Ethereum | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 1103651539 | USDY | paper_premium_mean_reversion_watch | 1.128610 | 0.128610 | 53.6800 | yield asset trades above peg, so carry can be offset by premium reversion |
| Stellar | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 525500511 | USDY | paper_premium_mean_reversion_watch | 1.128610 | 0.128610 | 53.6800 | yield asset trades above peg, so carry can be offset by premium reversion |
| Ethereum | re | REUSD | paper_yield_premium_conflict_watch | 6.7440 | 6.7440 | 164178619 | reUSD | paper_premium_mean_reversion_watch | 1.081349 | 0.081349 | 51.4496 | yield asset trades above peg, so carry can be offset by premium reversion |
| Ethereum | apyx-protocol | APXUSD | paper_yield_depeg_conflict_watch | 12.5436 | 12.5436 | 195328294 | apxUSD | paper_depeg_repeg_watch | 0.954108 | -0.045892 | 50.7096 | yield may be compensation for below-peg, redemption, or issuer risk |
| Sei | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 256841301 | USDY | paper_premium_mean_reversion_watch | 1.128610 | 0.128610 | 48.8168 | yield asset trades above peg, so carry can be offset by premium reversion |
| Solana | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 181136169 | USDY | paper_premium_mean_reversion_watch | 1.128610 | 0.128610 | 47.3027 | yield asset trades above peg, so carry can be offset by premium reversion |
| Mantle | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 29407251 | USDY | paper_premium_mean_reversion_watch | 1.128610 | 0.128610 | 44.2681 | yield asset trades above peg, so carry can be offset by premium reversion |
| Sui | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 23185008 | USDY | paper_premium_mean_reversion_watch | 1.128610 | 0.128610 | 44.1437 | yield asset trades above peg, so carry can be offset by premium reversion |
| Noble | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 14991180 | USDY | paper_premium_mean_reversion_watch | 1.128610 | 0.128610 | 43.9798 | yield asset trades above peg, so carry can be offset by premium reversion |
| Ethereum | mainstreet | MSUSD | yield_supply_stress_watch | 12.0000 | 12.0000 | 81845813 | MSUSD | peg_supply_stress_watch | 0.999617 | -0.000383 | 21.1252 | yield asset has material supply stress even though price is near peg |
| Ethereum | ember-protocol | USDC | paper_yield_without_peg_stress_watch | 12.4619 | 12.4619 | 37630835 | USDC | watch | 0.999692 | -0.000308 | 19.1702 | yield candidate has no current material peg stress in the peg screen |
| Ethereum | re | REUSDE | yield_context_watch | 12.0003 | 12.0003 | 19696309 | - | - | 0.000000 | 0.000000 | 19.0601 | yield is context but peg linkage is not yet actionable |
| Arbitrum | usd-ai | SUSDAI | yield_supply_stress_watch | 8.0000 | 8.0000 | 289508959 | USDai | peg_supply_stress_watch | 1.002935 | 0.002935 | 18.6869 | yield asset has material supply stress even though price is near peg |
| Ethereum | morpho-blue | FAUSDE | yield_context_watch | 11.0022 | 11.0022 | 16484781 | - | - | 0.000000 | 0.000000 | 17.4101 | yield is context but peg linkage is not yet actionable |
| Ethereum | maple | USDC | paper_yield_without_peg_stress_watch | 4.6441 | 4.6441 | 3140623415 | USDC | watch | 0.999692 | -0.000308 | 17.3511 | yield candidate has no current material peg stress in the peg screen |
| Ethereum | ethena-usde | SUSDE | yield_supply_stress_watch | 4.4944 | 4.4944 | 1778506504 | USDe | peg_supply_stress_watch | 0.999255 | -0.000745 | 17.2051 | yield asset has material supply stress even though price is near peg |
| BSC | unitas | SUSDU | yield_supply_stress_watch | 9.9572 | 9.9572 | 43504812 | USDU | peg_supply_stress_watch | 0.998686 | -0.001314 | 17.1789 | yield asset has material supply stress even though price is near peg |
| Ethereum | goldfinch | USDC | paper_yield_without_peg_stress_watch | 10.1234 | 10.1234 | 36716539 | USDC | watch | 0.999692 | -0.000308 | 16.8974 | yield candidate has no current material peg stress in the peg screen |
| Ethereum | centrifuge-protocol | USDS | paper_yield_without_peg_stress_watch | 4.7600 | 4.7600 | 865275344 | USDS | watch | 0.999518 | -0.000482 | 16.7489 | yield candidate has no current material peg stress in the peg screen |
| Hyperliquid L1 | harmonix-finance | USDC | paper_yield_without_peg_stress_watch | 10.1804 | 10.1804 | 17123039 | USDC | watch | 0.999692 | -0.000308 | 16.5998 | yield candidate has no current material peg stress in the peg screen |
