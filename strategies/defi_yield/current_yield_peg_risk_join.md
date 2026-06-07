# Current Yield Peg Risk Join

This joins stable-yield candidates with stablecoin peg stress. It is a cross-lane risk screen, not a trade instruction.

| chain | project | symbol | status | apy | base | tvl USD | peg symbol | peg status | price | peg deviation | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| Ethereum | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 1098537594 | USDY | paper_premium_mean_reversion_watch | 1.126703 | 0.126703 | 53.6800 | yield asset trades above peg, so carry can be offset by premium reversion |
| Stellar | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 525450288 | USDY | paper_premium_mean_reversion_watch | 1.126703 | 0.126703 | 53.6800 | yield asset trades above peg, so carry can be offset by premium reversion |
| Ethereum | apyx-protocol | APXUSD | paper_yield_depeg_conflict_watch | 12.5040 | 12.5040 | 195189741 | apxUSD | paper_depeg_repeg_watch | 0.950415 | -0.049585 | 52.8634 | yield may be compensation for below-peg, redemption, or issuer risk |
| Ethereum | re | REUSD | paper_yield_premium_conflict_watch | 6.6113 | 6.6113 | 163854772 | reUSD | paper_premium_mean_reversion_watch | 1.081412 | 0.081412 | 51.2962 | yield asset trades above peg, so carry can be offset by premium reversion |
| Sei | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 256816858 | USDY | paper_premium_mean_reversion_watch | 1.126703 | 0.126703 | 48.8163 | yield asset trades above peg, so carry can be offset by premium reversion |
| Solana | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 181135834 | USDY | paper_premium_mean_reversion_watch | 1.126703 | 0.126703 | 47.3027 | yield asset trades above peg, so carry can be offset by premium reversion |
| Mantle | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 29404440 | USDY | paper_premium_mean_reversion_watch | 1.126703 | 0.126703 | 44.2681 | yield asset trades above peg, so carry can be offset by premium reversion |
| Sui | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 23182793 | USDY | paper_premium_mean_reversion_watch | 1.126703 | 0.126703 | 44.1437 | yield asset trades above peg, so carry can be offset by premium reversion |
| Noble | ondo-yield-assets | USDY | paper_yield_premium_conflict_watch | 3.5500 | 3.5500 | 14989747 | USDY | paper_premium_mean_reversion_watch | 1.126703 | 0.126703 | 43.9798 | yield asset trades above peg, so carry can be offset by premium reversion |
| Ethereum | mainstreet | MSUSD | yield_supply_stress_watch | 12.0053 | 12.0053 | 81815845 | MSUSD | peg_supply_stress_watch | 0.999608 | -0.000392 | 21.1308 | yield asset has material supply stress even though price is near peg |
| Ethereum | ember-protocol | USDC | paper_yield_without_peg_stress_watch | 12.4619 | 12.4619 | 37629815 | USDC | watch | 0.999557 | -0.000443 | 19.1325 | yield candidate has no current material peg stress in the peg screen |
| Ethereum | re | REUSDE | yield_context_watch | 12.0003 | 12.0003 | 19689842 | - | - | 0.000000 | 0.000000 | 19.0600 | yield is context but peg linkage is not yet actionable |
| Arbitrum | usd-ai | SUSDAI | yield_supply_stress_watch | 8.0000 | 8.0000 | 289502859 | USDai | peg_supply_stress_watch | 0.999870 | -0.000130 | 18.3983 | yield asset has material supply stress even though price is near peg |
| Ethereum | morpho-blue | FAUSDE | yield_context_watch | 11.0022 | 11.0022 | 16338561 | - | - | 0.000000 | 0.000000 | 17.3802 | yield is context but peg linkage is not yet actionable |
| Ethereum | maple | USDC | paper_yield_without_peg_stress_watch | 4.6353 | 4.6353 | 3080692305 | USDC | watch | 0.999557 | -0.000443 | 17.3240 | yield candidate has no current material peg stress in the peg screen |
| BSC | unitas | SUSDU | yield_supply_stress_watch | 9.9561 | 9.9561 | 43518191 | USDU | peg_supply_stress_watch | 0.998421 | -0.001579 | 17.2043 | yield asset has material supply stress even though price is near peg |
| Ethereum | ethena-usde | SUSDE | yield_supply_stress_watch | 4.4768 | 4.4768 | 1785366278 | USDe | peg_supply_stress_watch | 0.999418 | -0.000582 | 17.1615 | yield asset has material supply stress even though price is near peg |
| Ethereum | goldfinch | USDC | paper_yield_without_peg_stress_watch | 10.1234 | 10.1234 | 36715543 | USDC | watch | 0.999557 | -0.000443 | 16.8966 | yield candidate has no current material peg stress in the peg screen |
| Ethereum | maple | USDT | paper_yield_without_peg_stress_watch | 4.1157 | 4.1157 | 833430670 | USDT | watch | 0.999508 | -0.000492 | 16.5065 | yield candidate has no current material peg stress in the peg screen |
| BSC | bitway-earn | USDT | paper_yield_without_peg_stress_watch | 10.0000 | 10.0000 | 23677304 | USDT | watch | 0.999508 | -0.000492 | 16.2426 | yield candidate has no current material peg stress in the peg screen |
