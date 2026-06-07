# DeFi Yield Results

Generated on 2026-06-07 UTC.

## Screen

- Data source: DeFiLlama yield pools
- Filter: stablecoin, no IL risk flag, single exposure, non-outlier
- Minimum TVL: 10,000,000 USD
- Maximum APY: 30%

## Top Snapshot Rows

| chain | project | symbol | tvl_usd | apy | score |
| --- | --- | --- | ---: | ---: | ---: |
| Flare | mystic-finance-lending | COREUSDT0 | 19655935 | 13.6601 | 13.2717 |
| Ethereum | apyx-protocol | APXUSD | 191010974 | 12.4788 | 12.0411 |
| Ethereum | mainstreet | MSUSD | 81785890 | 11.9982 | 11.9978 |
| Ethereum | ember-protocol | USDC | 37630629 | 12.4619 | 11.7937 |
| Ethereum | re | REUSDE | 19689842 | 12.0003 | 11.7708 |

This lane is promising only if the next layer can explain why the yield exists
and whether the operational risks are acceptable. APY alone is not an edge.

## Yield Quality

The quality screen separates base APY from reward-heavy APY and penalizes large
30d APY deviation.

Current base-yield watches:

- `Ethereum/apyx-protocol APXUSD`: material base APY with large TVL.
- `Ethereum/mainstreet MSUSD`: material base APY with stable 30d context.
- `Arbitrum/usd-ai SUSDAI`: material base APY candidate.

Important caveat:

- Base yield is still not free alpha. Custody, smart-contract risk, issuer risk,
  withdrawal route, APY decay, capacity, gas, bridge cost, and exit liquidity
  must be checked before paper allocation.

## Yield Peg Risk Join

The peg-risk join connects stable-yield candidates to the stablecoin peg-stress
screen. This avoids treating high APY as carry alpha when peg, redemption, or
issuer risk may explain the yield.

Current conflicts:

- `Ethereum/apyx-protocol APXUSD`: high base APY overlaps with below-peg
  `apxUSD`, so it is an avoid-or-repeg research candidate rather than a clean
  allocation candidate.
- `Ethereum/ondo-yield-assets USDY`: yield overlaps with above-peg `USDY`, so
  premium reversion can erase carry.
- `Ethereum/mainstreet MSUSD`: high base APY overlaps with supply-stress watch,
  so redemption and issuer context come before allocation.
