# DeFi Lending Results

Generated on 2026-06-07 UTC.

## Current Morpho Lending Rates

This screen uses the Morpho public GraphQL API to inspect current supply,
borrow, liquidity, utilization, and average APYs.

Current observations:

- Several large isolated markets are fully utilized with little or no remaining
  liquidity. These are borrow/liquidity stress watches, not clean lend signals.
- The most actionable interpretation is not "deposit into the highest APY".
  The useful signal is that borrow demand or collateral stress may be visible
  before it appears in price, funding, or liquidation screens.
- Direct lending candidates require rate persistence, collateral drawdown,
  oracle, liquidation, withdrawal, gas, and smart-contract checks.

