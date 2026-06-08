# Stablecoin Liquidity Results

## Chain Stablecoin Migration

The chain migration screen aggregates DeFiLlama stablecoin chain-circulating
data into chain-level capital-flow proxies.

Current notable rows:

- `Solana`: stablecoin inflow watch.
- `Base`: stablecoin inflow watch.
- `Polygon`: stablecoin outflow watch.

This is not a bridge-fill feed. It still needs bridge route checks, chain-token
mapping, venue coverage, DEX liquidity, and forward labels.

Generated on 2026-06-07 UTC.

Run:

```bash
uv run python -m strategies.stablecoin_liquidity.current_supply_snapshot
uv run python -m strategies.stablecoin_liquidity.current_chain_stablecoin_migration
uv run python -m strategies.stablecoin_liquidity.current_chain_stablecoin_migration_forward_labels
uv run python -m strategies.stablecoin_liquidity.current_supply_market_forward_labels
uv run python -m strategies.stablecoin_liquidity.current_peg_stress_screen
```

Interpretation:

- supply expansion can proxy risk-on liquidity
- supply contraction can proxy liquidity withdrawal or capital rotation
- peg price deviations can indicate stress
- this is a current snapshot, not a causal model

## Chain Migration Forward Labels

Current 4h chain-migration labels from the 2026-06-08T00:43 UTC observation:

- `Ethereum/ETH`: weekly stablecoin outflow / reversal context aligned with a
  4h ETH decline (`directional_return_4h=0.02702544`).
- `Arbitrum/ARB`: weekly stablecoin outflow context aligned with a 4h ARB
  decline (`directional_return_4h=0.02647341`).
- `Hyperliquid L1/HYPE`: weekly stablecoin outflow / reversal context aligned
  with a 4h HYPE decline (`directional_return_4h=0.02111262`).
- `Polygon/POL`: stablecoin outflow short context aligned with a 4h POL decline
  (`directional_return_4h=0.01461835`).
- `Solana/SOL`: the large stablecoin inflow long context was contradicted over
  4h (`directional_return_4h=-0.02024000`).

Interpretation:

- Chain-level stablecoin migration is more useful than the broad stablecoin
  supply aggregate for short-horizon labels in this sample.
- The SOL inflow candidate should not be promoted from the current observation.
- ETH, ARB, HYPE, and POL have only 4h support. They still need 12h labels,
  repeat snapshots, venue depth, funding, and execution costs.

## Snapshot

| symbol | name | current supply USD | week change USD | price |
| --- | --- | ---: | ---: | ---: |
| USDT | Tether | 186835318744 | -1515562340 | 0.999500 |
| USDC | USD Coin | 75488754565 | -398272074 | 0.999607 |
| USDS | Sky Dollar | 8580752527 | -247523326 | 0.999706 |
| PYUSD | PayPal USD | 2843312909 | -210073288 | 1.000139 |
| USDf | Falcon USD | 1297428576 | -204957476 | 0.996244 |

The largest visible weekly changes are contractions in major stablecoins. That
may be risk-off liquidity context, but it must be joined to market returns and
funding regimes before it becomes useful.

## Market Forward Labels

| asset | week change USD | week change % | expected dir | raw 1h | dir 1h | raw 4h | dir 4h | raw 12h | dir 12h | action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BTC | -2978915790 | -0.009988 | -1 | 0.002909 | -0.002909 | 0.015877 | -0.015877 | 0.010092 | -0.010092 | liquidity_direction_contradicted |
| ETH | -2978915790 | -0.009988 | -1 | 0.010529 | -0.010529 | 0.019797 | -0.019797 | 0.026102 | -0.026102 | liquidity_direction_contradicted |
| SOL | -2978915790 | -0.009988 | -1 | 0.012956 | -0.012956 | 0.023183 | -0.023183 | 0.025266 | -0.025266 | liquidity_direction_contradicted |
| HYPE | -2978915790 | -0.009988 | -1 | 0.021014 | -0.021014 | 0.039408 | -0.039408 | 0.007137 | -0.007137 | liquidity_direction_contradicted |
| BASKET | -2978915790 | -0.009988 | -1 | 0.011852 | -0.011852 | 0.024566 | -0.024566 | 0.017149 | -0.017149 | liquidity_direction_contradicted |

Interpretation:

- Major stablecoin supply contracted by roughly `$2.98B` week-over-week in the
  current snapshot.
- A naive liquidity rule would expect risk-off returns, but BTC/ETH/SOL/HYPE
  rose over the following 4h and 12h.
- This weakens stablecoin supply as a direct short-term trade direction and
  makes it more useful as a regime or divergence feature to combine with other
  sources.

## Peg Stress

Current depeg/repeg and premium mean-reversion watches:

- `pmUSD`: large below-peg deviation.
- `USYC`: large above-peg premium.
- `USDY`: large above-peg premium.
- `reUSD`: large above-peg premium.
- `apxUSD`: large below-peg deviation.
- `DOLA`: moderate below-peg deviation.

Important caveat:

- Stablecoin price data can be stale, thin, or not accessible on a usable venue.
- These candidates require redemption route, exchange depth, custody, issuer
  risk, and repeated peg snapshots before paper action.
