# Stablecoin Liquidity Results

Generated on 2026-06-07 UTC.

Run:

```bash
uv run python -m strategies.stablecoin_liquidity.current_supply_snapshot
```

Interpretation:

- supply expansion can proxy risk-on liquidity
- supply contraction can proxy liquidity withdrawal or capital rotation
- peg price deviations can indicate stress
- this is a current snapshot, not a causal model

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

