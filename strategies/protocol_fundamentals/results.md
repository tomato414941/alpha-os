# Protocol Fundamentals Results

## Current Status

Current protocol fee-growth candidates:

- `JUP`: Jupiter Perpetual Exchange fee growth accelerated and funding is not
  expensive.
- `MORPHO`: Morpho Blue lending fee growth accelerated.
- `HYPE`: Hyperliquid Perps has a large fee base and positive 7d-over-7d growth.
- `AAVE`: Aave V3 lending fees accelerated.
- `UNI`: Uniswap V3 fees accelerated.
- `ETH`: Ethereum chain fees accelerated, but this is a broad beta context.

Interpretation:

- Protocol fee growth is not a direct token valuation model.
- It is useful as a non-price input when it overlaps with tradable perps,
  funding, unlocks, attention, or sector flow.
- The next useful test is forward labeling: after fee-growth snapshots, do the
  mapped tokens outperform over 4h, 12h, 24h, and multi-day windows?

## Candidate Review

Current review against other sources:

- `JUP`: fee growth is strong, but the current sector label is slightly
  negative. Treat this as an early-or-lagging candidate, not confirmed momentum.
- `MORPHO`: strong fee growth, but no confirming short-horizon context yet.
- `AAVE`: fee growth plus long-carry discount context, but still unconfirmed.
- `UNI`: fee growth exists, but perp pressure is not clearly supportive.
- `ETH`: broad fee-growth context, not a focused token alpha by itself.
- `HYPE`: fee-growth long context conflicts with token-unlock short pressure.

Interpretation:

- `JUP` is the cleanest fee-growth follow-up because the thesis is clear and
  funding is not expensive.
- `HYPE` should not be averaged into one signal. The protocol-growth thesis and
  unlock-short thesis must be labeled separately.

## Fee Valuation

Current fee-yield candidates:

- `JUP`: high annualized fees versus market cap and FDV, with strong 7d fee
  growth.
- `AAVE`: high fee-to-market-cap and fee-to-FDV context, with strong 7d fee
  growth.
- `UNI`: material fee yield plus fee growth on Uniswap V3.
- `CRV`: fee yield and growth are both strong, but fee base is smaller.
- `MORPHO`: fee yield and growth are both strong.

Important caveat:

- This screen uses DeFiLlama fees, not a strict token-holder revenue model.
  Fee yield is valuation context, not proof that the token captures those fees.
- `HYPE` has strong protocol fees and growth, but its FDV makes fee-to-FDV weak
  in the current snapshot.

## Forward Labels

Current 4h protocol-fee thesis labels:

- `HYPE`: the fee-decay / weak-price short thesis has one supporting 4h label
  (`mean_directional_4h=0.01234720`). It is only a watch until another 4h
  repeat and execution context pass.
- `AAVE`, `JUP`, and `MORPHO`: the fee-growth / price-lag long thesis failed
  the current 4h labels and should be deprioritized until a fresh independent
  snapshot appears.
- `CRV`: one of three 4h labels was positive, but the mean label is still
  slightly negative and the current spread gate is weak.
- `UNI`, `PENDLE`, and `SOL`: labels are still pending because CoinGecko did not
  provide a usable 4h target in the current run.

Interpretation:

- Protocol fees are useful as a non-price input, but the current broad pass does
  not support blindly buying fee-growth lag candidates.
- The next useful work is to repeat the HYPE short thesis and refresh execution
  evidence, while keeping the failed long fee-growth candidates out of the
  promotion path until new data appears.
