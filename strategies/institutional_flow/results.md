# Institutional Flow Results

This lane was added to cover slower institutional demand, starting with Bitcoin
spot ETF flows.

Initial result:

- Bitbo exposes daily BTC ETF flow history in page data.
- The page data must be aggregated by date because the embedded rows include
  same-day issuer-level entries.
- BTC ETF flow was labeled from the day after the flow date to avoid using
  same-day price movement.
- Labeled observations: 555
- Mean directional 1d: 0.00233183
- Mean directional 3d: 0.00742992
- Mean directional 5d: 0.01268852
- 5d hit rate: 0.5730

Current read:

- ETF inflow as long BTC context and ETF outflow as short BTC context has
  positive coarse 5-day directional behavior in this panel.
- This is not deployable PnL. It excludes funding PnL, intraday timing,
  transaction costs, and regime splits.
- The next useful test is whether ETF flow improves after splitting by perp
  funding alignment/divergence and large-flow thresholds.
