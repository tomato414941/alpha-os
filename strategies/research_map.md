# Strategy Research Map

The current direction is to grow profit-seeking strategy candidates before
promoting abstractions into `src/alpha_os`.

## Lanes

- Daily close rotation: keep as a baseline, not the main edge source.
- Market structure: funding, premium, taker flow, order flow, open interest,
  and basis where historical data is available.
- Cross-asset context: crypto, equity index ETFs, duration, gold, and cash-like
  risk-off assets.
- Portfolio construction: allocation should improve a real signal, not rescue a
  weak one.
- Execution and costs: separate strategy edge from market and broker frictions.

## Current Priority

Crypto daily-close variants have not beaten clean buy-and-hold benchmarks except
for manual universe exclusions. The next broad step is market-structure data.
