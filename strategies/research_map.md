# Strategy Research Map

The current direction is to grow profit-seeking strategy candidates before
promoting abstractions into `src/alpha_os`.

## Lanes

- Funding carry / basis: prioritize market-neutral or near-market-neutral
  sources before more directional prediction work.
- Daily close rotation: keep as a baseline, not the main edge source.
- Market structure: funding, premium, taker flow, order flow, open interest,
  and basis where historical data is available.
- Event flow and LOB: include trades, aggTrades, order book snapshots, and
  sequence models when the data path is available.
- Cross-exchange: include exchange-specific basis, fees, borrow, margin, and
  transfer constraints.
- Cross-asset context: crypto, equity index ETFs, duration, gold, and cash-like
  risk-off assets.
- Portfolio construction: allocation should improve a real signal, not rescue a
  weak one.
- Execution and costs: separate strategy edge from market and broker frictions.

## Anti-Constraints

Do not treat current local data, daily frequency, linear models, low compute,
or no-new-dependency work as strategy constraints. Those are convenience
constraints, not profit constraints.

The durable constraints are:

- no lookahead
- explicit costs, slippage, and execution feasibility
- measurable risk
- reproducibility
- trading strategy boundary remains `observation -> action`

## Current Priority

Crypto daily-close variants have not beaten clean buy-and-hold benchmarks except
for manual universe exclusions. Directional market-structure screens improved
results, but the more profit-adjacent next lane is funding carry / basis and
other market-neutral or near-market-neutral opportunities.
