# Modern Alpha Stale Input Refresh - 2026-06-10 13:16 UTC

Purpose: refresh the stale-input rows from `current_modern_alpha_candidate_batch_20260609T2332Z` without waiting for unrelated validation windows.

## Result

Refreshed into actionable paper candidates:

- `wallet-eth-seed-flow-long`: wallet position is still open and profitable. Treat as a wallet-flow paper label, not a standalone long thesis.
- `wallet-zec-seed-flow-long`: wallet signal is fresh again, and ZEC also has a live OKX/HL funding dislocation. High crowding/path risk remains.
- `wallet-apt-seed-flow-long`: fresh enough for small paper labeling, but not for promotion on wallet evidence alone.
- `xvenue-zec-okx-hl-funding`: still visible. The edge is positive over 24h but very small over 8h after rough cost.
- `basis-btc-25sep26-rich-future`: still visible as a basis/carry candidate. This is not directional BTC alpha.

Refreshed but not actionable:

- `wallet-hype-seed-flow-short`: no open tradable position; keep only as context.
- `xvenue-stable-bybit-hl-funding`: spread remains interesting, but route/account feasibility is still unresolved.
- `basis-eth-12jun26-cheap-future`: still cheap, but near expiry and thin enough that costs can dominate.

Rejected from current stale set:

- `xvenue-brett-bybit-hl-funding`: did not appear in the refreshed outputs, so the stale row should not be scored.

New refresh lead:

- `0G` funding spread appeared strongly, but 8h net is negative and capacity is tiny. Watch only.

## Operating Read

This refresh did not find a broad new alpha family. It did clear the stale-input blocker on a small number of candidates. The useful next work is not more tracking infrastructure; it is to run fresh labels on the refreshed actionable candidates while continuing discovery in parallel.
