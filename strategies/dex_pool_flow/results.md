# DEX Pool Flow Results

Generated on 2026-06-07 UTC.

Run:

```bash
uv run python -m strategies.dex_pool_flow.current_geckoterminal_pool_flow
```

Interpretation:

- DEX pool flow can reveal early liquidity, momentum, or reversal-risk events.
- This is not the same as centralized exchange momentum.
- High volume relative to reserves can be opportunity or danger.
- Any candidate needs route depth, slippage, gas, MEV, token-transfer
  restrictions, contract risk, and repeated flow checks.

## Current Candidates

- `Bountywork / SOL`: extreme turnover after a large 24h move; reversal-risk
  watch.
- `BOUTYWORK / SOL`: similar high-turnover reversal-risk watch.
- `$tupid / SOL`: short-term pool-flow momentum watch.

Important caveat:

- These are DEX pool-flow candidates, not deployable trades.
- Thin liquidity, token restrictions, routing failure, gas, MEV, and stale pool
  data can destroy the apparent edge.
