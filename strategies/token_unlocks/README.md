# Token Unlocks

This lane tracks scheduled token supply releases as event-driven context.

It is not a trade instruction. A token unlock can create sell pressure,
attention, hedging demand, or no tradable effect at all.

## Commands

```bash
uv run python -m strategies.token_unlocks.current_token_unlock_snapshot
uv run python -m strategies.token_unlocks.current_token_unlock_market_join
uv run python -m strategies.token_unlocks.current_token_unlock_paper_tickets
uv run python -m strategies.token_unlocks.current_token_unlock_actionability
```
