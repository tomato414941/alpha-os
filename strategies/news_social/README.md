# News And Social Attention

This lane looks for event, attention, and sentiment driven opportunities.

The first probe is intentionally broad and simple:

- Alternative.me Fear & Greed Index
- CoinGecko trending search coins

## Commands

```bash
uv run python -m strategies.news_social.current_attention_snapshot
uv run python -m strategies.news_social.current_attention_market_join
uv run python -m strategies.news_social.current_attention_forward_labels
uv run python -m strategies.news_social.current_news_event_screen
```

## Current Status

This is not yet a news strategy. It now has:

- a timestamped attention snapshot that can be joined to returns, funding,
  liquidity, and execution feasibility
- a current RSS headline event screen that classifies external catalysts and
  joins tradable symbols to current perp state

Both outputs are candidate-generation inputs only. They still need duplicate
source checks, leakage-safe timestamps, forward-return labels, and execution
feasibility checks.
