# News And Social Attention

This lane looks for event, attention, and sentiment driven opportunities.

The first probe is intentionally broad and simple:

- Alternative.me Fear & Greed Index
- CoinGecko trending search coins

## Commands

```bash
uv run python -m strategies.news_social.current_attention_snapshot
uv run python -m strategies.news_social.current_attention_market_join
uv run python -m strategies.news_social.current_attention_price_context
uv run python -m strategies.news_social.current_attention_price_history
uv run python -m strategies.news_social.current_attention_price_labels
uv run python -m strategies.news_social.current_attention_forward_labels
uv run python -m strategies.news_social.current_news_event_screen
uv run python -m strategies.news_social.current_news_event_forward_labels
```

## Current Status

This is not yet a news strategy. It now has:

- a timestamped attention snapshot that can be joined to returns, funding,
  liquidity, and execution feasibility
- a current RSS headline event screen that classifies external catalysts and
  joins tradable symbols to current perp state
- a current RSS headline forward-label check against Binance USD-M 1-minute
  returns where historical archives are available

These outputs are candidate-generation inputs only. They still need duplicate
source checks, stricter leakage controls, repeated labels, and execution
feasibility checks.
