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
uv run python -m strategies.news_social.current_news_event_quality_gate
uv run python -m strategies.news_social.current_news_event_source_independence
```

## Current Status

This is not yet a news strategy. It now has:

- a timestamped attention snapshot that can be joined to returns, funding,
  liquidity, and execution feasibility
- a current RSS headline event screen that classifies external catalysts and
  joins tradable symbols to current perp state
- a current RSS headline forward-label check against Binance USD-M 1-minute
  returns where historical archives are available
- a news-event quality gate that groups labels by symbol/event/side and checks
  source diversity, repeat support, rejected labels, and pending archives
- a source-independence gate that separates independent stories from multiple
  outlets repeating the same story

These outputs are candidate-generation inputs only. They still need duplicate
source review, stricter leakage controls, longer OOS windows, and execution
feasibility checks.
