# News And Social Attention

This lane looks for event, attention, and sentiment driven opportunities.

The first probe is intentionally broad and simple:

- Alternative.me Fear & Greed Index
- CoinGecko trending search coins

## Commands

```bash
uv run python -m strategies.news_social.current_attention_snapshot
```

## Current Status

This is not yet a news strategy. It does not read headlines or social posts.
It only creates a timestamped attention snapshot that can later be joined to
returns, funding, liquidity, and execution feasibility.

