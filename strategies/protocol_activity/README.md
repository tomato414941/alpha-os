# Protocol Activity

This lane watches developer and community activity as non-price context.

Run:

```bash
uv run python -m strategies.protocol_activity.current_coingecko_protocol_activity
uv run python -m strategies.protocol_activity.current_protocol_activity_market_join
uv run python -m strategies.protocol_activity.current_protocol_activity_forward_labels
```

This is not a trade instruction. Developer activity is usually a slower
fundamental context, not a 15m momentum signal.
