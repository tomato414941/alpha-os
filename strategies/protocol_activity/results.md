# Protocol Activity Results

Run:

```bash
uv run python -m strategies.protocol_activity.current_coingecko_protocol_activity
uv run python -m strategies.protocol_activity.current_protocol_activity_market_join
uv run python -m strategies.protocol_activity.current_protocol_activity_forward_labels
```

This lane uses current CoinGecko attention/category candidates, maps known
protocols to public GitHub repositories, and joins developer activity to
Hyperliquid perp context.

## Current Activity

Current high-activity rows:

- `ZEC`: active Zcash repo, strong source context from trending/category.
- `SUI`: active Sui repo and current attention context.
- `BTC`: active Bitcoin repo and current trending context.
- `APT`: active Aptos repo and current trending context.
- `NEAR`: active NEAR repo and current trending/network-event context.
- `TON`: active TON repo and current trending context.

Interpretation:

- This is a non-price input and broadens the project beyond price/funding/event
  feeds.
- Developer activity is not alpha by itself. It should be used as context for
  longer-horizon filters or combined with funding, listings, sector rotation,
  or on-chain flow.

## Market Join

Current market-joined rows show:

- `ZEC` overlaps protocol activity and material current funding state.
- `SUI`, `BTC`, `APT`, `NEAR`, and `TON` are tradable activity-context rows.

Interpretation:

- `ZEC` is the strongest protocol-activity/funding overlap, but other sector and
  short-horizon labels are currently weak.
- `NEAR` has a cleaner event/protocol overlap than most rows because the
  exchange-catalyst lane also has a positive network-event label.

## Forward Labels

The current 15m labels are weak across the activity rows:

- `ZEC`, `SUI`, `BTC`, `APT`, `NEAR`, and `TON` were all negative over the
  first 15m label window.

Interpretation:

- This does not fully falsify protocol activity because the input is slower than
  a 15m return label.
- The next useful test is longer horizon labeling and overlap checks with
  exchange catalysts, funding/carry, and sector rotation.
