# Strategy Exploration Board

This board tracks broad profit-source exploration. It is not a ranking of deployable strategies.

| lane | status | strongest current signal | main gap | next step |
| --- | --- | --- | --- | --- |
| crypto_market_structure | implemented_probe | spot_perp_positive_funding_top_3_14d: total=0.0596891489, sharpe=4.9676985675 | execution and borrow/margin feasibility remain shallow | stress fees, margin, and venue availability before treating carry as tradable |
| cross_exchange_funding | current_snapshot | STABLE: BybitPerp->BinPerp, annualized=2.80085670 | snapshot only; external venue execution and transfer constraints unknown | collect repeated snapshots and add venue-specific execution constraints |
| perp_market_map | current_snapshot | MANTA: ann_funding=3.23186009, volume=652165.69526300 | no history yet, so no persistence or PnL evidence | collect snapshots over time and test carry/crowding persistence |
| event_flow | implemented_probe | top_20 imbalance mean_next_return=-0.0000414307, hit_rate=0.500000 | tiny sample and naive label; no order book or liquidation context | extend sample window and add liquidation/funding-time labels |
| defi_yield | current_snapshot | Flare/mystic-finance-lending COREUSDT0: apy=13.660140, tvl=19655935.00 | risk, custody, exit liquidity, and APY decay not modeled | separate real yield from incentive yield and add operational risk checklist |
| market_making | not_started | none | needs L2 book, queue/fill model, and fee tier assumptions | probe reachable bookDepth/bookTicker history and define fill simulation |
| news_social | not_started | none | needs timestamped event source and labels | inventory public/paid event feeds and build one event-to-return label set |
| on_chain_flow | not_started | none | needs wallet/exchange-flow source and leakage-safe timestamps | inventory reachable on-chain/exchange-flow APIs |
