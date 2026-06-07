# Strategy Exploration Board

This board tracks broad profit-source exploration. It is not a ranking of deployable strategies.

| lane | status | strongest current signal | main gap | next step |
| --- | --- | --- | --- | --- |
| crypto_market_structure | implemented_probe | spot_perp_positive_funding_top_3_14d: total=0.0596891489, sharpe=4.9676985675 | execution and borrow/margin feasibility remain shallow | stress fees, margin, and venue availability before treating carry as tradable |
| cross_exchange_funding | current_snapshot | STABLE: BybitPerp->BinPerp, annualized=2.80085670 | snapshot only; external venue execution and transfer constraints unknown | collect repeated snapshots and add venue-specific execution constraints |
| perp_market_map | current_snapshot | MANTA: ann_funding=3.23186009, volume=652165.69526300 | no history yet, so no persistence or PnL evidence | collect snapshots over time and test carry/crowding persistence |
| event_flow | implemented_probe | top_20 imbalance mean_next_return=-0.0000414307, hit_rate=0.500000 | tiny sample and naive label; no order book or liquidation context | extend sample window and add liquidation/funding-time labels |
| defi_yield | current_snapshot | Flare/mystic-finance-lending COREUSDT0: apy=13.660140, tvl=19655935.00 | risk, custody, exit liquidity, and APY decay not modeled | separate real yield from incentive yield and add operational risk checklist |
| market_making | current_snapshot | SOL: spread_bps=0.15731309, imbalance10=0.40999977 | no queue position, fill probability, adverse selection, or fee model | collect repeated L2 snapshots and estimate fill/adverse-selection risk |
| news_social | current_snapshot | fear_greed=12.00000000 Extreme Fear; top_trending=PENGU | attention data is not yet joined to leakage-safe return labels | build event-to-return labels and add richer news/social sources |
| stablecoin_liquidity | current_snapshot | USDT: week_change_usd=-1515562339.74 | supply changes are not yet joined to returns, funding, or regimes | test stablecoin supply change as market liquidity context |
| on_chain_flow | partial_proxy | stablecoin supply proxy exists | wallet, bridge, and exchange inflow/outflow data not connected | add direct flow source instead of only stablecoin supply proxy |
