# Strategy Exploration Board

This board tracks broad profit-source exploration. It is not a ranking of deployable strategies.

| lane | status | strongest current signal | main gap | next step |
| --- | --- | --- | --- | --- |
| crypto_market_structure | historical_dislocation | 2024 spot_perp_positive_funding_top_3_14d sharpe=9.1726039299; 2026_to_date best_total=-0.0052691774 | spot/perp carry did not persist after 2024 under the current rule | search current funding dislocations or regime filters before paper trading |
| cross_exchange_funding | execution_assumption_gate | STABLE: conservative_taker_monitor, fee=2.000000bps/fill/venue, conservative_net24=0.0002841668 | real account fees, fills, margin, collateral movement, and liquidation buffer are unvalidated | run longer STABLE monitoring and validate real fee/fill/margin assumptions before paper trading |
| perp_market_map | short_window_carry_reversion_monitor | MON: long_carry_reversion_watch, obs=6, mean_score=14.78059599, mean_funding=-0.73725416 | persistent crowding proxy is not yet joined to future returns or execution costs | label top carry/reversion rows for subsequent returns, funding decay, and liquidation risk |
| event_flow | implemented_probe | top_20 imbalance mean_next_return=-0.0000414307, hit_rate=0.500000 | tiny sample and naive label; no order book or liquidation context | extend sample window and add liquidation/funding-time labels |
| defi_yield | current_snapshot | Flare/mystic-finance-lending COREUSDT0: apy=13.660140, tvl=19655935.00 | risk, custody, exit liquidity, and APY decay not modeled | separate real yield from incentive yield and add operational risk checklist |
| market_making | current_snapshot | SOL: spread_bps=0.15731309, imbalance10=0.40999977 | no queue position, fill probability, adverse selection, or fee model | collect repeated L2 snapshots and estimate fill/adverse-selection risk |
| news_social | current_snapshot | fear_greed=12.00000000 Extreme Fear; top_trending=PENGU | attention data is not yet joined to leakage-safe return labels | build event-to-return labels and add richer news/social sources |
| stablecoin_liquidity | current_snapshot | USDT: week_change_usd=-1515562339.74 | supply changes are not yet joined to returns, funding, or regimes | test stablecoin supply change as market liquidity context |
| on_chain_flow | partial_proxy | stablecoin supply proxy exists | wallet, bridge, and exchange inflow/outflow data not connected | add direct flow source instead of only stablecoin supply proxy |
