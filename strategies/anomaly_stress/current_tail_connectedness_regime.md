# Current Tail Connectedness Regime

This groups current anomaly, event-pressure, and sector states into broad tail or connectedness regimes. It is a regime/control table, not a directional trade list.

| regime | role | status | assets | sources | severity | connectedness | next probe |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| sector_momentum_connectedness_regime | sector_connectedness_control | sector_regime_before_single_asset_label | LINK, ZEC | 13 | 8.2869 | 163.1650 | label Privacy as a sector regime before promoting a single constituent |
| event_pressure_connectedness_regime | event_connectedness_control | mixed_event_tail_context | BTC, ZEC, HYPE, SOL, ALLO, ETH, USDT, BEAT | 10 | 5.4611 | 139.5328 | label event-pressure assets as a connected regime before using any one event as standalone alpha |
| defi_lending_tail_regime | credit_liquidity_tail_control | lending_liquidity_tail_regime | Ethereum USR/BONDUSD, Ethereum USDC/PAXG, Ethereum USDC/sdeUSD, Ethereum USDT/USDT, Ethereum USDC/wstUSR, Base USDC/HERMES | 6 | 6.0000 | 120.1332 | condition lending/yield labels on credit-liquidity stress before any supply action |
| volatility_tail_regime | volatility_tail_control | volatility_mispricing_tail_regime | ETH 2026-06-26 long_atm_straddle, ETH 2026-06-10 long_atm_straddle, ETH 2026-06-19 long_atm_straddle, ETH 2026-06-11 long_atm_straddle, BTC 2026-06-12 long_atm_straddle, BTC 2026-06-10 long_atm_straddle | 6 | 2.7200 | 112.6386 | test whether cheap-vol candidates persist as a volatility regime after hedge costs |
| stablecoin_peg_tail_regime | peg_tail_control | multi_peg_tail_regime | pmUSD, USDY, USYC, reUSD, apxUSD, USDai | 8 | 0.7366 | 108.5808 | treat peg anomalies as a tail regime and validate redemption/route mechanics before trading |

## Interpretation

Tail-regime rows should condition downstream labels. They should not be collapsed into a long/short action until the connected assets, timestamps, execution costs, and regime persistence are measured.
