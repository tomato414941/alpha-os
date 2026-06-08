# Current Exchange Stablecoin Inflow Readiness

This separates direct exchange-stablecoin-inflow alpha from chain-level stablecoin migration proxies. The current local data can support proxy labels, but direct exchange-inflow alpha needs tagged exchange deposits.

| subject | status | alpha kind | score | flow | week change | week % | proxy label | next probe |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| Ethereum/ETH | direct_exchange_inflow_data_required | direct_exchange_stablecoin_inflow | 180.6956 | stablecoin_supply_outflow | -2499493730 | -0.016684 | 1h=-, 4h=- | obtain or emulate exchange-tagged stablecoin netflow for ETH; keep Ethereum chain supply out of the direct alpha label |
| Solana/SOL | proxy_label_candidate_not_exchange_inflow | chain_liquidity_proxy | 114.0209 | stablecoin_supply_inflow | 881166597 | 0.072406 | 1h=-, 4h=- | label SOL as chain-liquidity proxy, then compare against beta and costs before any trade |
| Polygon/POL | proxy_label_candidate_not_exchange_inflow | chain_liquidity_proxy | 102.2329 | stablecoin_supply_outflow | -218594199 | -0.059924 | 1h=-, 4h=- | label POL as chain-liquidity proxy, then compare against beta and costs before any trade |
| Hyperliquid L1/HYPE | proxy_label_candidate_not_exchange_inflow | chain_liquidity_proxy | 101.5019 | stablecoin_supply_inflow | 235067902 | 0.035850 | 1h=-, 4h=- | label HYPE as chain-liquidity proxy, then compare against beta and costs before any trade |
| Arbitrum/ARB | proxy_label_candidate_not_exchange_inflow | chain_liquidity_proxy | 99.4308 | stablecoin_supply_outflow | -121911731 | -0.032340 | 1h=-, 4h=- | label ARB as chain-liquidity proxy, then compare against beta and costs before any trade |
| Tron/TRX | chain_proxy_watch_not_exchange_inflow | context_only | 55.0822 | stablecoin_supply_outflow | -448381368 | -0.004992 | 1h=-, 4h=- | keep Tron as context until mapped to a tradable token and labeled |
| Sui/SUI | chain_proxy_watch_not_exchange_inflow | context_only | 50.5812 | stablecoin_supply_inflow | 23370580 | 0.069856 | 1h=-, 4h=- | keep Sui as context until mapped to a tradable token and labeled |
| Berachain/BERA | chain_proxy_watch_not_exchange_inflow | context_only | 50.2230 | stablecoin_supply_outflow | -5179279 | -0.068690 | 1h=-, 4h=- | keep Berachain as context until mapped to a tradable token and labeled |
| BSC/BNB | chain_proxy_watch_not_exchange_inflow | context_only | 48.4716 | stablecoin_supply_outflow | -60081922 | -0.004766 | 1h=-, 4h=- | keep BSC as context until mapped to a tradable token and labeled |
| Starknet/unmapped | unmapped_chain_context_not_alpha | context_only | 25.3865 | stablecoin_supply_outflow | -81560659 | -0.311116 | 1h=-, 4h=- | keep Starknet as context until mapped to a tradable token and labeled |
| Stellar/unmapped | unmapped_chain_context_not_alpha | context_only | 25.1944 | stablecoin_supply_inflow | 70260435 | 0.343273 | 1h=-, 4h=- | keep Stellar as context until mapped to a tradable token and labeled |
| Flow/unmapped | unmapped_chain_context_not_alpha | context_only | 24.9383 | stablecoin_supply_inflow | 55196512 | 2.880633 | 1h=-, 4h=- | keep Flow as context until mapped to a tradable token and labeled |
| Ink/unmapped | unmapped_chain_context_not_alpha | context_only | 24.4573 | stablecoin_supply_inflow | 26902030 | 0.174483 | 1h=-, 4h=- | keep Ink as context until mapped to a tradable token and labeled |
| XDC/unmapped | unmapped_chain_context_not_alpha | context_only | 24.4274 | stablecoin_supply_inflow | 25141936 | 0.406690 | 1h=-, 4h=- | keep XDC as context until mapped to a tradable token and labeled |
| Cardano/unmapped | unmapped_chain_context_not_alpha | context_only | 24.1018 | stablecoin_supply_outflow | -5985502 | -0.227003 | 1h=-, 4h=- | keep Cardano as context until mapped to a tradable token and labeled |
| World Chain/unmapped | unmapped_chain_context_not_alpha | context_only | 24.0799 | stablecoin_supply_inflow | 4701424 | 0.226146 | 1h=-, 4h=- | keep World Chain as context until mapped to a tradable token and labeled |
| ZKsync Era/unmapped | unmapped_chain_context_not_alpha | context_only | 23.8204 | stablecoin_supply_outflow | -6804829 | -0.159637 | 1h=-, 4h=- | keep ZKsync Era as context until mapped to a tradable token and labeled |
| Aptos/unmapped | unmapped_chain_context_not_alpha | context_only | 23.2327 | stablecoin_supply_inflow | 108010874 | 0.104680 | 1h=-, 4h=- | keep Aptos as context until mapped to a tradable token and labeled |
| Base/unmapped | unmapped_chain_context_not_alpha | context_only | 21.3449 | stablecoin_supply_inflow | 162405398 | 0.037714 | 1h=-, 4h=- | keep Base as context until mapped to a tradable token and labeled |
| Algorand/unmapped | unmapped_chain_context_not_alpha | context_only | 20.3181 | stablecoin_supply_inflow | 4027032 | 0.077371 | 1h=-, 4h=- | keep Algorand as context until mapped to a tradable token and labeled |
