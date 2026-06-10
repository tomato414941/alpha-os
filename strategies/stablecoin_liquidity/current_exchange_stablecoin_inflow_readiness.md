# Current Exchange Stablecoin Inflow Readiness

This separates direct exchange-stablecoin-inflow alpha from chain-level stablecoin migration proxies. The current local data can support proxy labels, but direct exchange-inflow alpha needs tagged exchange deposits.

| subject | status | alpha kind | score | flow | week change | week % | proxy label | next probe |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| Ethereum/ETH | direct_exchange_inflow_data_required | direct_exchange_stablecoin_inflow | 179.3378 | stablecoin_supply_outflow | -2367417075 | -0.015800 | 1h=-, 4h=- | obtain or emulate exchange-tagged stablecoin netflow for ETH; keep Ethereum chain supply out of the direct alpha label |
| Solana/SOL | proxy_label_candidate_not_exchange_inflow | chain_liquidity_proxy | 112.3215 | stablecoin_supply_inflow | 798085984 | 0.065573 | 1h=-, 4h=- | label SOL as chain-liquidity proxy, then compare against beta and costs before any trade |
| Polygon/POL | proxy_label_candidate_not_exchange_inflow | chain_liquidity_proxy | 101.6627 | stablecoin_supply_outflow | -198600917 | -0.054439 | 1h=-, 4h=- | label POL as chain-liquidity proxy, then compare against beta and costs before any trade |
| Arbitrum/ARB | proxy_label_candidate_not_exchange_inflow | chain_liquidity_proxy | 99.3600 | stablecoin_supply_outflow | -119397591 | -0.031672 | 1h=-, 4h=- | label ARB as chain-liquidity proxy, then compare against beta and costs before any trade |
| Tron/TRX | chain_proxy_watch_not_exchange_inflow | context_only | 55.1025 | stablecoin_supply_outflow | -449545785 | -0.005004 | 1h=-, 4h=- | keep Tron as context until mapped to a tradable token and labeled |
| Hyperliquid L1/HYPE | chain_proxy_watch_not_exchange_inflow | context_only | 50.5573 | stablecoin_supply_inflow | 141303949 | 0.021550 | 1h=-, 4h=- | keep Hyperliquid L1 as context until mapped to a tradable token and labeled |
| Sui/SUI | chain_proxy_watch_not_exchange_inflow | context_only | 50.4904 | stablecoin_supply_inflow | 22733236 | 0.067951 | 1h=-, 4h=- | keep Sui as context until mapped to a tradable token and labeled |
| Berachain/BERA | chain_proxy_watch_not_exchange_inflow | context_only | 50.2834 | stablecoin_supply_outflow | -5285286 | -0.070084 | 1h=-, 4h=- | keep Berachain as context until mapped to a tradable token and labeled |
| BSC/BNB | chain_proxy_watch_not_exchange_inflow | context_only | 48.4732 | stablecoin_supply_outflow | -60162055 | -0.004771 | 1h=-, 4h=- | keep BSC as context until mapped to a tradable token and labeled |
| Starknet/unmapped | unmapped_chain_context_not_alpha | context_only | 25.3847 | stablecoin_supply_outflow | -81451909 | -0.310699 | 1h=-, 4h=- | keep Starknet as context until mapped to a tradable token and labeled |
| Stellar/unmapped | unmapped_chain_context_not_alpha | context_only | 25.1945 | stablecoin_supply_inflow | 70264766 | 0.343293 | 1h=-, 4h=- | keep Stellar as context until mapped to a tradable token and labeled |
| Flow/unmapped | unmapped_chain_context_not_alpha | context_only | 24.9309 | stablecoin_supply_inflow | 54758159 | 2.857789 | 1h=-, 4h=- | keep Flow as context until mapped to a tradable token and labeled |
| Ink/unmapped | unmapped_chain_context_not_alpha | context_only | 24.4677 | stablecoin_supply_inflow | 27512621 | 0.178418 | 1h=-, 4h=- | keep Ink as context until mapped to a tradable token and labeled |
| XDC/unmapped | unmapped_chain_context_not_alpha | context_only | 24.4274 | stablecoin_supply_inflow | 25142059 | 0.406690 | 1h=-, 4h=- | keep XDC as context until mapped to a tradable token and labeled |
| Cardano/unmapped | unmapped_chain_context_not_alpha | context_only | 24.1064 | stablecoin_supply_outflow | -6261617 | -0.237473 | 1h=-, 4h=- | keep Cardano as context until mapped to a tradable token and labeled |
| World Chain/unmapped | unmapped_chain_context_not_alpha | context_only | 24.0842 | stablecoin_supply_inflow | 4953257 | 0.238259 | 1h=-, 4h=- | keep World Chain as context until mapped to a tradable token and labeled |
| ZKsync Era/unmapped | unmapped_chain_context_not_alpha | context_only | 23.8203 | stablecoin_supply_outflow | -6804851 | -0.159634 | 1h=-, 4h=- | keep ZKsync Era as context until mapped to a tradable token and labeled |
| Aptos/unmapped | unmapped_chain_context_not_alpha | context_only | 23.5680 | stablecoin_supply_inflow | 113834259 | 0.110306 | 1h=-, 4h=- | keep Aptos as context until mapped to a tradable token and labeled |
| Base/unmapped | unmapped_chain_context_not_alpha | context_only | 21.7374 | stablecoin_supply_inflow | 177077294 | 0.041121 | 1h=-, 4h=- | keep Base as context until mapped to a tradable token and labeled |
| Algorand/unmapped | unmapped_chain_context_not_alpha | context_only | 20.2962 | stablecoin_supply_inflow | 4000521 | 0.076862 | 1h=-, 4h=- | keep Algorand as context until mapped to a tradable token and labeled |
