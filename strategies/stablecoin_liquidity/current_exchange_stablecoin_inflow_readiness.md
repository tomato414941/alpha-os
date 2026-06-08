# Current Exchange Stablecoin Inflow Readiness

This separates direct exchange-stablecoin-inflow alpha from chain-level stablecoin migration proxies. The current local data can support proxy labels, but direct exchange-inflow alpha needs tagged exchange deposits.

| subject | status | alpha kind | score | flow | week change | week % | proxy label | next probe |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| Ethereum/ETH | direct_exchange_inflow_data_required | direct_exchange_stablecoin_inflow | 181.2082 | stablecoin_supply_outflow | -2549343868 | -0.017018 | 1h=-, 4h=- | obtain or emulate exchange-tagged stablecoin netflow for ETH; keep Ethereum chain supply out of the direct alpha label |
| Solana/SOL | proxy_label_candidate_not_exchange_inflow | chain_liquidity_proxy | 115.0369 | stablecoin_supply_inflow | 930837276 | 0.076493 | 1h=-, 4h=- | label SOL as chain-liquidity proxy, then compare against beta and costs before any trade |
| Polygon/POL | proxy_label_candidate_not_exchange_inflow | chain_liquidity_proxy | 102.0567 | stablecoin_supply_outflow | -212411978 | -0.058230 | 1h=-, 4h=- | label POL as chain-liquidity proxy, then compare against beta and costs before any trade |
| Tron/TRX | chain_proxy_watch_not_exchange_inflow | context_only | 55.2032 | stablecoin_supply_outflow | -455311943 | -0.005070 | 1h=-, 4h=- | keep Tron as context until mapped to a tradable token and labeled |
| Sui/SUI | chain_proxy_watch_not_exchange_inflow | context_only | 50.8271 | stablecoin_supply_inflow | 25095019 | 0.075012 | 1h=-, 4h=- | keep Sui as context until mapped to a tradable token and labeled |
| Avalanche/AVAX | chain_proxy_watch_not_exchange_inflow | context_only | 50.4809 | stablecoin_supply_outflow | -49109349 | -0.057048 | 1h=-, 4h=- | keep Avalanche as context until mapped to a tradable token and labeled |
| Berachain/BERA | chain_proxy_watch_not_exchange_inflow | context_only | 50.2293 | stablecoin_supply_outflow | -5189917 | -0.068835 | 1h=-, 4h=- | keep Berachain as context until mapped to a tradable token and labeled |
| Arbitrum/ARB | chain_proxy_watch_not_exchange_inflow | context_only | 50.0021 | stablecoin_supply_outflow | -97793289 | -0.025944 | 1h=-, 4h=- | keep Arbitrum as context until mapped to a tradable token and labeled |
| Starknet/unmapped | unmapped_chain_context_not_alpha | context_only | 25.3885 | stablecoin_supply_outflow | -81675886 | -0.311558 | 1h=-, 4h=- | keep Starknet as context until mapped to a tradable token and labeled |
| Stellar/unmapped | unmapped_chain_context_not_alpha | context_only | 25.1191 | stablecoin_supply_inflow | 65828329 | 0.321621 | 1h=-, 4h=- | keep Stellar as context until mapped to a tradable token and labeled |
| Ink/unmapped | unmapped_chain_context_not_alpha | context_only | 24.4446 | stablecoin_supply_inflow | 26150368 | 0.169620 | 1h=-, 4h=- | keep Ink as context until mapped to a tradable token and labeled |
| XDC/unmapped | unmapped_chain_context_not_alpha | context_only | 24.4275 | stablecoin_supply_inflow | 25149357 | 0.406813 | 1h=-, 4h=- | keep XDC as context until mapped to a tradable token and labeled |
| Cardano/unmapped | unmapped_chain_context_not_alpha | context_only | 24.1021 | stablecoin_supply_outflow | -6004799 | -0.227737 | 1h=-, 4h=- | keep Cardano as context until mapped to a tradable token and labeled |
| World Chain/unmapped | unmapped_chain_context_not_alpha | context_only | 24.0787 | stablecoin_supply_inflow | 4630178 | 0.222721 | 1h=-, 4h=- | keep World Chain as context until mapped to a tradable token and labeled |
| ZKsync Era/unmapped | unmapped_chain_context_not_alpha | context_only | 23.8087 | stablecoin_supply_outflow | -6793080 | -0.159363 | 1h=-, 4h=- | keep ZKsync Era as context until mapped to a tradable token and labeled |
| Aptos/unmapped | unmapped_chain_context_not_alpha | context_only | 22.7344 | stablecoin_supply_inflow | 99369542 | 0.096313 | 1h=-, 4h=- | keep Aptos as context until mapped to a tradable token and labeled |
| Flow/unmapped | unmapped_chain_context_not_alpha | context_only | 22.0265 | stablecoin_supply_inflow | 2274597 | 0.118758 | 1h=-, 4h=- | keep Flow as context until mapped to a tradable token and labeled |
| Base/unmapped | unmapped_chain_context_not_alpha | context_only | 21.5276 | stablecoin_supply_inflow | 169236523 | 0.039300 | 1h=-, 4h=- | keep Base as context until mapped to a tradable token and labeled |
| Katana/unmapped | unmapped_chain_context_not_alpha | context_only | 21.0205 | stablecoin_supply_outflow | -2410893 | -0.094751 | 1h=-, 4h=- | keep Katana as context until mapped to a tradable token and labeled |
| Algorand/unmapped | unmapped_chain_context_not_alpha | context_only | 20.3199 | stablecoin_supply_inflow | 4029241 | 0.077414 | 1h=-, 4h=- | keep Algorand as context until mapped to a tradable token and labeled |
