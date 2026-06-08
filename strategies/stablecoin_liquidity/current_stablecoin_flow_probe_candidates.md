# Current Stablecoin Flow Probe Candidates

This separates direct exchange-flow data probes from chain-liquidity proxy labels. Chain stablecoin supply is not treated as exchange inflow.

| candidate | subject | type | status | priority | flow | week change | context | next step |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| stablecoin-flow-ethereum-eth | Ethereum/ETH | direct_exchange_flow_data_probe | direct_exchange_inflow_data_required | 192.1565 | stablecoin_supply_outflow | -2499493730.17000008 | 0.57304300 | collect or emulate exchange-tagged stablecoin netflow for ETH; do not use chain supply as the label |
| stablecoin-flow-solana-sol | Solana/SOL | chain_liquidity_proxy_label | proxy_label_candidate_not_exchange_inflow | 115.5640 | stablecoin_supply_inflow | 881166597.41999996 | 0.07715655 | open a chain-liquidity proxy label for SOL and compare it against beta, funding, spread, and depth |
| stablecoin-flow-polygon-pol | Polygon/POL | chain_liquidity_proxy_label | proxy_label_candidate_not_exchange_inflow | 102.2329 | stablecoin_supply_outflow | -218594198.83000001 | 0.00000000 | open a chain-liquidity proxy label for POL and compare it against beta, funding, spread, and depth |
| stablecoin-flow-hyperliquid-l1-hype | Hyperliquid L1/HYPE | chain_liquidity_proxy_label | proxy_label_candidate_not_exchange_inflow | 101.4804 | stablecoin_supply_inflow | 235067901.74000001 | -0.00107116 | open a chain-liquidity proxy label for HYPE and compare it against beta, funding, spread, and depth |
| stablecoin-flow-arbitrum-arb | Arbitrum/ARB | chain_liquidity_proxy_label | proxy_label_candidate_not_exchange_inflow | 98.5086 | stablecoin_supply_outflow | -121911731.11000000 | -0.04610859 | open a chain-liquidity proxy label for ARB and compare it against beta, funding, spread, and depth |
