# Current Stablecoin Flow Probe Candidates

This separates direct exchange-flow data probes from chain-liquidity proxy labels. Chain stablecoin supply is not treated as exchange inflow.

| candidate | subject | type | status | priority | flow | week change | context | next step |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| stablecoin-flow-ethereum-eth | Ethereum/ETH | direct_exchange_flow_data_probe | direct_exchange_inflow_data_required | 192.6690 | stablecoin_supply_outflow | -2549343868.46000004 | 0.57304300 | collect or emulate exchange-tagged stablecoin netflow for ETH; do not use chain supply as the label |
| stablecoin-flow-solana-sol | Solana/SOL | chain_liquidity_proxy_label | proxy_label_candidate_not_exchange_inflow | 116.5801 | stablecoin_supply_inflow | 930837276.44000006 | 0.07715655 | open a chain-liquidity proxy label for SOL and compare it against beta, funding, spread, and depth |
| stablecoin-flow-polygon-pol | Polygon/POL | chain_liquidity_proxy_label | proxy_label_candidate_not_exchange_inflow | 102.0567 | stablecoin_supply_outflow | -212411978.06999999 | 0.00000000 | open a chain-liquidity proxy label for POL and compare it against beta, funding, spread, and depth |
