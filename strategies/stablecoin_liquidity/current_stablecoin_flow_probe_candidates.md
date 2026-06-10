# Current Stablecoin Flow Probe Candidates

This separates direct exchange-flow data probes from chain-liquidity proxy labels. Chain stablecoin supply is not treated as exchange inflow.

| candidate | subject | type | status | priority | flow | week change | context | next step |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| stablecoin-flow-ethereum-eth | Ethereum/ETH | direct_exchange_flow_data_probe | direct_exchange_inflow_data_required | 190.7821 | stablecoin_supply_outflow | -2367417074.51000023 | 0.57221532 | collect or emulate exchange-tagged stablecoin netflow for ETH; do not use chain supply as the label |
| stablecoin-flow-solana-sol | Solana/SOL | chain_liquidity_proxy_label | proxy_label_candidate_not_exchange_inflow | 115.3047 | stablecoin_supply_inflow | 798085983.92999995 | 0.14915675 | open a chain-liquidity proxy label for SOL and compare it against beta, funding, spread, and depth |
| stablecoin-flow-polygon-pol | Polygon/POL | chain_liquidity_proxy_label | proxy_label_candidate_not_exchange_inflow | 101.6627 | stablecoin_supply_outflow | -198600917.22000000 | 0.00000000 | open a chain-liquidity proxy label for POL and compare it against beta, funding, spread, and depth |
| stablecoin-flow-arbitrum-arb | Arbitrum/ARB | chain_liquidity_proxy_label | proxy_label_candidate_not_exchange_inflow | 98.5024 | stablecoin_supply_outflow | -119397590.81000000 | -0.04287645 | open a chain-liquidity proxy label for ARB and compare it against beta, funding, spread, and depth |
