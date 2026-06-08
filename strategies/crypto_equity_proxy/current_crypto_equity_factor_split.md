# Current Crypto Equity Factor Split

This separates crypto-equity proxy observations into beta-hedge, residual relative-value, stress-control, and market-hours-gap roles. It is not a trade instruction.

| factor | role | target | status | side | score | next probe |
| --- | --- | --- | --- | --- | ---: | --- |
| ethm_market_hours_gap | market_hours_gap_control | ETH | timestamp_boundary_required | none | 75.0189 | separate ETHM market-hours gap before using it as crypto alpha |
| mstr_btc_dislocation | residual_relative_value | BTC | separate_residual_from_beta_before_label | long_mstr_short_btc | 14.1808 | label mstr_btc_dislocation as residual spread, not as broad BTC beta |
| miner_stress_vs_btc | equity_stress_control | BTC | stress_control_before_directional_trade | de_risk_alt_beta | 10.7038 | use miner stress as a regime control before any directional crypto action |
| crypto_equity_proxy_lead_short | crypto_beta_hedge | BTC_ETH | beta_hedge_label_candidate | short_btc_eth | 10.6427 | label BTC_ETH beta hedge with explicit equity-hours and crypto-hours windows |
| exchange_beta_lead | crypto_beta_hedge | BTC_ETH | beta_hedge_label_candidate | short_btc_eth | 8.8878 | label BTC_ETH beta hedge with explicit equity-hours and crypto-hours windows |
| eth_treasury_proxy_lead | crypto_beta_hedge | ETH | context_only | eth_treasury_vs_eth_context | 2.9687 | label ETH beta hedge with explicit equity-hours and crypto-hours windows |

## Interpretation

A beta hedge candidate is different from a residual relative-value candidate. Market-hours gaps are timing controls: they should not be merged with 24/7 crypto signals until the timestamp boundary is explicit.
