# Current Event Crypto Hedge Beta Attribution

This checks whether event-crypto hedge paper returns are mostly a common BTC/ETH/SOL beta move or an asset-specific residual. It is a diagnostic, not a trade instruction.

| candidate | asset | action | status | asset bps | basket bps | residual bps | gap | edge | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| sol_1971905_event_crypto_hedge | SOL | paper_long_risk_relief | event_crypto_beta_move_supported | 188.23209757 | 151.79272002 | 36.43937755 | 0.335000 | 0.330000 | repeat on fresh event markets and add funding, spread/depth, and event timestamp controls |
| eth_1971905_event_crypto_hedge | ETH | paper_long_risk_relief | event_crypto_beta_move_supported | 170.17196956 | 151.79272002 | 18.37924954 | 0.335000 | 0.330000 | repeat on fresh event markets and add funding, spread/depth, and event timestamp controls |
| btc_1971905_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_beta_move_supported | 96.97409294 | 151.79272002 | -54.81862708 | 0.335000 | 0.330000 | repeat on fresh event markets and add funding, spread/depth, and event timestamp controls |
| btc_1962237_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_beta_attribution_pending |  |  |  | 0.085000 | 0.080000 | wait for a ready reaction label before attributing beta |

## Summary

- event_crypto_beta_attribution_pending: 1
- event_crypto_beta_move_supported: 3
- best asset return: sol_1971905_event_crypto_hedge asset=188.23209757bps basket=151.79272002bps residual=36.43937755bps
