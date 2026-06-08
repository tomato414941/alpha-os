# Current Event Crypto Hedge Beta Attribution

This checks whether event-crypto hedge paper returns are mostly a common BTC/ETH/SOL beta move or an asset-specific residual. It is a diagnostic, not a trade instruction.

| candidate | asset | action | status | asset bps | basket bps | residual bps | gap | edge | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| eth_1971905_event_crypto_hedge | ETH | paper_long_risk_relief | event_crypto_beta_move_supported | 99.46671460 | 76.66745814 | 22.79925646 | 0.255000 | 0.250000 | repeat on fresh event markets and add funding, spread/depth, and event timestamp controls |
| sol_1971905_event_crypto_hedge | SOL | paper_long_risk_relief | event_crypto_beta_move_supported | 71.24743388 | 76.66745814 | -5.42002426 | 0.255000 | 0.250000 | repeat on fresh event markets and add funding, spread/depth, and event timestamp controls |
| btc_1971905_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_beta_move_supported | 59.28822593 | 76.66745814 | -17.37923221 | 0.255000 | 0.250000 | repeat on fresh event markets and add funding, spread/depth, and event timestamp controls |

## Summary

- event_crypto_beta_move_supported: 3
- best asset return: eth_1971905_event_crypto_hedge asset=99.46671460bps basket=76.66745814bps residual=22.79925646bps
