# Current Event Crypto Hedge Beta Attribution

This checks whether event-crypto hedge paper returns are mostly a common BTC/ETH/SOL beta move or an asset-specific residual. It is a diagnostic, not a trade instruction.

| candidate | asset | action | status | asset bps | basket bps | residual bps | gap | edge | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| btc_2296152_event_crypto_hedge | BTC | paper_short_risk_escalation | event_crypto_beta_move_supported | 160.18270594 | 160.18270594 | 0.00000000 | 0.125000 | 0.120000 | repeat on fresh event markets and add funding, spread/depth, and event timestamp controls |
| eth_1971905_event_crypto_hedge | ETH | paper_long_risk_relief | event_crypto_beta_attribution_negative | -199.13981029 | -247.99196387 | 48.85215358 | 0.185000 | 0.180000 | refresh marks and keep this as diagnostic evidence only |
| btc_1971905_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_beta_attribution_negative | -243.77674360 | -247.99196387 | 4.21522027 | 0.185000 | 0.180000 | refresh marks and keep this as diagnostic evidence only |
| sol_1971905_event_crypto_hedge | SOL | paper_long_risk_relief | event_crypto_beta_attribution_negative | -301.05933773 | -247.99196387 | -53.06737386 | 0.185000 | 0.180000 | refresh marks and keep this as diagnostic evidence only |

## Summary

- event_crypto_beta_attribution_negative: 3
- event_crypto_beta_move_supported: 1
- best asset return: btc_2296152_event_crypto_hedge asset=160.18270594bps basket=160.18270594bps residual=0.00000000bps
