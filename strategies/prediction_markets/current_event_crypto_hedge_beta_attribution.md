# Current Event Crypto Hedge Beta Attribution

This checks whether event-crypto hedge paper returns are mostly a common BTC/ETH/SOL beta move or an asset-specific residual. It is a diagnostic, not a trade instruction.

| candidate | asset | action | status | asset bps | basket bps | residual bps | gap | edge | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| btc_2296152_event_crypto_hedge | BTC | paper_short_risk_escalation | event_crypto_beta_move_supported | 38.74625925 | 38.74625925 | 0.00000000 | 0.085000 | 0.080000 | repeat on fresh event markets and add funding, spread/depth, and event timestamp controls |
| eth_1962237_event_crypto_hedge | ETH | paper_long_risk_relief | event_crypto_beta_not_supported | 45.03970606 | -49.61230386 | 94.65200992 | 0.175000 | 0.170000 | refresh marks and keep this as diagnostic evidence only |
| eth_1971905_event_crypto_hedge | ETH | paper_long_risk_relief | event_crypto_beta_attribution_negative | -13.55093384 | -74.03812939 | 60.48719555 | 0.185000 | 0.180000 | refresh marks and keep this as diagnostic evidence only |
| btc_1962237_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_beta_attribution_negative | -38.74625925 | -49.61230386 | 10.86604460 | 0.175000 | 0.170000 | refresh marks and keep this as diagnostic evidence only |
| sol_1962237_event_crypto_hedge | SOL | paper_long_risk_relief | event_crypto_beta_attribution_negative | -81.37070358 | -49.61230386 | -31.75839972 | 0.175000 | 0.170000 | refresh marks and keep this as diagnostic evidence only |
| sol_1971905_event_crypto_hedge | SOL | paper_long_risk_relief | event_crypto_beta_attribution_negative | -85.19149567 | -74.03812939 | -11.15336628 | 0.185000 | 0.180000 | refresh marks and keep this as diagnostic evidence only |
| btc_1971905_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_beta_attribution_negative | -123.37195865 | -74.03812939 | -49.33382926 | 0.185000 | 0.180000 | refresh marks and keep this as diagnostic evidence only |
| btc_1962237_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_beta_attribution_negative | -123.37195865 | -49.61230386 | -73.75965479 | 0.175000 | 0.170000 | refresh marks and keep this as diagnostic evidence only |

## Summary

- event_crypto_beta_attribution_negative: 6
- event_crypto_beta_move_supported: 1
- event_crypto_beta_not_supported: 1
- best asset return: eth_1962237_event_crypto_hedge asset=45.03970606bps basket=-49.61230386bps residual=94.65200992bps
