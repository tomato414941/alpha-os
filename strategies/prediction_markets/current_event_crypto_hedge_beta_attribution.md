# Current Event Crypto Hedge Beta Attribution

This checks whether event-crypto hedge paper returns are mostly a common BTC/ETH/SOL beta move or an asset-specific residual. It is a diagnostic, not a trade instruction.

| candidate | asset | action | status | asset bps | basket bps | residual bps | gap | edge | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| btc_1962237_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_beta_attribution_pending |  |  |  | 0.175000 | 0.170000 | wait for a ready reaction label before attributing beta |
| eth_1962237_event_crypto_hedge | ETH | paper_long_risk_relief | event_crypto_beta_attribution_pending |  |  |  | 0.175000 | 0.170000 | wait for a ready reaction label before attributing beta |
| sol_1962237_event_crypto_hedge | SOL | paper_long_risk_relief | event_crypto_beta_attribution_pending |  |  |  | 0.175000 | 0.170000 | wait for a ready reaction label before attributing beta |
| btc_2296152_event_crypto_hedge | BTC | paper_short_risk_escalation | event_crypto_beta_attribution_pending |  |  |  | 0.095000 | 0.090000 | wait for a ready reaction label before attributing beta |
| btc_1962237_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_beta_attribution_negative | -84.95486773 | -84.95486773 | 0.00000000 | 0.175000 | 0.170000 | refresh marks and keep this as diagnostic evidence only |

## Summary

- event_crypto_beta_attribution_negative: 1
- event_crypto_beta_attribution_pending: 4
- best asset return: btc_1962237_event_crypto_hedge asset=bps basket=bps residual=bps
