# Current Event Crypto Hedge Event Alignment

This checks whether the prediction-market event price moved with the crypto hedge return. It also compares the hedge return to same-asset non-event paper tickets. It is a rejection/control artifact, not a trade instruction.

| candidate | asset | status | asset bps | basket bps | event bps | controls | control mean | gap | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sol_1971905_event_crypto_hedge | SOL | event_probability_flat_crypto_moved | 188.23209757 | 151.79272002 | 0.00000000 | 3 | 164.20708983 | 24.02500774 | do not promote this event hedge; require event-market probability movement or stronger timestamp evidence |
| eth_1971905_event_crypto_hedge | ETH | event_probability_flat_crypto_moved | 170.17196956 | 151.79272002 | 0.00000000 | 9 | 114.52581949 | 55.64615007 | do not promote this event hedge; require event-market probability movement or stronger timestamp evidence |
| btc_1971905_event_crypto_hedge | BTC | event_probability_flat_crypto_moved | 96.97409294 | 151.79272002 | 0.00000000 | 1 | -96.97409294 | 193.94818588 | do not promote this event hedge; require event-market probability movement or stronger timestamp evidence |
| btc_1962237_event_crypto_hedge | BTC | event_alignment_missing_event_ticket |  |  |  | 1 | -96.97409294 | 96.97409294 | collect a ready event-market ticket and same-asset controls before judging |

## Summary

- event_alignment_missing_event_ticket: 1
- event_probability_flat_crypto_moved: 3
