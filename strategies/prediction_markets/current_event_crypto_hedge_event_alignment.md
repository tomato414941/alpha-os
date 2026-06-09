# Current Event Crypto Hedge Event Alignment

This checks whether the prediction-market event price moved with the crypto hedge return. It also compares the hedge return to same-asset non-event paper tickets. It is a rejection/control artifact, not a trade instruction.

| candidate | asset | status | asset bps | basket bps | event bps | controls | control mean | gap | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| eth_1971905_event_crypto_hedge | ETH | event_alignment_inconclusive | -199.13981029 | -247.99196387 | -909.09090909 | 7 | -75.23122018 | -123.90859011 | collect a ready event-market ticket and same-asset controls before judging |
| btc_1971905_event_crypto_hedge | BTC | event_alignment_inconclusive | -243.77674360 | -247.99196387 | -909.09090909 | 6 | -121.89518042 | -121.88156318 | collect a ready event-market ticket and same-asset controls before judging |
| sol_1971905_event_crypto_hedge | SOL | event_alignment_inconclusive | -301.05933773 | -247.99196387 | -909.09090909 | 6 | -276.10091811 | -24.95841962 | collect a ready event-market ticket and same-asset controls before judging |
| btc_2296152_event_crypto_hedge | BTC | event_alignment_missing_event_ticket | 160.18270594 | 160.18270594 |  | 6 | -121.89518042 | 282.07788636 | collect a ready event-market ticket and same-asset controls before judging |

## Summary

- event_alignment_inconclusive: 3
- event_alignment_missing_event_ticket: 1
