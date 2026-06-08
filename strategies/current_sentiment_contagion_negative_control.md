# Current Sentiment Contagion Negative Control

This separates belief, attention, and event-probability movement from return-predictive alpha. It is a negative-control table, not a trade instruction.

| symbol | status | belief proxy | return support | gap | strongest source | reason | next probe |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| BTC | belief_price_decoupling_control_required | 243.1922 | 5.2756 | 237.9165 | event_pressure:multi_source_event_pressure | crypto moved while the event-probability ticket was flat; do not treat the belief market as causal yet | for BTC, require event-probability movement or stronger timestamp evidence before event hedge promotion |
| HYPE | conflicting_social_source_control | 153.6390 | 0.0000 | 153.6390 | attention:ticker_specific_attention_alpha_candidate | cross-modal source split already marks this source as a conflict or negative control | keep HYPE as a negative control in the social/event alpha lane |
| ETH | belief_price_decoupling_control_required | 154.2859 | 1.6250 | 152.6609 | cross_modal_control:wallet_flow | crypto moved while the event-probability ticket was flat; do not treat the belief market as causal yet | for ETH, require event-probability movement or stronger timestamp evidence before event hedge promotion |

## Interpretation

A high control gap means attention or event-belief evidence is stronger than clean return evidence. Those rows should be used as controls or falsification tests before promoting a social/event signal.
