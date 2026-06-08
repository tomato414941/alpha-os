# Current Sentiment Contagion Negative Control

This separates belief, attention, and event-probability movement from return-predictive alpha. It is a negative-control table, not a trade instruction.

| symbol | status | belief proxy | return support | gap | strongest source | reason | next probe |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| ETH | conflicting_social_source_control | 209.5012 | 7.4998 | 202.0014 | attention:dedupe_news_before_attention_label | cross-modal source split already marks this source as a conflict or negative control | keep ETH as a negative control in the social/event alpha lane |
| BTC | conflicting_social_source_control | 204.7545 | 21.8964 | 182.8581 | event_pressure:multi_source_event_pressure | cross-modal source split already marks this source as a conflict or negative control | keep BTC as a negative control in the social/event alpha lane |
| HYPE | conflicting_social_source_control | 159.4076 | 0.0000 | 159.4076 | event_pressure:two_source_event_pressure | cross-modal source split already marks this source as a conflict or negative control | keep HYPE as a negative control in the social/event alpha lane |

## Interpretation

A high control gap means attention or event-belief evidence is stronger than clean return evidence. Those rows should be used as controls or falsification tests before promoting a social/event signal.
