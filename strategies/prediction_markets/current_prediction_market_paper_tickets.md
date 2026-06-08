# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| US x Iran permanent peace deal by June 30, 2026? | Yes | geopolitical_event | event_probability_model | 0.1400 | 0.1500 | 0.0100 | 240.7956 | 1201120 | 154.957109 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 30, 2026? | No | geopolitical_event | event_probability_model | 0.8500 | 0.8600 | 0.0100 | 240.7956 | 1201120 | 154.957109 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | geopolitical_event | event_probability_model | 0.0900 | 0.1000 | 0.0100 | 291.1396 | 787919 | 153.059392 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | No | geopolitical_event | event_probability_model | 0.9000 | 0.9100 | 0.0100 | 291.1396 | 787919 | 153.059392 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | No | geopolitical_event | event_probability_model | 0.8200 | 0.8300 | 0.0100 | 152.4355 | 381991 | 148.136854 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | Yes | geopolitical_event | event_probability_model | 0.1700 | 0.1800 | 0.0100 | 139.2285 | 381991 | 148.136854 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by July 31, 2026? | No | geopolitical_event | event_probability_model | 0.7100 | 0.7200 | 0.0100 | 121.6246 | 221742 | 146.468668 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by July 31, 2026? | Yes | geopolitical_event | event_probability_model | 0.2800 | 0.2900 | 0.0100 | 112.3240 | 221742 | 146.468668 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by July 31? | No | geopolitical_event | event_probability_model | 0.7200 | 0.7300 | 0.0100 | 71.2893 | 185735 | 117.323317 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | geopolitical_event | event_probability_model | 0.2700 | 0.2800 | 0.0100 | 63.1352 | 185735 | 109.169307 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | No | geopolitical_event | event_probability_model | 0.8200 | 0.8300 | 0.0100 | 58.7706 | 351880 | 107.366842 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | Yes | geopolitical_event | event_probability_model | 0.1700 | 0.1800 | 0.0100 | 55.9507 | 351880 | 104.546932 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Israel closes its airspace by June 15? | Yes | geopolitical_event | event_probability_model | 0.5300 | 0.5500 | 0.0200 | 18.4168 | 2397434 | 81.669828 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Israel closes its airspace by June 15? | No | geopolitical_event | event_probability_model | 0.4500 | 0.4700 | 0.0200 | 18.4168 | 2397434 | 81.669828 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Libema Open: Otto Virtanen vs Kamil Majchrzak | Yes | event_market | event_probability_model | 0.5200 | 0.5300 | 0.0100 | 52.2627 | 218386 | 73.360447 | paper_event_model_watch | depth exists but external signal source is not identified |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
