# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| US x Iran permanent peace deal by June 30, 2026? | No | geopolitical_event | event_probability_model | 0.8200 | 0.8300 | 0.0100 | 101.6404 | 1217226 | 155.145995 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | geopolitical_event | event_probability_model | 0.1000 | 0.1100 | 0.0100 | 213.9081 | 882648 | 153.640911 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | No | geopolitical_event | event_probability_model | 0.8900 | 0.9000 | 0.0100 | 213.9081 | 882648 | 153.640911 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | No | geopolitical_event | event_probability_model | 0.8200 | 0.8300 | 0.0100 | 153.3195 | 474894 | 149.152466 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | Yes | geopolitical_event | event_probability_model | 0.1700 | 0.1800 | 0.0100 | 139.9125 | 474894 | 149.152466 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 30, 2026? | Yes | geopolitical_event | event_probability_model | 0.1700 | 0.1800 | 0.0100 | 81.5700 | 1217226 | 136.716035 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Stuttgart Open: Tommy Paul vs Giovanni Mpetshi Perricard | Yes | event_market | event_probability_model | 0.5000 | 0.5100 | 0.0100 | 109.6475 | 445238 | 126.299213 | paper_event_model_watch | depth exists but external signal source is not identified |
| Stuttgart Open: Tommy Paul vs Giovanni Mpetshi Perricard | No | event_market | event_probability_model | 0.4900 | 0.5000 | 0.0100 | 109.6475 | 445238 | 126.299213 | paper_event_model_watch | depth exists but external signal source is not identified |
| US-Iran nuclear deal by June 30? | Yes | geopolitical_event | event_probability_model | 0.1900 | 0.2000 | 0.0100 | 44.8869 | 452668 | 94.181810 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | No | geopolitical_event | event_probability_model | 0.8000 | 0.8100 | 0.0100 | 44.8869 | 452668 | 94.181810 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Libema Open: Mia Pohankova vs Clara Tauson | Yes | event_market | event_probability_model | 0.5700 | 0.5800 | 0.0100 | 36.6678 | 275795 | 65.588183 | paper_event_model_watch | depth exists but external signal source is not identified |
| Libema Open: Mia Pohankova vs Clara Tauson | No | event_market | event_probability_model | 0.4200 | 0.4300 | 0.0100 | 36.6678 | 275795 | 65.588183 | paper_event_model_watch | depth exists but external signal source is not identified |
| Israel closes its airspace by June 15? | Yes | geopolitical_event | event_probability_model | 0.1400 | 0.1600 | 0.0200 | 9.6371 | 2605423 | 64.177531 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Israel closes its airspace by June 15? | No | geopolitical_event | event_probability_model | 0.8400 | 0.8600 | 0.0200 | 9.6371 | 2605423 | 64.177531 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.6600 | 0.6700 | 0.0100 | 10.0004 | 3472578 | 58.152546 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
