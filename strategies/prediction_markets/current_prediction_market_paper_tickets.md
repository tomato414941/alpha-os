# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| US x Iran permanent peace deal by June 15, 2026? | Yes | geopolitical_event | event_probability_model | 0.0500 | 0.0600 | 0.0100 | 273.8467 | 2441466 | 155.425933 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 15, 2026? | No | geopolitical_event | event_probability_model | 0.9400 | 0.9500 | 0.0100 | 273.8467 | 2441466 | 155.425933 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | geopolitical_event | event_probability_model | 0.0900 | 0.1000 | 0.0100 | 181.3166 | 621658 | 151.267781 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | No | geopolitical_event | event_probability_model | 0.9000 | 0.9100 | 0.0100 | 181.3166 | 621658 | 151.267781 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.8800 | 0.8900 | 0.0100 | 236.8950 | 5098413 | 150.275832 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.1100 | 0.1200 | 0.0100 | 208.3841 | 5098413 | 150.275832 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | Yes | geopolitical_event | event_probability_model | 0.1800 | 0.1900 | 0.0100 | 96.3102 | 383186 | 144.413790 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | No | geopolitical_event | event_probability_model | 0.8100 | 0.8200 | 0.0100 | 96.3102 | 383186 | 144.413790 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 30, 2026? | Yes | geopolitical_event | event_probability_model | 0.1600 | 0.1700 | 0.0100 | 74.3276 | 856966 | 127.664529 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 30, 2026? | No | geopolitical_event | event_probability_model | 0.8300 | 0.8400 | 0.0100 | 74.3276 | 856966 | 127.664529 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | Yes | geopolitical_event | event_probability_model | 0.2000 | 0.2100 | 0.0100 | 42.8869 | 305148 | 90.401343 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | No | geopolitical_event | event_probability_model | 0.7900 | 0.8000 | 0.0100 | 42.6869 | 305148 | 90.201343 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Israel closes its airspace by June 15? | Yes | geopolitical_event | event_probability_model | 0.1600 | 0.1700 | 0.0100 | 32.2236 | 1487472 | 88.175855 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Israel closes its airspace by June 15? | No | geopolitical_event | event_probability_model | 0.8300 | 0.8400 | 0.0100 | 32.2236 | 1487472 | 88.175855 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Seattle Mariners vs. Baltimore Orioles | Yes | sports_event | maker_research | 0.5400 | 0.5500 | 0.0100 | 68.4398 | 300942 | 84.490898 | sports_market_making_watch | sports market has depth, but needs a dedicated model |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
