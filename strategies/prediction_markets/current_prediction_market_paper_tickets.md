# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| US x Iran permanent peace deal by June 30, 2026? | No | geopolitical_event | event_probability_model | 0.8500 | 0.8600 | 0.0100 | 206.5537 | 1103861 | 154.897401 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 30, 2026? | Yes | geopolitical_event | event_probability_model | 0.1400 | 0.1500 | 0.0100 | 186.4461 | 1103861 | 154.897401 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | geopolitical_event | event_probability_model | 0.0900 | 0.1000 | 0.0100 | 241.1148 | 787546 | 153.242517 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | No | geopolitical_event | event_probability_model | 0.9000 | 0.9100 | 0.0100 | 241.1148 | 787546 | 153.242517 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | No | geopolitical_event | event_probability_model | 0.8200 | 0.8300 | 0.0100 | 140.1961 | 363453 | 147.905784 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | Yes | geopolitical_event | event_probability_model | 0.1700 | 0.1800 | 0.0100 | 126.9891 | 363453 | 147.905784 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by July 31, 2026? | Yes | geopolitical_event | event_probability_model | 0.2900 | 0.3000 | 0.0100 | 111.8661 | 291837 | 147.061834 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by July 31, 2026? | No | geopolitical_event | event_probability_model | 0.7000 | 0.7100 | 0.0100 | 111.8661 | 291837 | 147.061834 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | Yes | sports_event | maker_research | 0.1900 | 0.2000 | 0.0100 | 332.1495 | 337538 | 117.338600 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | No | sports_event | maker_research | 0.8000 | 0.8100 | 0.0100 | 332.1495 | 337538 | 117.338600 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| Strait of Hormuz traffic returns to normal by July 31? | No | geopolitical_event | event_probability_model | 0.7100 | 0.7200 | 0.0100 | 49.5234 | 194474 | 95.353588 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | geopolitical_event | event_probability_model | 0.2800 | 0.2900 | 0.0100 | 47.5192 | 194474 | 93.349408 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | Yes | geopolitical_event | event_probability_model | 0.1900 | 0.2000 | 0.0100 | 40.2922 | 344353 | 88.597702 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | No | geopolitical_event | event_probability_model | 0.8000 | 0.8100 | 0.0100 | 40.2922 | 344353 | 88.597702 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.6700 | 0.6800 | 0.0100 | 17.6025 | 2472281 | 65.964030 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
