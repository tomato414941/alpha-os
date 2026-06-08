# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| US x Iran permanent peace deal by June 15, 2026? | Yes | geopolitical_event | event_probability_model | 0.0500 | 0.0600 | 0.0100 | 291.7520 | 2409467 | 155.635279 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 15, 2026? | No | geopolitical_event | event_probability_model | 0.9400 | 0.9500 | 0.0100 | 291.7520 | 2409467 | 155.635279 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | geopolitical_event | event_probability_model | 0.0900 | 0.1000 | 0.0100 | 168.1930 | 605062 | 151.089447 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | No | geopolitical_event | event_probability_model | 0.9000 | 0.9100 | 0.0100 | 168.1930 | 605062 | 151.089447 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.1240 | 0.1250 | 0.0010 | 111.1499 | 5536204 | 150.606419 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.8750 | 0.8760 | 0.0010 | 111.1499 | 5536204 | 150.606419 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.8800 | 0.8900 | 0.0100 | 196.0500 | 5689507 | 149.869054 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.1100 | 0.1200 | 0.0100 | 191.3120 | 5689507 | 149.869054 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | Yes | geopolitical_event | event_probability_model | 0.1800 | 0.1900 | 0.0100 | 96.4738 | 376045 | 144.527510 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | No | geopolitical_event | event_probability_model | 0.8100 | 0.8200 | 0.0100 | 96.4738 | 376045 | 144.527510 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Philadelphia Phillies vs. Toronto Blue Jays | Yes | sports_event | maker_research | 0.8200 | 0.8300 | 0.0100 | 99.9890 | 374337 | 117.073245 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| Philadelphia Phillies vs. Toronto Blue Jays | No | sports_event | maker_research | 0.1700 | 0.1800 | 0.0100 | 98.7905 | 374337 | 115.874755 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| US-Iran nuclear deal by June 30? | Yes | geopolitical_event | event_probability_model | 0.1800 | 0.1900 | 0.0100 | 65.6955 | 301255 | 113.501059 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | No | geopolitical_event | event_probability_model | 0.8100 | 0.8200 | 0.0100 | 65.6955 | 301255 | 113.501059 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Israel closes its airspace by June 15? | Yes | geopolitical_event | event_probability_model | 0.1200 | 0.1300 | 0.0100 | 18.4402 | 1501012 | 75.223611 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
