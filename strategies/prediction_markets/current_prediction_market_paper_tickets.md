# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| US x Iran permanent peace deal by December 31, 2026? | Yes | geopolitical_event | event_probability_model | 0.6800 | 0.6900 | 0.0100 | 333.5841 | 1964656 | 156.043556 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by December 31, 2026? | No | geopolitical_event | event_probability_model | 0.3100 | 0.3200 | 0.0100 | 333.5841 | 1964656 | 156.043556 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 15, 2026? | Yes | geopolitical_event | event_probability_model | 0.0500 | 0.0600 | 0.0100 | 327.9224 | 2378068 | 155.632640 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 15, 2026? | No | geopolitical_event | event_probability_model | 0.9400 | 0.9500 | 0.0100 | 327.9224 | 2378068 | 155.632640 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | geopolitical_event | event_probability_model | 0.0900 | 0.1000 | 0.0100 | 156.6111 | 597683 | 150.804836 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | No | geopolitical_event | event_probability_model | 0.9000 | 0.9100 | 0.0100 | 156.6111 | 597683 | 150.804836 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.8600 | 0.8700 | 0.0100 | 312.7405 | 6081546 | 149.741878 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.1300 | 0.1400 | 0.0100 | 312.7405 | 6081546 | 149.741878 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | Yes | geopolitical_event | event_probability_model | 0.1800 | 0.1900 | 0.0100 | 102.2128 | 363919 | 147.930412 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | No | geopolitical_event | event_probability_model | 0.8100 | 0.8200 | 0.0100 | 102.2128 | 363919 | 147.930412 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Spurs vs. Knicks: O/U 216.5 | Yes | sports_event | maker_research | 0.4700 | 0.4800 | 0.0100 | 122.0410 | 476102 | 118.041663 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| Spurs vs. Knicks: O/U 216.5 | No | sports_event | maker_research | 0.5200 | 0.5300 | 0.0100 | 120.6510 | 476102 | 118.041663 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.1290 | 0.1380 | 0.0090 | 64.9492 | 5747094 | 114.488511 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.8620 | 0.8710 | 0.0090 | 64.9492 | 5747094 | 114.488511 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | Yes | geopolitical_event | event_probability_model | 0.1900 | 0.2000 | 0.0100 | 60.9693 | 307452 | 108.860506 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
