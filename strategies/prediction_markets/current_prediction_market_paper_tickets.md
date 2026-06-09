# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| US x Iran permanent peace deal by June 15, 2026? | Yes | geopolitical_event | event_probability_model | 0.0500 | 0.0600 | 0.0100 | 320.6518 | 2379062 | 155.634018 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 15, 2026? | No | geopolitical_event | event_probability_model | 0.9400 | 0.9500 | 0.0100 | 320.6518 | 2379062 | 155.634018 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | geopolitical_event | event_probability_model | 0.0900 | 0.1000 | 0.0100 | 152.9854 | 611199 | 150.930226 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | No | geopolitical_event | event_probability_model | 0.9000 | 0.9100 | 0.0100 | 152.9854 | 611199 | 150.930226 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.8600 | 0.8700 | 0.0100 | 179.5685 | 6164636 | 149.850129 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.1300 | 0.1400 | 0.0100 | 179.5685 | 6164636 | 149.850129 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | Yes | geopolitical_event | event_probability_model | 0.1800 | 0.1900 | 0.0100 | 106.3417 | 363775 | 147.930868 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | No | geopolitical_event | event_probability_model | 0.8100 | 0.8200 | 0.0100 | 106.3417 | 363775 | 147.930868 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Boston Red Sox vs. Tampa Bay Rays | No | sports_event | maker_research | 0.8200 | 0.8300 | 0.0100 | 90.7405 | 440114 | 113.938048 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| Boston Red Sox vs. Tampa Bay Rays | Yes | sports_event | maker_research | 0.1700 | 0.1800 | 0.0100 | 90.7145 | 440114 | 113.912048 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| US-Iran nuclear deal by June 30? | No | geopolitical_event | event_probability_model | 0.8000 | 0.8200 | 0.0200 | 52.1823 | 316177 | 98.955240 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | Yes | geopolitical_event | event_probability_model | 0.1800 | 0.2000 | 0.0200 | 52.1823 | 316177 | 98.955240 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.1410 | 0.1450 | 0.0040 | 46.4076 | 5830536 | 96.360371 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.8550 | 0.8590 | 0.0040 | 46.4076 | 5830536 | 96.360371 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| HSBC Championships: Katie Boulter vs Leylah Fernandez | Yes | event_market | event_probability_model | 0.3000 | 0.3100 | 0.0100 | 49.8260 | 402859 | 75.582678 | paper_event_model_watch | depth exists but external signal source is not identified |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
