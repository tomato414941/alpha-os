# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| US x Iran permanent peace deal by June 15, 2026? | Yes | geopolitical_event | event_probability_model | 0.0500 | 0.0600 | 0.0100 | 290.7698 | 2407739 | 155.633087 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 15, 2026? | No | geopolitical_event | event_probability_model | 0.9400 | 0.9500 | 0.0100 | 290.7698 | 2407739 | 155.633087 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | geopolitical_event | event_probability_model | 0.0900 | 0.1000 | 0.0100 | 167.7378 | 597684 | 150.803398 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | No | geopolitical_event | event_probability_model | 0.9000 | 0.9100 | 0.0100 | 167.7378 | 597684 | 150.803398 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.8600 | 0.8700 | 0.0100 | 107.0550 | 5733651 | 149.834740 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.1300 | 0.1400 | 0.0100 | 107.0550 | 5733651 | 149.834740 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.1350 | 0.1420 | 0.0070 | 75.2990 | 5632659 | 125.161047 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.8580 | 0.8650 | 0.0070 | 75.2638 | 5632659 | 125.125827 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | No | geopolitical_event | event_probability_model | 0.8000 | 0.8200 | 0.0200 | 63.7775 | 305532 | 110.442158 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | Yes | geopolitical_event | event_probability_model | 0.1800 | 0.2000 | 0.0200 | 63.7775 | 305532 | 110.442158 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Israel closes its airspace by June 15? | Yes | geopolitical_event | event_probability_model | 0.1200 | 0.1300 | 0.0100 | 21.1441 | 1480091 | 76.537547 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Israel closes its airspace by June 15? | No | geopolitical_event | event_probability_model | 0.8700 | 0.8800 | 0.0100 | 21.1441 | 1480091 | 76.537547 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US announces new Iran agreement/ceasefire extension by June 12? | No | geopolitical_event | event_probability_model | 0.8800 | 0.8900 | 0.0100 | 10.8567 | 246898 | 57.795443 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | geopolitical_event | event_probability_model | 0.1100 | 0.1200 | 0.0100 | 10.6545 | 246898 | 57.593223 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| HSBC Championships: Katie Boulter vs Leylah Fernandez | Yes | event_market | event_probability_model | 0.2900 | 0.3000 | 0.0100 | 24.3908 | 401460 | 50.154450 | paper_event_model_watch | depth exists but external signal source is not identified |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
