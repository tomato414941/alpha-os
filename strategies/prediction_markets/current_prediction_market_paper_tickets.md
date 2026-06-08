# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| US x Iran permanent peace deal by June 30, 2026? | Yes | geopolitical_event | event_probability_model | 0.1600 | 0.1700 | 0.0100 | 117.7805 | 1124183 | 155.309617 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 30, 2026? | No | geopolitical_event | event_probability_model | 0.8300 | 0.8400 | 0.0100 | 117.7805 | 1124183 | 155.309617 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| HSBC Championships: Katie Boulter vs Leylah Fernandez | Yes | event_market | event_probability_model | 0.2700 | 0.2800 | 0.0100 | 73.3196 | 320825 | 98.588070 | paper_event_model_watch | depth exists but external signal source is not identified |
| HSBC Championships: Katie Boulter vs Leylah Fernandez | No | event_market | event_probability_model | 0.7200 | 0.7300 | 0.0100 | 73.3196 | 320825 | 98.588070 | paper_event_model_watch | depth exists but external signal source is not identified |
| US-Iran nuclear deal by June 30? | Yes | geopolitical_event | event_probability_model | 0.2000 | 0.2100 | 0.0100 | 43.8462 | 344002 | 91.314669 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | No | geopolitical_event | event_probability_model | 0.7900 | 0.8000 | 0.0100 | 43.8462 | 344002 | 91.314669 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Israel closes its airspace by June 15? | Yes | geopolitical_event | event_probability_model | 0.1500 | 0.1600 | 0.0100 | 23.9581 | 1987780 | 86.418684 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Israel closes its airspace by June 15? | No | geopolitical_event | event_probability_model | 0.8400 | 0.8500 | 0.0100 | 23.9581 | 1987780 | 86.418684 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.8300 | 0.8400 | 0.0100 | 28.1883 | 4042690 | 82.663451 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.1600 | 0.1700 | 0.0100 | 28.1883 | 4042690 | 82.663451 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.1580 | 0.1590 | 0.0010 | 12.3413 | 4415948 | 67.488262 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.8410 | 0.8420 | 0.0010 | 12.3413 | 4415948 | 67.488262 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Israel closes its airspace by June 30? | No | geopolitical_event | event_probability_model | 0.7600 | 0.7700 | 0.0100 | 4.9069 | 541904 | 61.645173 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US announces new Iran agreement/ceasefire extension by June 12? | No | geopolitical_event | event_probability_model | 0.8800 | 0.9100 | 0.0300 | 7.2159 | 284259 | 52.440505 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | geopolitical_event | event_probability_model | 0.0900 | 0.1200 | 0.0300 | 5.4804 | 284259 | 50.704985 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
