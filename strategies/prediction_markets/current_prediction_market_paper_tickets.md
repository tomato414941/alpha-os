# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | geopolitical_event | event_probability_model | 0.0900 | 0.1000 | 0.0100 | 241.4217 | 774421 | 152.910794 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | No | geopolitical_event | event_probability_model | 0.9000 | 0.9100 | 0.0100 | 241.4217 | 774421 | 152.910794 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | No | geopolitical_event | event_probability_model | 0.8200 | 0.8300 | 0.0100 | 169.1343 | 362000 | 147.915250 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | Yes | geopolitical_event | event_probability_model | 0.1700 | 0.1800 | 0.0100 | 155.9273 | 362000 | 147.915250 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by July 31, 2026? | No | geopolitical_event | event_probability_model | 0.7100 | 0.7200 | 0.0100 | 126.1595 | 228094 | 146.530510 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by July 31, 2026? | Yes | geopolitical_event | event_probability_model | 0.2800 | 0.2900 | 0.0100 | 119.3421 | 228094 | 146.530510 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by July 31? | No | geopolitical_event | event_probability_model | 0.7100 | 0.7200 | 0.0100 | 76.0951 | 185295 | 121.939210 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | Yes | sports_event | maker_research | 0.1900 | 0.2000 | 0.0100 | 318.1782 | 397512 | 118.023834 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | No | sports_event | maker_research | 0.8000 | 0.8100 | 0.0100 | 318.1782 | 397512 | 118.023834 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | geopolitical_event | event_probability_model | 0.2800 | 0.2900 | 0.0100 | 71.8690 | 185295 | 117.713030 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | No | geopolitical_event | event_probability_model | 0.8200 | 0.8300 | 0.0100 | 55.1897 | 350776 | 103.553403 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | Yes | geopolitical_event | event_probability_model | 0.1700 | 0.1800 | 0.0100 | 49.9000 | 350776 | 98.263703 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | geopolitical_event | event_probability_model | 0.1500 | 0.1600 | 0.0100 | 21.6187 | 186305 | 67.904846 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US announces new Iran agreement/ceasefire extension by June 12? | No | geopolitical_event | event_probability_model | 0.8400 | 0.8500 | 0.0100 | 21.6187 | 186305 | 67.904846 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Israel closes its airspace by June 15? | Yes | geopolitical_event | event_probability_model | 0.4500 | 0.4600 | 0.0100 | 5.2076 | 2253445 | 67.829739 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
