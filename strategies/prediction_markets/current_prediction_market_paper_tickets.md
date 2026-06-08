# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| US x Iran permanent peace deal by June 30, 2026? | Yes | geopolitical_event | event_probability_model | 0.1400 | 0.1500 | 0.0100 | 241.7012 | 1164420 | 154.945831 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 30, 2026? | No | geopolitical_event | event_probability_model | 0.8500 | 0.8600 | 0.0100 | 241.7012 | 1164420 | 154.945831 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | geopolitical_event | event_probability_model | 0.0900 | 0.1000 | 0.0100 | 240.3773 | 778899 | 152.958368 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | No | geopolitical_event | event_probability_model | 0.9000 | 0.9100 | 0.0100 | 240.3773 | 778899 | 152.958368 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | No | geopolitical_event | event_probability_model | 0.8200 | 0.8300 | 0.0100 | 167.1319 | 363697 | 147.936640 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | Yes | geopolitical_event | event_probability_model | 0.1700 | 0.1800 | 0.0100 | 153.9249 | 363697 | 147.936640 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by July 31, 2026? | No | geopolitical_event | event_probability_model | 0.7100 | 0.7200 | 0.0100 | 130.9228 | 229230 | 146.556394 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by July 31, 2026? | Yes | geopolitical_event | event_probability_model | 0.2800 | 0.2900 | 0.0100 | 121.3942 | 229230 | 146.556394 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | Yes | sports_event | maker_research | 0.1900 | 0.2000 | 0.0100 | 338.3326 | 502451 | 119.170705 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | No | sports_event | maker_research | 0.8000 | 0.8100 | 0.0100 | 338.3326 | 502451 | 119.170705 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | geopolitical_event | event_probability_model | 0.2700 | 0.2800 | 0.0100 | 71.7769 | 186250 | 117.825501 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by July 31? | No | geopolitical_event | event_probability_model | 0.7200 | 0.7300 | 0.0100 | 71.7769 | 186250 | 117.825501 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | No | geopolitical_event | event_probability_model | 0.8200 | 0.8300 | 0.0100 | 50.3753 | 353460 | 98.969412 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | Yes | geopolitical_event | event_probability_model | 0.1700 | 0.1800 | 0.0100 | 46.7674 | 353460 | 95.361442 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | geopolitical_event | event_probability_model | 0.1600 | 0.1700 | 0.0100 | 38.8646 | 185515 | 84.962263 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
