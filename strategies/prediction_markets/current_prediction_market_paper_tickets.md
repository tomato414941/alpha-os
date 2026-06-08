# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| US x Iran permanent peace deal by June 30, 2026? | Yes | geopolitical_event | event_probability_model | 0.1400 | 0.1500 | 0.0100 | 235.9524 | 1171147 | 154.928724 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 30, 2026? | No | geopolitical_event | event_probability_model | 0.8500 | 0.8600 | 0.0100 | 235.9524 | 1171147 | 154.928724 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | geopolitical_event | event_probability_model | 0.0900 | 0.1000 | 0.0100 | 239.7011 | 774195 | 152.908417 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | No | geopolitical_event | event_probability_model | 0.9000 | 0.9100 | 0.0100 | 239.7011 | 774195 | 152.908417 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | No | geopolitical_event | event_probability_model | 0.8200 | 0.8300 | 0.0100 | 165.3919 | 362006 | 147.916970 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | Yes | geopolitical_event | event_probability_model | 0.1700 | 0.1800 | 0.0100 | 152.1849 | 362006 | 147.916970 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by July 31, 2026? | No | geopolitical_event | event_probability_model | 0.7100 | 0.7200 | 0.0100 | 132.2321 | 228032 | 146.540444 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by July 31, 2026? | Yes | geopolitical_event | event_probability_model | 0.2800 | 0.2900 | 0.0100 | 123.7433 | 228032 | 146.540444 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by July 31? | No | geopolitical_event | event_probability_model | 0.7100 | 0.7200 | 0.0100 | 74.8067 | 185295 | 120.650720 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | Yes | sports_event | maker_research | 0.1900 | 0.2000 | 0.0100 | 314.5348 | 397512 | 118.023834 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | No | sports_event | maker_research | 0.8000 | 0.8100 | 0.0100 | 314.5348 | 397512 | 118.023834 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | geopolitical_event | event_probability_model | 0.2800 | 0.2900 | 0.0100 | 70.5805 | 185295 | 116.424540 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | No | geopolitical_event | event_probability_model | 0.8200 | 0.8300 | 0.0100 | 47.3526 | 350776 | 95.716323 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | Yes | geopolitical_event | event_probability_model | 0.1700 | 0.1800 | 0.0100 | 44.5907 | 350776 | 92.954353 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | geopolitical_event | event_probability_model | 0.1500 | 0.1600 | 0.0100 | 32.2252 | 187993 | 78.641949 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
