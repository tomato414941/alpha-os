# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| US x Iran permanent peace deal by June 15, 2026? | Yes | geopolitical_event | event_probability_model | 0.0500 | 0.0600 | 0.0100 | 210.7070 | 2063624 | 155.316668 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 15, 2026? | No | geopolitical_event | event_probability_model | 0.9400 | 0.9500 | 0.0100 | 210.7070 | 2063624 | 155.316668 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 30, 2026? | Yes | geopolitical_event | event_probability_model | 0.1500 | 0.1600 | 0.0100 | 109.0312 | 888475 | 153.636936 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 30, 2026? | No | geopolitical_event | event_probability_model | 0.8400 | 0.8500 | 0.0100 | 109.0312 | 888475 | 153.636936 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | geopolitical_event | event_probability_model | 0.1000 | 0.1100 | 0.0100 | 149.8507 | 656303 | 151.226864 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | No | geopolitical_event | event_probability_model | 0.8900 | 0.9000 | 0.0100 | 149.8507 | 656303 | 151.226864 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by July 31, 2026? | Yes | geopolitical_event | event_probability_model | 0.3000 | 0.3100 | 0.0100 | 85.8183 | 288929 | 133.243011 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by July 31, 2026? | No | geopolitical_event | event_probability_model | 0.6900 | 0.7000 | 0.0100 | 85.8183 | 288929 | 133.243011 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | Yes | sports_event | maker_research | 0.1900 | 0.2000 | 0.0100 | 111.1483 | 188194 | 114.993114 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | No | sports_event | maker_research | 0.8000 | 0.8100 | 0.0100 | 111.1483 | 188194 | 114.993114 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| Israel closes its airspace by June 15? | Yes | geopolitical_event | event_probability_model | 0.5500 | 0.5800 | 0.0300 | 10.3332 | 1570287 | 64.620372 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Israel closes its airspace by June 15? | No | geopolitical_event | event_probability_model | 0.4200 | 0.4500 | 0.0300 | 10.3332 | 1570287 | 64.620372 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.4500 | 0.4600 | 0.0100 | 7.3332 | 1014857 | 57.016357 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.5400 | 0.5500 | 0.0100 | 6.2492 | 1014857 | 55.932357 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.5500 | 0.5780 | 0.0280 | 7.9732 | 964619 | 55.354384 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
