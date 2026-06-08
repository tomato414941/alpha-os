# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| US x Iran permanent peace deal by June 30, 2026? | Yes | geopolitical_event | event_probability_model | 0.1400 | 0.1500 | 0.0100 | 283.9257 | 1196627 | 154.986229 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 30, 2026? | No | geopolitical_event | event_probability_model | 0.8500 | 0.8600 | 0.0100 | 283.9257 | 1196627 | 154.986229 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | geopolitical_event | event_probability_model | 0.1000 | 0.1100 | 0.0100 | 272.0489 | 951874 | 154.576029 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | No | geopolitical_event | event_probability_model | 0.8900 | 0.9000 | 0.0100 | 272.0489 | 951874 | 154.576029 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | No | geopolitical_event | event_probability_model | 0.8200 | 0.8300 | 0.0100 | 151.8432 | 387807 | 148.200113 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | Yes | geopolitical_event | event_probability_model | 0.1700 | 0.1800 | 0.0100 | 138.6362 | 387807 | 148.200113 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by July 31, 2026? | No | geopolitical_event | event_probability_model | 0.7100 | 0.7200 | 0.0100 | 103.5087 | 210460 | 146.228047 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by July 31, 2026? | Yes | geopolitical_event | event_probability_model | 0.2800 | 0.2900 | 0.0100 | 96.6002 | 210460 | 142.828207 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by July 31? | No | geopolitical_event | event_probability_model | 0.7100 | 0.7200 | 0.0100 | 66.7540 | 196051 | 112.519525 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | geopolitical_event | event_probability_model | 0.2800 | 0.2900 | 0.0100 | 64.9684 | 196051 | 110.733945 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | No | geopolitical_event | event_probability_model | 0.7900 | 0.8000 | 0.0100 | 37.1773 | 386747 | 85.735527 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | Yes | geopolitical_event | event_probability_model | 0.2000 | 0.2100 | 0.0100 | 34.1409 | 386747 | 82.699197 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.7800 | 0.7900 | 0.0100 | 31.9261 | 3210242 | 81.641995 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.2100 | 0.2200 | 0.0100 | 31.9261 | 3210242 | 81.641995 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.2200 | 0.2230 | 0.0030 | 26.0194 | 2805173 | 76.647138 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
