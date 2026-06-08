# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| US-Iran nuclear deal by June 30? | Yes | geopolitical_event | event_probability_model | 0.2200 | 0.2300 | 0.0100 | 41.1093 | 427967 | 89.572559 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | No | geopolitical_event | event_probability_model | 0.7700 | 0.7800 | 0.0100 | 41.1093 | 427967 | 89.572559 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.7700 | 0.7800 | 0.0100 | 23.6025 | 3239338 | 73.684279 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.2200 | 0.2300 | 0.0100 | 23.6025 | 3239338 | 73.684279 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US announces new Iran agreement/ceasefire extension by June 9? | Yes | geopolitical_event | event_probability_model | 0.1020 | 0.1140 | 0.0120 | 20.5732 | 287382 | 67.961172 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US announces new Iran agreement/ceasefire extension by June 9? | No | geopolitical_event | event_probability_model | 0.8860 | 0.8980 | 0.0120 | 20.5732 | 287382 | 67.961172 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.2320 | 0.2350 | 0.0030 | 5.3335 | 2950024 | 55.660181 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.7650 | 0.7680 | 0.0030 | 5.3335 | 2950024 | 55.660181 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | geopolitical_event | event_probability_model | 0.2100 | 0.2300 | 0.0200 | 4.9547 | 268865 | 52.546956 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US announces new Iran agreement/ceasefire extension by June 12? | No | geopolitical_event | event_probability_model | 0.7700 | 0.7900 | 0.0200 | 4.9547 | 268865 | 52.546956 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Iran closes its airspace by June 8? | Yes | geopolitical_event | none | 0.9990 | 0.0000 | 0.0000 | 0.0000 | 8302263 | -3.440803 | too_thin | visible near-top depth is too thin |
| Iran closes its airspace by June 15? | Yes | geopolitical_event | none | 0.9990 | 0.0000 | 0.0000 | 0.0000 | 1731414 | -5.544727 | too_thin | visible near-top depth is too thin |
| Iran closes its airspace by June 30? | Yes | geopolitical_event | none | 0.9990 | 0.0000 | 0.0000 | 0.0000 | 1048651 | -7.921493 | too_thin | visible near-top depth is too thin |
| Iran closes its airspace by June 8? | No | geopolitical_event | none | 0.0000 | 0.0010 | 0.0000 | 0.0000 | 8302263 | -13.440803 | too_thin | visible near-top depth is too thin |
| Iran closes its airspace by June 15? | No | geopolitical_event | none | 0.0000 | 0.0010 | 0.0000 | 0.0000 | 1731414 | -15.544727 | too_thin | visible near-top depth is too thin |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
