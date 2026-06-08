# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | geopolitical_event | event_probability_model | 0.1000 | 0.1100 | 0.0100 | 116.1052 | 981151 | 154.661354 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | No | geopolitical_event | event_probability_model | 0.8900 | 0.9000 | 0.0100 | 116.1052 | 981151 | 154.661354 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | No | geopolitical_event | event_probability_model | 0.8200 | 0.8300 | 0.0100 | 148.0760 | 428163 | 148.654428 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | Yes | geopolitical_event | event_probability_model | 0.1700 | 0.1800 | 0.0100 | 134.8690 | 428163 | 148.654428 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | No | geopolitical_event | event_probability_model | 0.7800 | 0.8000 | 0.0200 | 53.3877 | 433698 | 101.058864 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | Yes | geopolitical_event | event_probability_model | 0.2000 | 0.2200 | 0.0200 | 41.2418 | 433698 | 88.912994 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.2300 | 0.2400 | 0.0100 | 26.3929 | 3257111 | 76.477898 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.7600 | 0.7700 | 0.0100 | 26.2929 | 3257111 | 76.377898 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US announces new Iran agreement/ceasefire extension by June 9? | Yes | geopolitical_event | event_probability_model | 0.0970 | 0.0980 | 0.0010 | 24.0758 | 293954 | 72.356248 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US announces new Iran agreement/ceasefire extension by June 9? | No | geopolitical_event | event_probability_model | 0.9020 | 0.9030 | 0.0010 | 24.0758 | 293954 | 72.356248 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | geopolitical_event | event_probability_model | 0.2000 | 0.2200 | 0.0200 | 5.9703 | 269961 | 53.404969 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US announces new Iran agreement/ceasefire extension by June 12? | No | geopolitical_event | event_probability_model | 0.7800 | 0.8000 | 0.0200 | 5.9703 | 269961 | 53.404969 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Bab el-Mandeb Strait effectively closed by June 30? | Yes | event_market | event_probability_model | 0.1030 | 0.1130 | 0.0100 | 10.7492 | 238107 | 32.438196 | paper_event_model_watch | depth exists but external signal source is not identified |
| Bab el-Mandeb Strait effectively closed by June 30? | No | event_market | event_probability_model | 0.8870 | 0.8970 | 0.0100 | 10.7492 | 238107 | 32.438196 | paper_event_model_watch | depth exists but external signal source is not identified |
| Iran closes its airspace by June 8? | Yes | geopolitical_event | none | 0.9990 | 0.0000 | 0.0000 | 0.0000 | 8286239 | -3.582898 | too_thin | visible near-top depth is too thin |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
