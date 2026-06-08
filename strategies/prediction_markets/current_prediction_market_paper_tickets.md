# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| US x Iran permanent peace deal by June 30, 2026? | No | geopolitical_event | event_probability_model | 0.8200 | 0.8300 | 0.0100 | 99.6375 | 1213217 | 154.782618 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | geopolitical_event | event_probability_model | 0.1000 | 0.1100 | 0.0100 | 222.3849 | 975344 | 154.606566 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Strait of Hormuz traffic returns to normal by end of June? | No | geopolitical_event | event_probability_model | 0.8900 | 0.9000 | 0.0100 | 222.3849 | 975344 | 154.606566 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | No | geopolitical_event | event_probability_model | 0.8200 | 0.8300 | 0.0100 | 126.2233 | 469896 | 149.111836 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will the U.S. invade Iran before 2027? | Yes | geopolitical_event | event_probability_model | 0.1700 | 0.1800 | 0.0100 | 112.8163 | 469896 | 149.111836 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US x Iran permanent peace deal by June 30, 2026? | Yes | geopolitical_event | event_probability_model | 0.1700 | 0.1800 | 0.0100 | 80.3671 | 1213217 | 135.512258 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Stuttgart Open: Tommy Paul vs Giovanni Mpetshi Perricard | Yes | event_market | event_probability_model | 0.5000 | 0.5100 | 0.0100 | 109.6638 | 445085 | 126.298900 | paper_event_model_watch | depth exists but external signal source is not identified |
| Stuttgart Open: Tommy Paul vs Giovanni Mpetshi Perricard | No | event_market | event_probability_model | 0.4900 | 0.5000 | 0.0100 | 109.6638 | 445085 | 126.298900 | paper_event_model_watch | depth exists but external signal source is not identified |
| US-Iran nuclear deal by June 30? | Yes | geopolitical_event | event_probability_model | 0.1900 | 0.2000 | 0.0100 | 54.3869 | 452297 | 103.669938 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| US-Iran nuclear deal by June 30? | No | geopolitical_event | event_probability_model | 0.8000 | 0.8100 | 0.0100 | 54.3869 | 452297 | 103.669938 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.6700 | 0.6900 | 0.0200 | 29.5507 | 3462936 | 77.103225 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.3100 | 0.3300 | 0.0200 | 29.5507 | 3462936 | 77.103225 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | political_event | event_probability_model | 0.6800 | 0.6810 | 0.0010 | 13.9821 | 3260617 | 63.028492 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | political_event | event_probability_model | 0.3190 | 0.3200 | 0.0010 | 13.1454 | 3260617 | 62.191772 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Libema Open: Marin Cilic vs Denis Shapovalov | No | event_market | event_probability_model | 0.6200 | 0.6300 | 0.0100 | 33.2383 | 376715 | 62.128336 | paper_event_model_watch | depth exists but external signal source is not identified |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
