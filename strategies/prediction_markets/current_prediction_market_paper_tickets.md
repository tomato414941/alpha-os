# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Will Microstrategy announce a Bitcoin purchase June 2-8? | Yes | crypto_event | event_probability_model | 0.9630 | 0.9690 | 0.0060 | 87.5074 | 599551 | 141.598618 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Will Microstrategy announce a Bitcoin purchase June 2-8? | No | crypto_event | event_probability_model | 0.0310 | 0.0370 | 0.0060 | 87.5074 | 599551 | 141.598618 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Israel closes its airspace by June 30? | Yes | geopolitical_event | event_probability_model | 0.3500 | 0.3800 | 0.0300 | 7.1150 | 429573 | 60.898903 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Israel closes its airspace by June 30? | No | geopolitical_event | event_probability_model | 0.6200 | 0.6500 | 0.0300 | 7.1150 | 429573 | 60.898903 | paper_event_model_candidate | depth exists and the event can be tied to external information feeds |
| Iran closes its airspace by June 8? | Yes | geopolitical_event | none | 0.5120 | 0.5310 | 0.0190 | 4.4330 | 1166236 | 49.904901 | too_thin | visible near-top depth is too thin |
| Iran closes its airspace by June 8? | No | geopolitical_event | none | 0.4690 | 0.4880 | 0.0190 | 4.4330 | 1166236 | 49.904901 | too_thin | visible near-top depth is too thin |
| Iran closes its airspace by June 30? | No | geopolitical_event | none | 0.3290 | 0.3500 | 0.0210 | 0.0566 | 431874 | 38.080056 | too_thin | visible near-top depth is too thin |
| Iran closes its airspace by June 30? | Yes | geopolitical_event | none | 0.6500 | 0.6710 | 0.0210 | 0.0566 | 431874 | 38.080056 | too_thin | visible near-top depth is too thin |
| Iran closes its airspace by June 15? | No | geopolitical_event | none | 0.3540 | 0.4090 | 0.0550 | 1.8070 | 491191 | 36.639706 | too_thin | visible near-top depth is too thin |
| Iran closes its airspace by June 15? | Yes | geopolitical_event | none | 0.5910 | 0.6460 | 0.0550 | 1.8070 | 491191 | 36.639706 | too_thin | visible near-top depth is too thin |
| Chicago Sky vs. Toronto Tempo | Yes | sports_event | maker_research | 0.0400 | 0.0480 | 0.0080 | 15.8160 | 462561 | 31.359573 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| Chicago Sky vs. Toronto Tempo | No | sports_event | maker_research | 0.9520 | 0.9600 | 0.0080 | 15.8160 | 462561 | 31.359573 | sports_market_making_watch | sports market has depth, but needs a dedicated model |
| LoL: G2 Esports vs Karmine Corp - Game 4 Winner | Yes | sports_event | none | 0.9990 | 0.0000 | 0.0000 | 0.0000 | 746312 | 27.521386 | too_thin | visible near-top depth is too thin |
| Tampa Bay Rays vs. Miami Marlins | No | sports_event | none | 0.9990 | 0.0000 | 0.0000 | 0.0000 | 324760 | 23.789779 | too_thin | visible near-top depth is too thin |
| Valorant: Leviatán Esports vs Global Esports (BO3) - VCT Masters London Group Stage | Yes | event_market | none | 0.9990 | 0.0000 | 0.0000 | 0.0000 | 394134 | 22.165596 | too_thin | visible near-top depth is too thin |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
