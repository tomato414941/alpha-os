# Current Prediction Market Paper Tickets

This converts current Polymarket microstructure and CLOB depth into research paper tickets. It is not a live trade instruction and does not estimate true event probability.

| question | outcome | category | structure | bid | ask | spread | depth score | volume 24h | score | status | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Iran closes its airspace by June 8? | Yes | geopolitical_event | none | 0.9960 | 0.9970 | 0.0010 | 634.3619 | 5702309 | 25.687561 | near_certain_event | market is too close to expiry/certainty for clean research |
| Iran closes its airspace by June 8? | No | geopolitical_event | none | 0.0030 | 0.0040 | 0.0010 | 634.3619 | 5702309 | 25.687561 | near_certain_event | market is too close to expiry/certainty for clean research |
| Iran closes its airspace by June 15? | Yes | geopolitical_event | none | 0.9970 | 0.9980 | 0.0010 | 143.6314 | 970316 | 22.829452 | near_certain_event | market is too close to expiry/certainty for clean research |
| Iran closes its airspace by June 15? | No | geopolitical_event | none | 0.0020 | 0.0030 | 0.0010 | 143.6314 | 970316 | 22.829452 | near_certain_event | market is too close to expiry/certainty for clean research |
| Will Microstrategy announce a Bitcoin purchase June 2-8? | Yes | crypto_event | none | 0.9590 | 0.9700 | 0.0110 | 94.4224 | 648386 | 8.677036 | near_certain_event | market is too close to expiry/certainty for clean research |
| Will Microstrategy announce a Bitcoin purchase June 2-8? | No | crypto_event | none | 0.0300 | 0.0410 | 0.0110 | 94.4224 | 648386 | 8.677036 | near_certain_event | market is too close to expiry/certainty for clean research |
| Iran closes its airspace by June 30? | Yes | geopolitical_event | none | 0.9970 | 0.9980 | 0.0010 | 87.9416 | 799015 | 7.096120 | near_certain_event | market is too close to expiry/certainty for clean research |
| Iran closes its airspace by June 30? | No | geopolitical_event | none | 0.0020 | 0.0030 | 0.0010 | 87.9416 | 799015 | 7.096120 | near_certain_event | market is too close to expiry/certainty for clean research |
| Israel closes its airspace by June 30? | Yes | geopolitical_event | none | 0.3000 | 0.3100 | 0.0100 | 4.7831 | 499036 | -23.016630 | too_thin | visible near-top depth is too thin |
| Israel closes its airspace by June 30? | No | geopolitical_event | none | 0.6900 | 0.7000 | 0.0100 | 4.7831 | 499036 | -23.016630 | too_thin | visible near-top depth is too thin |
| Iran closes its airspace by July 31? | Yes | geopolitical_event | none | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 262658 | -29.416910 | too_thin | visible near-top depth is too thin |
| Iran closes its airspace by July 31? | No | geopolitical_event | none | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 262658 | -29.416910 | too_thin | visible near-top depth is too thin |
| LoL: Fluxo W7M vs paiN Gaming (BO3) - Esports World Cup South America & LATAM Qualifier Playoffs | Yes | event_market | none | 0.9990 | 0.0000 | 0.0000 | 0.0000 | 219303 | -41.269736 | too_thin | visible near-top depth is too thin |
| PortlandFire vs. Los Angeles Sparks | No | sports_event | none | 0.9970 | 0.0000 | 0.0000 | 0.0000 | 205103 | -42.644415 | too_thin | visible near-top depth is too thin |
| LoL: Fluxo W7M vs paiN Gaming (BO3) - Esports World Cup South America & LATAM Qualifier Playoffs | No | event_market | none | 0.0000 | 0.0010 | 0.0000 | 0.0000 | 219303 | -51.269736 | too_thin | visible near-top depth is too thin |

## Caveat

Depth-positive event markets still need a true-probability model, news feed, latency checks, and adverse-selection monitoring. Sports rows are treated as market-making research unless a dedicated sports model is added.
