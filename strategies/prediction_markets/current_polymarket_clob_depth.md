# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| US x Iran permanent peace deal by June 15, 2026? | Yes | 0.0500 | 0.0600 | 0.0100 | 101365.79 | 30115.23 | 2794789.08 | 273946.71 | 273.8467 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 15, 2026? | No | 0.9400 | 0.9500 | 0.0100 | 30115.23 | 101365.79 | 273946.71 | 2794789.08 | 273.8467 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | 0.8800 | 0.8900 | 0.0100 | 22822.00 | 12756.44 | 236995.05 | 278169.60 | 236.8950 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | 0.1100 | 0.1200 | 0.0100 | 12756.44 | 22822.00 | 278169.60 | 208484.05 | 208.3841 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 128744.43 | 24660.52 | 702082.95 | 181416.57 | 181.3166 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 24660.52 | 128744.43 | 181416.57 | 702082.95 | 181.3166 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1800 | 0.1900 | 0.0100 | 1545.63 | 15009.90 | 448691.18 | 96410.17 | 96.3102 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8100 | 0.8200 | 0.0100 | 15009.90 | 1545.63 | 96410.17 | 448691.18 | 96.3102 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1600 | 0.1700 | 0.0100 | 36744.79 | 4980.98 | 863907.62 | 74427.62 | 74.3276 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8300 | 0.8400 | 0.0100 | 4980.98 | 36744.79 | 74427.62 | 863907.62 | 74.3276 | visible depth exists near both sides |
| Seattle Mariners vs. Baltimore Orioles | Yes | 0.5400 | 0.5500 | 0.0100 | 45300.45 | 41223.43 | 76749.70 | 68539.80 | 68.4398 | visible depth exists near both sides |
| Seattle Mariners vs. Baltimore Orioles | No | 0.4500 | 0.4600 | 0.0100 | 41223.43 | 45300.45 | 68539.80 | 77964.70 | 68.4398 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.2000 | 0.2100 | 0.0100 | 119.00 | 8041.35 | 57750.24 | 42986.94 | 42.8869 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.7900 | 0.8000 | 0.0100 | 8041.35 | 119.00 | 42786.94 | 73876.35 | 42.6869 | visible depth exists near both sides |
| HSBC Championships: Katie Boulter vs Leylah Fernandez | Yes | 0.3000 | 0.3100 | 0.0100 | 2201.40 | 4260.06 | 68419.42 | 42559.94 | 42.4599 | visible depth exists near both sides |
| HSBC Championships: Katie Boulter vs Leylah Fernandez | No | 0.6900 | 0.7000 | 0.0100 | 4260.06 | 2201.40 | 42559.94 | 68419.42 | 42.4599 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | Yes | 0.1600 | 0.1700 | 0.0100 | 388.47 | 2562.12 | 50960.65 | 32323.60 | 32.2236 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | No | 0.8300 | 0.8400 | 0.0100 | 2562.12 | 388.47 | 32323.60 | 50960.65 | 32.2236 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | 0.1180 | 0.1250 | 0.0070 | 58.70 | 8117.97 | 20211.76 | 131230.14 | 20.1418 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | 0.8750 | 0.8820 | 0.0070 | 8117.97 | 58.70 | 122830.14 | 20211.76 | 20.1418 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
