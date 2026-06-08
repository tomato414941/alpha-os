# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | 0.8800 | 0.8900 | 0.0100 | 47009.24 | 19564.98 | 265898.52 | 500765.02 | 265.7985 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 15, 2026? | Yes | 0.0500 | 0.0600 | 0.0100 | 84206.81 | 51267.27 | 2799692.11 | 251312.92 | 251.2129 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 15, 2026? | No | 0.9400 | 0.9500 | 0.0100 | 51267.27 | 84206.81 | 251312.92 | 2799692.11 | 251.2129 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | 0.1100 | 0.1200 | 0.0100 | 19564.98 | 47009.24 | 500765.02 | 237860.52 | 237.7605 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 128559.60 | 24677.13 | 691408.12 | 179759.44 | 179.6594 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 24677.13 | 128559.60 | 179759.44 | 691408.12 | 179.6594 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1800 | 0.1900 | 0.0100 | 1499.15 | 14935.38 | 445149.81 | 95079.45 | 94.9794 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8100 | 0.8200 | 0.0100 | 14935.38 | 1499.15 | 95079.45 | 445149.81 | 94.9794 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | 0.1130 | 0.1170 | 0.0040 | 1999.97 | 122.56 | 77575.38 | 124690.81 | 77.5354 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | 0.8830 | 0.8870 | 0.0040 | 122.56 | 1999.97 | 124690.81 | 77575.38 | 77.5354 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1800 | 0.1900 | 0.0100 | 5560.46 | 1150.26 | 75032.73 | 62342.73 | 62.2427 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8100 | 0.8200 | 0.0100 | 1150.26 | 5560.46 | 62342.73 | 75032.73 | 62.2427 | visible depth exists near both sides |
| Boston Red Sox vs. Tampa Bay Rays | Yes | 0.4700 | 0.4800 | 0.0100 | 17715.08 | 3942.85 | 41087.18 | 39730.73 | 39.6307 | visible depth exists near both sides |
| Boston Red Sox vs. Tampa Bay Rays | No | 0.5200 | 0.5300 | 0.0100 | 3942.85 | 17715.08 | 39668.73 | 41087.18 | 39.5687 | visible depth exists near both sides |
| HSBC Championships: Katie Boulter vs Leylah Fernandez | Yes | 0.3000 | 0.3100 | 0.0100 | 153.57 | 4225.50 | 66214.29 | 27554.94 | 27.4549 | visible depth exists near both sides |
| HSBC Championships: Katie Boulter vs Leylah Fernandez | No | 0.6900 | 0.7000 | 0.0100 | 4225.50 | 153.57 | 27554.94 | 66214.29 | 27.4549 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | Yes | 0.1200 | 0.1300 | 0.0100 | 11782.01 | 5120.44 | 92358.21 | 24186.19 | 24.0862 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | No | 0.8700 | 0.8800 | 0.0100 | 5120.44 | 11782.01 | 24186.19 | 92358.21 | 24.0862 | visible depth exists near both sides |
| New York Yankees vs. Cleveland Guardians | Yes | 0.7500 | 0.7600 | 0.0100 | 9899.36 | 2058.00 | 21706.73 | 32666.95 | 21.6067 | visible depth exists near both sides |
| New York Yankees vs. Cleveland Guardians | No | 0.2400 | 0.2500 | 0.0100 | 2058.00 | 9899.36 | 32666.95 | 21706.73 | 21.6067 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
