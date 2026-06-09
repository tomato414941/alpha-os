# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| US x Iran permanent peace deal by December 31, 2026? | Yes | 0.6800 | 0.6900 | 0.0100 | 6499.30 | 69959.86 | 333684.13 | 350384.18 | 333.5841 | visible depth exists near both sides |
| US x Iran permanent peace deal by December 31, 2026? | No | 0.3100 | 0.3200 | 0.0100 | 69959.86 | 6499.30 | 350384.18 | 333684.13 | 333.5841 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 15, 2026? | Yes | 0.0500 | 0.0600 | 0.0100 | 66362.69 | 73776.65 | 2726021.34 | 328022.40 | 327.9224 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 15, 2026? | No | 0.9400 | 0.9500 | 0.0100 | 73776.65 | 66362.69 | 328022.40 | 2725821.34 | 327.9224 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | 0.8600 | 0.8700 | 0.0100 | 114714.27 | 10195.55 | 312840.55 | 483400.26 | 312.7405 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | 0.1300 | 0.1400 | 0.0100 | 10195.55 | 114714.27 | 483400.26 | 312840.55 | 312.7405 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 132486.48 | 11000.29 | 707923.14 | 156711.11 | 156.6111 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 11000.29 | 132486.48 | 156711.11 | 707923.14 | 156.6111 | visible depth exists near both sides |
| Spurs vs. Knicks: O/U 216.5 | Yes | 0.4700 | 0.4800 | 0.0100 | 59278.01 | 63635.93 | 162887.84 | 122140.98 | 122.0410 | visible depth exists near both sides |
| Spurs vs. Knicks: O/U 216.5 | No | 0.5200 | 0.5300 | 0.0100 | 63635.93 | 59278.01 | 120750.98 | 162887.84 | 120.6510 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1800 | 0.1900 | 0.0100 | 21763.71 | 19145.57 | 536768.85 | 102312.81 | 102.2128 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8100 | 0.8200 | 0.0100 | 19145.57 | 21763.71 | 102312.81 | 536768.85 | 102.2128 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | 0.1290 | 0.1380 | 0.0090 | 5787.36 | 2214.75 | 65039.17 | 107872.39 | 64.9492 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | 0.8620 | 0.8710 | 0.0090 | 2214.75 | 5787.36 | 107872.39 | 65039.17 | 64.9492 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1900 | 0.2000 | 0.0100 | 232.22 | 6054.24 | 79539.11 | 61069.35 | 60.9693 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8000 | 0.8100 | 0.0100 | 6054.24 | 232.22 | 61069.35 | 79507.74 | 60.9693 | visible depth exists near both sides |
| Philadelphia Phillies vs. Toronto Blue Jays | Yes | 0.9000 | 0.9100 | 0.0100 | 15043.77 | 3692.90 | 34105.46 | 54265.20 | 34.0055 | visible depth exists near both sides |
| Philadelphia Phillies vs. Toronto Blue Jays | No | 0.0900 | 0.1000 | 0.0100 | 3692.90 | 15043.77 | 54265.20 | 34105.46 | 34.0055 | visible depth exists near both sides |
| New York Yankees vs. Cleveland Guardians | No | 0.4300 | 0.4400 | 0.0100 | 15315.51 | 4340.00 | 31512.32 | 33194.59 | 31.4123 | visible depth exists near both sides |
| New York Yankees vs. Cleveland Guardians | Yes | 0.5600 | 0.5700 | 0.0100 | 4340.00 | 15154.17 | 33194.59 | 31350.98 | 31.2510 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
