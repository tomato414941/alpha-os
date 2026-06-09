# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| US x Iran permanent peace deal by June 15, 2026? | Yes | 0.0500 | 0.0600 | 0.0100 | 83159.90 | 72841.75 | 2722024.23 | 320751.83 | 320.6518 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 15, 2026? | No | 0.9400 | 0.9500 | 0.0100 | 72841.75 | 83159.90 | 320751.83 | 2722024.23 | 320.6518 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | 0.8600 | 0.8700 | 0.0100 | 7764.05 | 37188.82 | 179668.53 | 412059.73 | 179.5685 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | 0.1300 | 0.1400 | 0.0100 | 37188.82 | 7764.05 | 412059.73 | 179668.53 | 179.5685 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 92284.09 | 10313.18 | 395799.14 | 153085.40 | 152.9854 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 10313.18 | 92284.09 | 153085.40 | 395799.14 | 152.9854 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1800 | 0.1900 | 0.0100 | 21263.71 | 23889.26 | 531548.71 | 106441.73 | 106.3417 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8100 | 0.8200 | 0.0100 | 23889.26 | 21263.71 | 106441.73 | 531239.71 | 106.3417 | visible depth exists near both sides |
| Boston Red Sox vs. Tampa Bay Rays | No | 0.8200 | 0.8300 | 0.0100 | 9095.41 | 3200.00 | 90840.46 | 93549.04 | 90.7405 | visible depth exists near both sides |
| Boston Red Sox vs. Tampa Bay Rays | Yes | 0.1700 | 0.1800 | 0.0100 | 3200.00 | 9095.41 | 93503.04 | 90814.46 | 90.7145 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8000 | 0.8200 | 0.0200 | 6204.46 | 3567.03 | 52382.31 | 75390.28 | 52.1823 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1800 | 0.2000 | 0.0200 | 3567.03 | 6204.46 | 75390.28 | 52382.31 | 52.1823 | visible depth exists near both sides |
| HSBC Championships: Katie Boulter vs Leylah Fernandez | Yes | 0.3000 | 0.3100 | 0.0100 | 2715.28 | 4220.84 | 95724.82 | 49925.99 | 49.8260 | visible depth exists near both sides |
| HSBC Championships: Katie Boulter vs Leylah Fernandez | No | 0.6900 | 0.7000 | 0.0100 | 4220.84 | 2651.81 | 49925.99 | 95661.35 | 49.8260 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | 0.1410 | 0.1450 | 0.0040 | 467.36 | 7756.08 | 46447.65 | 65574.95 | 46.4077 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | 0.8550 | 0.8590 | 0.0040 | 7480.21 | 467.36 | 65299.08 | 46447.65 | 46.4077 | visible depth exists near both sides |
| Philadelphia Phillies vs. Toronto Blue Jays | Yes | 0.8100 | 0.8200 | 0.0100 | 13217.73 | 2384.28 | 19152.32 | 27890.62 | 19.0523 | visible depth exists near both sides |
| Philadelphia Phillies vs. Toronto Blue Jays | No | 0.1800 | 0.1900 | 0.0100 | 2384.28 | 13217.73 | 27890.62 | 19152.32 | 19.0523 | visible depth exists near both sides |
| Will Vitality win IEM Cologne Major 2026? | Yes | 0.4400 | 0.4700 | 0.0300 | 90467.29 | 2778.49 | 102379.64 | 19075.66 | 18.7757 | visible depth exists near both sides |
| New York Yankees vs. Cleveland Guardians | Yes | 0.2600 | 0.2700 | 0.0100 | 2039.26 | 2340.43 | 28888.71 | 16732.08 | 16.6321 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
