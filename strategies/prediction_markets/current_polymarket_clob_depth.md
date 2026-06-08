# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1400 | 0.1500 | 0.0100 | 69234.94 | 70288.81 | 1098684.87 | 284025.69 | 283.9257 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8500 | 0.8600 | 0.0100 | 70288.81 | 69234.94 | 284025.69 | 1121547.44 | 283.9257 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.1000 | 0.1100 | 0.0100 | 2292.18 | 84876.66 | 535586.20 | 272148.86 | 272.0489 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.8900 | 0.9000 | 0.0100 | 84876.66 | 2292.18 | 272148.86 | 535586.20 | 272.0489 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8200 | 0.8300 | 0.0100 | 15551.89 | 14462.80 | 151943.22 | 422916.88 | 151.8432 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1700 | 0.1800 | 0.0100 | 14462.80 | 15551.89 | 416108.88 | 138736.21 | 138.6362 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | No | 0.7100 | 0.7200 | 0.0100 | 17157.27 | 15097.05 | 139058.09 | 103608.73 | 103.5087 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | Yes | 0.2800 | 0.2900 | 0.0100 | 15097.05 | 17157.27 | 96700.16 | 128770.69 | 96.6002 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | No | 0.7100 | 0.7200 | 0.0100 | 4566.24 | 188.47 | 69331.38 | 66853.99 | 66.7540 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | 0.2800 | 0.2900 | 0.0100 | 188.47 | 4566.24 | 65068.41 | 65462.20 | 64.9684 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.7900 | 0.8000 | 0.0100 | 13334.96 | 2041.45 | 37277.26 | 43108.87 | 37.1773 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.2000 | 0.2100 | 0.0100 | 2041.45 | 13334.96 | 34240.93 | 37277.26 | 34.1409 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | 0.7800 | 0.7900 | 0.0100 | 11125.24 | 2829.19 | 52780.31 | 32026.13 | 31.9261 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | 0.2100 | 0.2200 | 0.0100 | 2829.19 | 11125.24 | 32026.13 | 52780.31 | 31.9261 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | 0.2200 | 0.2230 | 0.0030 | 813.79 | 2216.65 | 74308.06 | 26049.38 | 26.0194 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | 0.7770 | 0.7800 | 0.0030 | 2216.65 | 813.79 | 26049.38 | 74308.06 | 26.0194 | visible depth exists near both sides |
| Game Handicap: BLG (-2.5) vs Anyone's Legend (+2.5) | Yes | 0.6500 | 0.6700 | 0.0200 | 518.35 | 8.00 | 17002.23 | 18362.99 | 16.8022 | visible depth exists near both sides |
| Game Handicap: BLG (-2.5) vs Anyone's Legend (+2.5) | No | 0.3300 | 0.3500 | 0.0200 | 8.00 | 518.35 | 18362.99 | 11002.23 | 10.8022 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 9? | Yes | 0.0640 | 0.0750 | 0.0110 | 1054.37 | 15.00 | 175039.42 | 10217.94 | 10.1079 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 9? | No | 0.9250 | 0.9360 | 0.0110 | 15.00 | 1054.37 | 10217.94 | 175039.42 | 10.1079 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
