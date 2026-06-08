# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | Yes | 0.1900 | 0.2000 | 0.0100 | 77383.37 | 120108.24 | 332249.47 | 437269.58 | 332.1495 | visible depth exists near both sides |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | No | 0.8000 | 0.8100 | 0.0100 | 120108.24 | 77383.37 | 437269.58 | 332249.47 | 332.1495 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 13308.20 | 53105.09 | 344083.44 | 241214.80 | 241.1148 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 53105.09 | 13308.20 | 241214.80 | 344083.44 | 241.1148 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8500 | 0.8600 | 0.0100 | 21991.14 | 6085.48 | 247543.09 | 206653.68 | 206.5537 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1400 | 0.1500 | 0.0100 | 6085.48 | 21991.14 | 186546.11 | 247543.09 | 186.4461 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8200 | 0.8300 | 0.0100 | 669.88 | 582.63 | 140296.13 | 272427.71 | 140.1961 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1700 | 0.1800 | 0.0100 | 582.63 | 669.88 | 266606.71 | 127089.12 | 126.9891 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | Yes | 0.2900 | 0.3000 | 0.0100 | 9900.83 | 1786.62 | 111966.13 | 116886.09 | 111.8661 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | No | 0.7000 | 0.7100 | 0.0100 | 1786.62 | 9900.83 | 116886.09 | 111966.13 | 111.8661 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | No | 0.7100 | 0.7200 | 0.0100 | 5351.88 | 4572.63 | 49623.36 | 67520.94 | 49.5234 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | 0.2800 | 0.2900 | 0.0100 | 4572.63 | 5351.88 | 66374.36 | 47619.18 | 47.5192 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1900 | 0.2000 | 0.0100 | 2844.18 | 9559.89 | 40392.22 | 70550.35 | 40.2922 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8000 | 0.8100 | 0.0100 | 9559.89 | 2844.18 | 70550.35 | 40392.22 | 40.2922 | visible depth exists near both sides |
| Boston Red Sox vs. New York Yankees | Yes | 0.3500 | 0.4300 | 0.0800 | 14632.64 | 9.99 | 54648.83 | 25849.85 | 25.0498 | spread is wide despite visible depth |
| Boston Red Sox vs. New York Yankees | No | 0.5700 | 0.6500 | 0.0800 | 9.99 | 14632.64 | 25849.85 | 54648.83 | 25.0498 | spread is wide despite visible depth |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | 0.6700 | 0.6800 | 0.0100 | 4911.45 | 6008.00 | 17702.53 | 27482.11 | 17.6025 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | 0.3200 | 0.3300 | 0.0100 | 6008.00 | 4911.45 | 27482.11 | 17702.53 | 17.6025 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | 0.3240 | 0.3380 | 0.0140 | 500.00 | 89.59 | 16010.60 | 16864.49 | 15.8706 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | 0.6620 | 0.6760 | 0.0140 | 89.59 | 500.00 | 15738.49 | 16010.60 | 15.5985 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
