# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| US-Iran nuclear deal by June 30? | Yes | 0.2200 | 0.2300 | 0.0100 | 208.00 | 18274.13 | 41209.31 | 51270.60 | 41.1093 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.7700 | 0.7800 | 0.0100 | 18274.13 | 208.00 | 51270.60 | 41209.31 | 41.1093 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | 0.7700 | 0.7800 | 0.0100 | 5921.05 | 3321.93 | 34999.10 | 23702.45 | 23.6025 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | 0.2200 | 0.2300 | 0.0100 | 3321.93 | 5921.05 | 23702.45 | 34999.10 | 23.6025 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 9? | Yes | 0.1020 | 0.1140 | 0.0120 | 394.81 | 32.64 | 22380.42 | 20693.16 | 20.5732 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 9? | No | 0.8860 | 0.8980 | 0.0120 | 32.64 | 394.81 | 20693.16 | 22380.42 | 20.5732 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | 0.2320 | 0.2350 | 0.0030 | 318.77 | 253.71 | 5363.46 | 22500.00 | 5.3335 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | 0.7650 | 0.7680 | 0.0030 | 253.71 | 318.77 | 23500.00 | 5363.46 | 5.3335 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | 0.2100 | 0.2300 | 0.0200 | 1138.41 | 13.65 | 10749.54 | 5154.74 | 4.9547 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | No | 0.7700 | 0.7900 | 0.0200 | 13.65 | 1138.41 | 5154.74 | 10749.54 | 4.9547 | visible depth exists near both sides |
| Bab el-Mandeb Strait effectively closed by June 30? | No | 0.8970 | 0.9010 | 0.0040 | 5.00 | 5.00 | 3438.01 | 18877.30 | 3.3980 | visible depth exists near both sides |
| Bab el-Mandeb Strait effectively closed by June 30? | Yes | 0.0990 | 0.1030 | 0.0040 | 5.00 | 5.00 | 18818.30 | 2766.01 | 2.7260 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | No | 0.7000 | 0.7200 | 0.0200 | 1895.66 | 44.00 | 4624.64 | 2534.26 | 2.3343 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | Yes | 0.2800 | 0.3000 | 0.0200 | 44.00 | 1895.66 | 1690.52 | 4624.64 | 1.4905 | visible depth exists near both sides |
| Israel closes its airspace by June 30? | Yes | 0.3900 | 0.4000 | 0.0100 | 117.83 | 9.09 | 946.52 | 5005.41 | 0.8465 | visible near-top depth is thin |
| Israel closes its airspace by June 30? | No | 0.6000 | 0.6100 | 0.0100 | 9.09 | 117.83 | 5005.41 | 946.52 | 0.8465 | visible near-top depth is thin |
| Will the price of Bitcoin be above $58,000 on June 8? | Yes | 0.9960 | 0.9970 | 0.0010 | 3145.74 | 1281.00 | 17772.39 | 24677.95 | 17.7624 | visible depth exists near both sides |
| Will the price of Bitcoin be above $58,000 on June 8? | No | 0.0030 | 0.0040 | 0.0010 | 1281.00 | 3145.74 | 24677.95 | 17772.39 | 17.7624 | visible depth exists near both sides |
| Iran closes its airspace by June 8? | Yes | 0.9990 | 0.0000 | 0.0000 | 610805.52 | 0.00 | 3212117.53 | 0.00 | 0.0000 | visible near-top depth is thin |
| Iran closes its airspace by June 8? | No | 0.0000 | 0.0010 | 0.0000 | 0.00 | 610805.52 | 0.00 | 3212117.53 | 0.0000 | visible near-top depth is thin |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
