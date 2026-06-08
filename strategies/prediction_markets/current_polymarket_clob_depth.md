# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | Yes | 0.1900 | 0.2000 | 0.0100 | 114699.84 | 133592.00 | 324243.73 | 341758.44 | 324.1437 | visible depth exists near both sides |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | No | 0.8000 | 0.8100 | 0.0100 | 133592.00 | 114699.84 | 341758.44 | 324243.73 | 324.1437 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 13783.93 | 65850.76 | 588766.63 | 242290.05 | 242.1900 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 65850.76 | 13783.93 | 242290.05 | 588766.63 | 242.1900 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1400 | 0.1500 | 0.0100 | 128980.31 | 22144.76 | 622093.15 | 235877.22 | 235.7772 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8500 | 0.8600 | 0.0100 | 22144.76 | 128980.31 | 235877.22 | 643039.72 | 235.7772 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8200 | 0.8300 | 0.0100 | 21707.93 | 9742.19 | 174690.99 | 403288.23 | 174.5910 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1700 | 0.1800 | 0.0100 | 9742.19 | 21707.93 | 398254.07 | 161483.98 | 161.3840 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | No | 0.7100 | 0.7200 | 0.0100 | 753.54 | 19425.71 | 127547.06 | 123854.69 | 123.7547 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | Yes | 0.2800 | 0.2900 | 0.0100 | 19425.71 | 753.54 | 115951.12 | 118018.47 | 115.8511 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | No | 0.7100 | 0.7200 | 0.0100 | 10240.79 | 160.19 | 68872.71 | 78482.26 | 68.7727 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | 0.2800 | 0.2900 | 0.0100 | 160.19 | 10240.79 | 77335.68 | 64646.53 | 64.5465 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8200 | 0.8300 | 0.0100 | 223.79 | 4449.16 | 61898.36 | 53484.85 | 53.3849 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1700 | 0.1800 | 0.0100 | 4449.16 | 223.79 | 50709.88 | 51581.68 | 50.6099 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | 0.1600 | 0.1700 | 0.0100 | 6423.07 | 9107.68 | 38328.01 | 37723.90 | 37.6239 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | No | 0.8300 | 0.8400 | 0.0100 | 9107.68 | 6423.07 | 37723.90 | 38328.01 | 37.6239 | visible depth exists near both sides |
| Iran leadership change by June 30? | Yes | 0.0720 | 0.0790 | 0.0070 | 51.00 | 983.49 | 81857.32 | 26919.49 | 26.8495 | visible depth exists near both sides |
| Iran leadership change by June 30? | No | 0.9210 | 0.9280 | 0.0070 | 983.49 | 51.00 | 26919.49 | 81857.32 | 26.8495 | visible depth exists near both sides |
| Boston Red Sox vs. New York Yankees | Yes | 0.3500 | 0.4200 | 0.0700 | 207.15 | 56.71 | 56824.70 | 26096.21 | 25.3962 | spread is wide despite visible depth |
| Boston Red Sox vs. New York Yankees | No | 0.5800 | 0.6500 | 0.0700 | 56.71 | 207.15 | 26096.21 | 56824.70 | 25.3962 | spread is wide despite visible depth |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
