# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.1000 | 0.1100 | 0.0100 | 24478.53 | 29282.43 | 502478.73 | 214008.09 | 213.9081 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.8900 | 0.9000 | 0.0100 | 29282.43 | 24478.53 | 214008.09 | 502478.73 | 213.9081 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8200 | 0.8300 | 0.0100 | 12951.42 | 11604.88 | 153419.53 | 288576.68 | 153.3195 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1700 | 0.1800 | 0.0100 | 11604.88 | 12951.42 | 282243.17 | 140012.52 | 139.9125 | visible depth exists near both sides |
| Stuttgart Open: Tommy Paul vs Giovanni Mpetshi Perricard | Yes | 0.5000 | 0.5100 | 0.0100 | 89706.72 | 184085.43 | 109747.53 | 204085.43 | 109.6475 | visible depth exists near both sides |
| Stuttgart Open: Tommy Paul vs Giovanni Mpetshi Perricard | No | 0.4900 | 0.5000 | 0.0100 | 184085.43 | 89706.72 | 204085.43 | 109747.53 | 109.6475 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8200 | 0.8300 | 0.0100 | 17120.45 | 9766.20 | 101740.40 | 881092.27 | 101.6404 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1700 | 0.1800 | 0.0100 | 9766.20 | 17120.45 | 506586.60 | 81670.04 | 81.5700 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1900 | 0.2000 | 0.0100 | 4008.14 | 9.94 | 44986.93 | 47207.18 | 44.8869 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8000 | 0.8100 | 0.0100 | 9.94 | 4008.14 | 47207.18 | 44986.93 | 44.8869 | visible depth exists near both sides |
| Libema Open: Mia Pohankova vs Clara Tauson | Yes | 0.5700 | 0.5800 | 0.0100 | 5756.07 | 18828.02 | 36767.77 | 126743.32 | 36.6678 | visible depth exists near both sides |
| Libema Open: Mia Pohankova vs Clara Tauson | No | 0.4200 | 0.4300 | 0.0100 | 18828.02 | 5756.07 | 126743.32 | 36767.77 | 36.6678 | visible depth exists near both sides |
| Libema Open: Marin Cilic vs Denis Shapovalov | Yes | 0.3800 | 0.3900 | 0.0100 | 25.00 | 25.00 | 46944.98 | 16056.70 | 15.9567 | visible depth exists near both sides |
| Libema Open: Marin Cilic vs Denis Shapovalov | No | 0.6100 | 0.6200 | 0.0100 | 25.00 | 25.00 | 16056.70 | 46944.98 | 15.9567 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | 0.6600 | 0.6700 | 0.0100 | 1153.88 | 217.82 | 25362.41 | 10100.41 | 10.0004 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | 0.3300 | 0.3400 | 0.0100 | 217.82 | 1153.88 | 10100.41 | 25362.41 | 10.0004 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 9? | Yes | 0.0750 | 0.0790 | 0.0040 | 375.23 | 24.39 | 35585.82 | 9913.75 | 9.8738 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 9? | No | 0.9210 | 0.9250 | 0.0040 | 24.39 | 375.23 | 9913.75 | 35585.82 | 9.8738 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | Yes | 0.1400 | 0.1600 | 0.0200 | 3504.34 | 1976.32 | 42212.63 | 9837.13 | 9.6371 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | No | 0.8400 | 0.8600 | 0.0200 | 1976.32 | 3504.34 | 9837.13 | 42412.67 | 9.6371 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
