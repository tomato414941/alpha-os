# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 8636.37 | 67231.95 | 633693.33 | 248136.74 | 248.0367 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 67231.95 | 8636.37 | 248136.74 | 633693.33 | 248.0367 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1400 | 0.1500 | 0.0100 | 75460.31 | 25801.89 | 613877.17 | 238727.46 | 238.6275 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8500 | 0.8600 | 0.0100 | 25801.89 | 75460.31 | 238727.46 | 635934.74 | 238.6275 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8200 | 0.8300 | 0.0100 | 9946.50 | 12012.12 | 152080.50 | 403358.76 | 151.9805 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1700 | 0.1800 | 0.0100 | 12012.12 | 9946.50 | 397537.76 | 138873.49 | 138.7735 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | No | 0.7100 | 0.7200 | 0.0100 | 4268.62 | 18590.82 | 126133.30 | 128361.76 | 126.0333 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | Yes | 0.2800 | 0.2900 | 0.0100 | 18590.82 | 4268.62 | 120458.19 | 113699.38 | 113.5994 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | No | 0.7200 | 0.7300 | 0.0100 | 31.00 | 7943.03 | 71943.41 | 72669.65 | 71.8434 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | 0.2700 | 0.2800 | 0.0100 | 7943.03 | 31.00 | 63939.88 | 71943.41 | 63.8399 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8200 | 0.8300 | 0.0100 | 13687.59 | 3563.23 | 74129.22 | 58415.90 | 58.3159 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1700 | 0.1800 | 0.0100 | 3563.23 | 13687.59 | 54820.93 | 64512.54 | 54.7209 | visible depth exists near both sides |
| Libema Open: Otto Virtanen vs Kamil Majchrzak | Yes | 0.5200 | 0.5300 | 0.0100 | 2838.45 | 2553.36 | 67502.48 | 46324.55 | 46.2246 | visible depth exists near both sides |
| Libema Open: Otto Virtanen vs Kamil Majchrzak | No | 0.4700 | 0.4800 | 0.0100 | 2553.36 | 2838.45 | 46324.55 | 67716.48 | 46.2246 | visible depth exists near both sides |
| Boston Red Sox vs. New York Yankees | Yes | 0.3500 | 0.4200 | 0.0700 | 230.38 | 104.48 | 56837.73 | 25876.95 | 25.1770 | spread is wide despite visible depth |
| Boston Red Sox vs. New York Yankees | No | 0.5800 | 0.6500 | 0.0700 | 104.48 | 230.38 | 25876.95 | 56837.73 | 25.1770 | spread is wide despite visible depth |
| Iran leadership change by June 30? | Yes | 0.0650 | 0.0710 | 0.0060 | 319.00 | 17.99 | 88737.74 | 24455.18 | 24.3952 | visible depth exists near both sides |
| Iran leadership change by June 30? | No | 0.9290 | 0.9350 | 0.0060 | 17.99 | 319.00 | 24455.18 | 88737.74 | 24.3952 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | No | 0.4500 | 0.4600 | 0.0100 | 2475.03 | 3571.89 | 18206.97 | 19959.57 | 18.1070 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | No | 0.8200 | 0.8300 | 0.0100 | 6807.63 | 40.00 | 23940.89 | 15638.28 | 15.5383 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
