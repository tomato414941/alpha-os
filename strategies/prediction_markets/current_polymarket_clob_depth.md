# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 9102.90 | 109786.18 | 634468.52 | 291239.63 | 291.1396 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 109786.18 | 9102.90 | 291239.63 | 634468.52 | 291.1396 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1400 | 0.1500 | 0.0100 | 76234.31 | 25783.38 | 950092.13 | 240895.57 | 240.7956 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8500 | 0.8600 | 0.0100 | 25783.38 | 76234.31 | 240895.57 | 972149.70 | 240.7956 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8200 | 0.8300 | 0.0100 | 10381.97 | 11982.57 | 152535.48 | 407179.40 | 152.4355 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1700 | 0.1800 | 0.0100 | 11982.57 | 10381.97 | 400358.40 | 139328.47 | 139.2285 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | No | 0.7100 | 0.7200 | 0.0100 | 4267.89 | 18360.68 | 121724.58 | 122345.16 | 121.6246 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | Yes | 0.2800 | 0.2900 | 0.0100 | 18360.68 | 4267.89 | 114441.59 | 112423.99 | 112.3240 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | No | 0.7200 | 0.7300 | 0.0100 | 29.00 | 7977.85 | 71389.26 | 71959.39 | 71.2893 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | 0.2700 | 0.2800 | 0.0100 | 7977.85 | 29.00 | 63235.25 | 71389.26 | 63.1352 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8200 | 0.8300 | 0.0100 | 15040.88 | 3159.82 | 74256.53 | 58870.63 | 58.7706 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1700 | 0.1800 | 0.0100 | 3159.82 | 15040.88 | 56050.72 | 64639.85 | 55.9507 | visible depth exists near both sides |
| Libema Open: Otto Virtanen vs Kamil Majchrzak | Yes | 0.5200 | 0.5300 | 0.0100 | 2838.45 | 1759.66 | 65244.82 | 52362.73 | 52.2627 | visible depth exists near both sides |
| Libema Open: Otto Virtanen vs Kamil Majchrzak | No | 0.4700 | 0.4800 | 0.0100 | 1759.66 | 2838.45 | 52362.73 | 65458.82 | 52.2627 | visible depth exists near both sides |
| Boston Red Sox vs. New York Yankees | Yes | 0.3500 | 0.4200 | 0.0700 | 228.00 | 47.20 | 56835.35 | 25819.67 | 25.1197 | spread is wide despite visible depth |
| Boston Red Sox vs. New York Yankees | No | 0.5800 | 0.6500 | 0.0700 | 47.20 | 228.00 | 25819.67 | 56835.35 | 25.1197 | spread is wide despite visible depth |
| Iran leadership change by June 30? | Yes | 0.0670 | 0.0700 | 0.0030 | 28.00 | 48.54 | 49047.22 | 24251.95 | 24.2219 | visible depth exists near both sides |
| Iran leadership change by June 30? | No | 0.9300 | 0.9330 | 0.0030 | 48.54 | 28.00 | 24251.95 | 49047.22 | 24.2219 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | Yes | 0.5300 | 0.5500 | 0.0200 | 7493.78 | 2419.81 | 27289.22 | 18616.79 | 18.4168 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | No | 0.4500 | 0.4700 | 0.0200 | 2419.81 | 7493.78 | 18616.79 | 33289.22 | 18.4168 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
