# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 2948.07 | 66116.54 | 575830.64 | 237642.61 | 237.5426 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 66116.54 | 2948.07 | 237642.61 | 575830.64 | 237.5426 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1400 | 0.1500 | 0.0100 | 84127.31 | 13823.07 | 567027.91 | 230599.36 | 230.4994 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8500 | 0.8600 | 0.0100 | 13823.07 | 84127.31 | 230599.36 | 589085.48 | 230.4994 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8200 | 0.8300 | 0.0100 | 8607.41 | 17008.43 | 154772.97 | 415920.30 | 154.6730 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1700 | 0.1800 | 0.0100 | 17008.43 | 8607.41 | 410066.14 | 141565.96 | 141.4660 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | No | 0.7100 | 0.7200 | 0.0100 | 1627.60 | 28863.80 | 128505.15 | 149899.38 | 128.4051 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | Yes | 0.2800 | 0.2900 | 0.0100 | 28863.80 | 1627.60 | 141995.81 | 118976.56 | 118.8766 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | No | 0.7200 | 0.7300 | 0.0100 | 13.00 | 7739.41 | 65747.64 | 57624.77 | 57.5248 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8200 | 0.8300 | 0.0100 | 14882.04 | 3537.35 | 75329.33 | 56081.85 | 55.9818 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1700 | 0.1800 | 0.0100 | 3537.35 | 14882.04 | 52486.88 | 65712.65 | 52.3869 | visible depth exists near both sides |
| Libema Open: Otto Virtanen vs Kamil Majchrzak | Yes | 0.5200 | 0.5300 | 0.0100 | 1945.70 | 4639.06 | 60507.68 | 49611.33 | 49.5113 | visible depth exists near both sides |
| Libema Open: Otto Virtanen vs Kamil Majchrzak | No | 0.4700 | 0.4800 | 0.0100 | 4639.06 | 1945.70 | 49611.33 | 60794.68 | 49.5113 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | 0.2700 | 0.2800 | 0.0100 | 7739.41 | 13.00 | 48900.63 | 65747.64 | 48.8006 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | Yes | 0.5700 | 0.5800 | 0.0100 | 60023.27 | 1762.33 | 69623.68 | 26623.27 | 26.5233 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | No | 0.4200 | 0.4300 | 0.0100 | 1762.33 | 60023.27 | 26623.27 | 69623.68 | 26.5233 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | 0.1600 | 0.1800 | 0.0200 | 3374.91 | 6412.63 | 32294.58 | 26145.89 | 25.9459 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | No | 0.8200 | 0.8400 | 0.0200 | 6412.63 | 3374.91 | 26145.89 | 32294.58 | 25.9459 | visible depth exists near both sides |
| Iran leadership change by June 30? | Yes | 0.0670 | 0.0750 | 0.0080 | 115.00 | 647.46 | 49545.28 | 24868.41 | 24.7884 | visible depth exists near both sides |
| Iran leadership change by June 30? | No | 0.9250 | 0.9330 | 0.0080 | 647.46 | 115.00 | 24868.41 | 49545.28 | 24.7884 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
