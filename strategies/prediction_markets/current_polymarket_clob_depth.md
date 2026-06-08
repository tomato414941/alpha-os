# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1400 | 0.1500 | 0.0100 | 86107.00 | 25641.72 | 579959.86 | 243549.68 | 243.4497 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8500 | 0.8600 | 0.0100 | 25641.72 | 86107.00 | 243549.68 | 602017.43 | 243.4497 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 7252.57 | 66086.97 | 581443.23 | 238543.89 | 238.4439 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 66086.97 | 7252.57 | 238543.89 | 581443.23 | 238.4439 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8200 | 0.8300 | 0.0100 | 18683.51 | 17019.86 | 198108.52 | 416218.22 | 198.0085 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1700 | 0.1800 | 0.0100 | 17019.86 | 18683.51 | 410351.06 | 184901.51 | 184.8015 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | No | 0.7100 | 0.7200 | 0.0100 | 1435.56 | 29627.18 | 118692.91 | 130167.40 | 118.5929 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | Yes | 0.2800 | 0.2900 | 0.0100 | 29627.18 | 1435.56 | 122263.83 | 109164.32 | 109.0643 | visible depth exists near both sides |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | Yes | 0.1800 | 0.1900 | 0.0100 | 38770.00 | 25014.39 | 69933.74 | 126811.15 | 69.8337 | visible depth exists near both sides |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | No | 0.8100 | 0.8200 | 0.0100 | 25014.39 | 38770.00 | 126811.15 | 69933.74 | 69.8337 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | No | 0.7200 | 0.7300 | 0.0100 | 5.00 | 8464.47 | 67982.48 | 68953.82 | 67.8825 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | 0.2700 | 0.2800 | 0.0100 | 8464.47 | 5.00 | 60229.68 | 67982.48 | 60.1297 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8200 | 0.8300 | 0.0100 | 14689.63 | 3104.25 | 63819.77 | 46952.55 | 46.8526 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1700 | 0.1800 | 0.0100 | 3104.25 | 14689.63 | 43357.58 | 54103.09 | 43.2576 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | 0.1600 | 0.1800 | 0.0200 | 3360.43 | 8128.66 | 29277.29 | 28261.92 | 28.0619 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | No | 0.8200 | 0.8400 | 0.0200 | 8128.66 | 3360.43 | 28261.92 | 29277.29 | 28.0619 | visible depth exists near both sides |
| Libema Open: Otto Virtanen vs Kamil Majchrzak | Yes | 0.5100 | 0.5200 | 0.0100 | 8761.82 | 110.00 | 67705.84 | 23413.46 | 23.3135 | visible depth exists near both sides |
| Libema Open: Otto Virtanen vs Kamil Majchrzak | No | 0.4800 | 0.4900 | 0.0100 | 110.00 | 8761.82 | 23413.46 | 67705.84 | 23.3135 | visible depth exists near both sides |
| Iran leadership change by June 30? | Yes | 0.0680 | 0.0790 | 0.0110 | 342.58 | 120.00 | 48886.34 | 20337.31 | 20.2273 | visible depth exists near both sides |
| Iran leadership change by June 30? | No | 0.9210 | 0.9320 | 0.0110 | 120.00 | 342.58 | 20337.31 | 48892.03 | 20.2273 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
