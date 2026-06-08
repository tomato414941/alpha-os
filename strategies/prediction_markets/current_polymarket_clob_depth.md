# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.1000 | 0.1100 | 0.0100 | 25331.11 | 31245.95 | 620142.69 | 222484.91 | 222.3849 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.8900 | 0.9000 | 0.0100 | 31245.95 | 25331.11 | 222484.91 | 620142.69 | 222.3849 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8200 | 0.8300 | 0.0100 | 13953.25 | 11708.18 | 126323.33 | 302830.15 | 126.2233 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1700 | 0.1800 | 0.0100 | 11708.18 | 13953.25 | 295576.64 | 112916.32 | 112.8163 | visible depth exists near both sides |
| Stuttgart Open: Tommy Paul vs Giovanni Mpetshi Perricard | Yes | 0.5000 | 0.5100 | 0.0100 | 89722.98 | 184109.93 | 109763.79 | 204109.93 | 109.6638 | visible depth exists near both sides |
| Stuttgart Open: Tommy Paul vs Giovanni Mpetshi Perricard | No | 0.4900 | 0.5000 | 0.0100 | 184109.93 | 89722.98 | 204109.93 | 109763.79 | 109.6638 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8200 | 0.8300 | 0.0100 | 17136.45 | 9949.90 | 99737.49 | 898866.05 | 99.6375 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1700 | 0.1800 | 0.0100 | 9949.90 | 17136.45 | 524360.38 | 80467.13 | 80.3671 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1900 | 0.2000 | 0.0100 | 2891.07 | 11.85 | 54486.91 | 57767.40 | 54.3869 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8000 | 0.8100 | 0.0100 | 11.85 | 2891.07 | 57767.40 | 54486.91 | 54.3869 | visible depth exists near both sides |
| Libema Open: Marin Cilic vs Denis Shapovalov | No | 0.6200 | 0.6300 | 0.0100 | 50.00 | 1524.72 | 33338.32 | 34567.14 | 33.2383 | visible depth exists near both sides |
| Libema Open: Marin Cilic vs Denis Shapovalov | Yes | 0.3700 | 0.3800 | 0.0100 | 1524.72 | 50.00 | 34600.54 | 33138.32 | 33.0383 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | 0.6700 | 0.6900 | 0.0200 | 7825.70 | 6981.59 | 57934.15 | 29750.67 | 29.5507 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | 0.3100 | 0.3300 | 0.0200 | 6981.59 | 7825.70 | 29750.67 | 57934.15 | 29.5507 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | 0.6800 | 0.6810 | 0.0010 | 539.19 | 1000.00 | 64425.64 | 13992.13 | 13.9821 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | 0.3190 | 0.3200 | 0.0010 | 1000.00 | 539.19 | 13155.41 | 64425.64 | 13.1454 | visible depth exists near both sides |
| Bab el-Mandeb Strait effectively closed by June 30? | Yes | 0.0960 | 0.1120 | 0.0160 | 12.33 | 26.20 | 25294.08 | 11626.83 | 11.4668 | visible depth exists near both sides |
| Bab el-Mandeb Strait effectively closed by June 30? | No | 0.8880 | 0.9040 | 0.0160 | 26.20 | 12.33 | 11626.83 | 25294.08 | 11.4668 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 9? | Yes | 0.0790 | 0.0880 | 0.0090 | 26.59 | 10.00 | 38517.56 | 7042.71 | 6.9527 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 9? | No | 0.9120 | 0.9210 | 0.0090 | 10.00 | 26.59 | 7042.71 | 38517.56 | 6.9527 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
