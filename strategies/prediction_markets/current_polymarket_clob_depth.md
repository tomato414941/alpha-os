# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Will the U.S. invade Iran before 2027? | No | 0.8200 | 0.8300 | 0.0100 | 11755.50 | 802.73 | 148176.04 | 314574.62 | 148.0760 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1700 | 0.1800 | 0.0100 | 802.73 | 11755.50 | 307753.62 | 134969.03 | 134.8690 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.1000 | 0.1100 | 0.0100 | 6298.69 | 6734.00 | 907354.32 | 116205.19 | 116.1052 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.8900 | 0.9000 | 0.0100 | 6734.00 | 6298.69 | 116205.19 | 907354.32 | 116.1052 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.7800 | 0.8000 | 0.0200 | 13606.45 | 4025.41 | 59837.71 | 53587.71 | 53.3877 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.2000 | 0.2200 | 0.0200 | 4025.41 | 13606.45 | 41441.84 | 59837.71 | 41.2418 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | 0.2300 | 0.2400 | 0.0100 | 1124.52 | 11541.98 | 26492.88 | 27676.00 | 26.3929 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | 0.7600 | 0.7700 | 0.0100 | 11541.98 | 1124.52 | 27676.00 | 26392.88 | 26.2929 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 9? | Yes | 0.0970 | 0.0980 | 0.0010 | 173.00 | 36.90 | 24085.78 | 26003.36 | 24.0758 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 9? | No | 0.9020 | 0.9030 | 0.0010 | 36.90 | 173.00 | 26003.36 | 24085.78 | 24.0758 | visible depth exists near both sides |
| Bab el-Mandeb Strait effectively closed by June 30? | Yes | 0.1030 | 0.1130 | 0.0100 | 5.00 | 98.15 | 12649.47 | 10849.15 | 10.7492 | visible depth exists near both sides |
| Bab el-Mandeb Strait effectively closed by June 30? | No | 0.8870 | 0.8970 | 0.0100 | 98.15 | 5.00 | 10849.15 | 12649.47 | 10.7492 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | 0.2000 | 0.2200 | 0.0200 | 7923.01 | 786.12 | 10476.77 | 6170.28 | 5.9703 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | No | 0.7800 | 0.8000 | 0.0200 | 786.12 | 7923.01 | 6170.28 | 14558.77 | 5.9703 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | 0.2290 | 0.2300 | 0.0010 | 1143.00 | 100.00 | 3991.49 | 28178.09 | 3.9815 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | 0.7700 | 0.7710 | 0.0010 | 100.00 | 1143.00 | 28178.09 | 3991.49 | 3.9815 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | Yes | 0.2200 | 0.2400 | 0.0200 | 126.98 | 210.00 | 3986.86 | 21741.84 | 3.7869 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | No | 0.7600 | 0.7800 | 0.0200 | 210.00 | 126.98 | 21741.84 | 3986.86 | 3.7869 | visible depth exists near both sides |
| Israel closes its airspace by June 30? | Yes | 0.3700 | 0.3800 | 0.0100 | 37.68 | 15.00 | 1761.31 | 3780.23 | 1.6613 | visible depth exists near both sides |
| Israel closes its airspace by June 30? | No | 0.6200 | 0.6300 | 0.0100 | 15.00 | 37.68 | 3780.23 | 1761.31 | 1.6613 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
