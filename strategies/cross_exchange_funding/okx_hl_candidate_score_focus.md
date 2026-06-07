# OKX-Hyperliquid Candidate Score

This ranks assets from a persistence summary after simple fee assumptions.

| asset | scenario | long | short | obs | net 8h after fee | net 24h after fee | capacity | survives 8h | survives 24h |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| JTO | very_low_fee | OkxSwap | HlPerp | 12 | -0.00078625 | 0.00060005 | 54543.56750198 | False | True |
| JTO | low_fee | OkxSwap | HlPerp | 12 | -0.00090625 | 0.00048005 | 54543.56750198 | False | True |
| JTO | one_bps_each | OkxSwap | HlPerp | 12 | -0.00110625 | 0.00028005 | 54543.56750198 | False | True |
| ZEC | very_low_fee | OkxSwap | HlPerp | 12 | -0.00039009 | 0.000269 | 106210.05564167 | False | True |
| BTC | very_low_fee | OkxSwap | HlPerp | 12 | -0.00004638 | 0.0001801 | 422448.80855333 | False | True |
| ZEC | low_fee | OkxSwap | HlPerp | 12 | -0.00051009 | 0.000149 | 106210.05564167 | False | True |
| BTC | low_fee | OkxSwap | HlPerp | 12 | -0.00016638 | 0.0000601 | 422448.80855333 | False | True |
| ZEC | one_bps_each | OkxSwap | HlPerp | 12 | -0.00071009 | -0.000051 | 106210.05564167 | False | False |
| BTC | one_bps_each | OkxSwap | HlPerp | 12 | -0.00036638 | -0.0001399 | 422448.80855333 | False | False |
| BABY | very_low_fee | HlPerp | OkxSwap | 12 | -0.00388683 | -0.00126648 | 18182.63949194 | False | False |
| BABY | low_fee | HlPerp | OkxSwap | 12 | -0.00400683 | -0.00138648 | 18182.63949194 | False | False |
| BABY | one_bps_each | HlPerp | OkxSwap | 12 | -0.00420683 | -0.00158648 | 18182.63949194 | False | False |

## Interpretation

The surviving set is dominated by the very-low-fee assumption. Under one bps per fill on both venues, the 1m persistence sample does not leave a robust top candidate. This makes fee tier and maker execution a hard requirement, not an optimization detail.
