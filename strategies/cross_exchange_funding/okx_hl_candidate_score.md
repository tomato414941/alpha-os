# OKX-Hyperliquid Candidate Score

This ranks all assets from the 1m persistence summary after simple fee assumptions.

| asset | scenario | long | short | obs | net 8h after fee | net 24h after fee | capacity | survives 8h | survives 24h |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| JTO | very_low_fee | OkxSwap | HlPerp | 6 | -0.00146767 | 0.0004448 | 51289.301529 | False | True |
| BABY | very_low_fee | HlPerp | OkxSwap | 6 | -0.00212432 | 0.00038917 | 18227.15298962 | False | True |
| JTO | low_fee | OkxSwap | HlPerp | 6 | -0.00158767 | 0.0003248 | 51289.301529 | False | True |
| ZEC | very_low_fee | OkxSwap | HlPerp | 6 | -0.00031688 | 0.00031437 | 112828.19123333 | False | True |
| BABY | low_fee | HlPerp | OkxSwap | 6 | -0.00224432 | 0.00026917 | 18227.15298962 | False | True |
| ZEC | low_fee | OkxSwap | HlPerp | 6 | -0.00043688 | 0.00019437 | 112828.19123333 | False | True |
| BTC | very_low_fee | OkxSwap | HlPerp | 6 | -0.00007567 | 0.00015519 | 493421.184865 | False | True |
| JTO | one_bps_each | OkxSwap | HlPerp | 6 | -0.00178767 | 0.0001248 | 51289.301529 | False | True |
| BABY | one_bps_each | HlPerp | OkxSwap | 6 | -0.00244432 | 0.00006917 | 18227.15298962 | False | True |
| TIA | very_low_fee | OkxSwap | HlPerp | 6 | -0.00107793 | 0.00003533 | 7005.96491417 | False | True |
| BTC | low_fee | OkxSwap | HlPerp | 6 | -0.00019567 | 0.00003519 | 493421.184865 | False | True |
| ZEC | one_bps_each | OkxSwap | HlPerp | 6 | -0.00063688 | -0.00000563 | 112828.19123333 | False | False |
| SOL | very_low_fee | HlPerp | OkxSwap | 6 | -0.00025195 | -0.00002678 | 3123575.75932456 | False | False |
| WLD | very_low_fee | OkxSwap | HlPerp | 6 | -0.00081907 | -0.00007809 | 484440.98077761 | False | False |
| TIA | low_fee | OkxSwap | HlPerp | 6 | -0.00119793 | -0.00008467 | 7005.96491417 | False | False |
| HYPE | very_low_fee | OkxSwap | HlPerp | 6 | -0.0004001 | -0.00008988 | 3679499.29666667 | False | False |
| SOL | low_fee | HlPerp | OkxSwap | 6 | -0.00037195 | -0.00014678 | 3123575.75932456 | False | False |
| BTC | one_bps_each | OkxSwap | HlPerp | 6 | -0.00039567 | -0.00016481 | 493421.184865 | False | False |
| DOGE | very_low_fee | OkxSwap | HlPerp | 6 | -0.00040755 | -0.00019635 | 98216.33029323 | False | False |
| WLD | low_fee | OkxSwap | HlPerp | 6 | -0.00093907 | -0.00019809 | 484440.98077761 | False | False |

## Interpretation

The surviving set is dominated by the very-low-fee assumption. Under one bps per fill on both venues, the 1m persistence sample does not leave a robust top candidate. This makes fee tier and maker execution a hard requirement, not an optimization detail.
