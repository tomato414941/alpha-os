# OKX-Hyperliquid Event Window Triage

This turns event-window scores into research actions. It should override the smooth execution-cost triage when the two disagree.

| asset | event action | previous action | long | short | capacity | very-low 8h | very-low 24h | low-fee 24h | one-bps 24h | max slippage bps | reason |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BTC | paper_8h_candidate | paper_8h_candidate | OkxSwap | HlPerp | 422448.80855333 | 0.00000805 | 0.0002197 | 0.0000997 | -0.0001003 | 0.08887218 | 8h event-window survives only under very low fee assumptions |
| ZEC | very_low_fee_24h_watch | fee_dependent_24h_monitor | OkxSwap | HlPerp | 106210.05564167 | -0.00046803 | 0.00001363 | -0.00010637 | -0.00030637 | 3.14430447 | 24h event-window only survives the very-low-fee assumption |
| JTO | drop_for_now | active_24h_monitor | OkxSwap | HlPerp | 54543.56750198 | -0.00152693 | -0.00140939 | -0.00152939 | -0.00172939 | 7.52849755 | no current event-window scenario survives |
| BABY | drop_for_now | thin_or_unstable_watch | HlPerp | OkxSwap | 18182.63949194 | -0.00230137 | -0.00156412 | -0.00168412 | -0.00188412 | 12.94997144 | no current event-window scenario survives |
