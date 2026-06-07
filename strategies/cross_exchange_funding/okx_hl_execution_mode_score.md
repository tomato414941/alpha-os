# OKX-Hyperliquid Execution Mode Score

This compares maker/cross execution modes against event-window funding edge. Maker rebates and real queue position are not modeled.

| asset | scenario | mode | gross 8h | gross 24h | entry slippage bps | cost | net 8h | net 24h | both touch | OKX only | HL only | capacity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BABY | very_low_fee | both_maker | 0.00036862 | 0.00110587 | 0 | 0.00008 | 0.00028862 | 0.00102587 | 0 | 0.2 | 0.2 | 18182.63949194 |
| BABY | low_fee | both_maker | 0.00036862 | 0.00110587 | 0 | 0.0002 | 0.00016862 | 0.00090587 | 0 | 0.2 | 0.2 | 18182.63949194 |
| BABY | one_bps_each | both_maker | 0.00036862 | 0.00110587 | 0 | 0.0004 | -0.00003138 | 0.00070587 | 0 | 0.2 | 0.2 | 18182.63949194 |
| ZEC | very_low_fee | both_maker | 0.00024083 | 0.00072249 | 0 | 0.00008 | 0.00016083 | 0.00064249 | 0 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | very_low_fee | okx_cross_hl_maker | 0.00024083 | 0.00072249 | 0.12315726 | 0.00010463 | 0.0001362 | 0.00061786 | 0 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | very_low_fee | okx_maker_hl_cross | 0.00024083 | 0.00072249 | 0.49341294 | 0.00017868 | 0.00006215 | 0.00054381 | 0 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | low_fee | both_maker | 0.00024083 | 0.00072249 | 0 | 0.0002 | 0.00004083 | 0.00052249 | 0 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | very_low_fee | both_cross | 0.00024083 | 0.00072249 | 0.6165702 | 0.00020331 | 0.00003752 | 0.00051918 | 0 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | low_fee | okx_cross_hl_maker | 0.00024083 | 0.00072249 | 0.12315726 | 0.00022463 | 0.0000162 | 0.00049786 | 0 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | low_fee | okx_maker_hl_cross | 0.00024083 | 0.00072249 | 0.49341294 | 0.00029868 | -0.00005785 | 0.00042381 | 0 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | low_fee | both_cross | 0.00024083 | 0.00072249 | 0.6165702 | 0.00032331 | -0.00008248 | 0.00039918 | 0 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | one_bps_each | both_maker | 0.00024083 | 0.00072249 | 0 | 0.0004 | -0.00015917 | 0.00032249 | 0 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | one_bps_each | okx_cross_hl_maker | 0.00024083 | 0.00072249 | 0.12315726 | 0.00042463 | -0.0001838 | 0.00029786 | 0 | 0.6 | 0.2 | 106210.05564167 |
| BTC | very_low_fee | both_maker | 0.00010582 | 0.00031747 | 0 | 0.00008 | 0.00002582 | 0.00023747 | 0.2 | 0.4 | 0.2 | 422448.80855333 |
| BTC | very_low_fee | okx_cross_hl_maker | 0.00010582 | 0.00031747 | 0.00810394 | 0.00008162 | 0.0000242 | 0.00023585 | 0.2 | 0.4 | 0.2 | 422448.80855333 |
| ZEC | one_bps_each | okx_maker_hl_cross | 0.00024083 | 0.00072249 | 0.49341294 | 0.00049868 | -0.00025785 | 0.00022381 | 0 | 0.6 | 0.2 | 106210.05564167 |
| BTC | very_low_fee | okx_maker_hl_cross | 0.00010582 | 0.00031747 | 0.08101889 | 0.0000962 | 0.00000962 | 0.00022127 | 0.2 | 0.4 | 0.2 | 422448.80855333 |
| BTC | very_low_fee | both_cross | 0.00010582 | 0.00031747 | 0.08912284 | 0.00009782 | 0.000008 | 0.00021965 | 0.2 | 0.4 | 0.2 | 422448.80855333 |
| ZEC | one_bps_each | both_cross | 0.00024083 | 0.00072249 | 0.6165702 | 0.00052331 | -0.00028248 | 0.00019918 | 0 | 0.6 | 0.2 | 106210.05564167 |
| BTC | low_fee | both_maker | 0.00010582 | 0.00031747 | 0 | 0.0002 | -0.00009418 | 0.00011747 | 0.2 | 0.4 | 0.2 | 422448.80855333 |
| BTC | low_fee | okx_cross_hl_maker | 0.00010582 | 0.00031747 | 0.00810394 | 0.00020162 | -0.0000958 | 0.00011585 | 0.2 | 0.4 | 0.2 | 422448.80855333 |
| BTC | low_fee | okx_maker_hl_cross | 0.00010582 | 0.00031747 | 0.08101889 | 0.0002162 | -0.00011038 | 0.00010127 | 0.2 | 0.4 | 0.2 | 422448.80855333 |
| BTC | low_fee | both_cross | 0.00010582 | 0.00031747 | 0.08912284 | 0.00021782 | -0.000112 | 0.00009965 | 0.2 | 0.4 | 0.2 | 422448.80855333 |
| JTO | very_low_fee | both_maker | 0.00005877 | 0.00017631 | 0 | 0.00008 | -0.00002123 | 0.00009631 | 0 | 0.8 | 0.2 | 54543.56750198 |

## Interpretation

If only both_maker survives, the candidate depends on maker availability. If a one-leg-cross mode survives, execution may be easier, but real fees and adverse selection still need account-level validation. Touch rates are short-sample public-book proxies, not real fill probabilities.
