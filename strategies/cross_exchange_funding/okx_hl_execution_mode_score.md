# OKX-Hyperliquid Execution Mode Score

This compares maker/cross execution modes against event-window funding edge. Maker rebates and real queue position are not modeled.

| asset | scenario | mode | gross 8h | gross 24h | entry slippage bps | cost | net 8h | net 24h | both touch | OKX only | HL only | capacity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BABY | very_low_fee | both_maker | 0.00036862 | 0.00110587 | 0 | 0.00008 | 0.00028862 | 0.00102587 |  |  |  | 18182.63949194 |
| BABY | low_fee | both_maker | 0.00036862 | 0.00110587 | 0 | 0.0002 | 0.00016862 | 0.00090587 |  |  |  | 18182.63949194 |
| BABY | one_bps_each | both_maker | 0.00036862 | 0.00110587 | 0 | 0.0004 | -0.00003138 | 0.00070587 |  |  |  | 18182.63949194 |
| ZEC | very_low_fee | both_maker | 0.00024083 | 0.00072249 | 0 | 0.00008 | 0.00016083 | 0.00064249 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | very_low_fee | okx_cross_hl_maker | 0.00024083 | 0.00072249 | 0.12385895 | 0.00010477 | 0.00013606 | 0.00061772 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | low_fee | both_maker | 0.00024083 | 0.00072249 | 0 | 0.0002 | 0.00004083 | 0.00052249 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | low_fee | okx_cross_hl_maker | 0.00024083 | 0.00072249 | 0.12385895 | 0.00022477 | 0.00001606 | 0.00049772 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| BABY | very_low_fee | okx_maker_hl_cross | 0.00036862 | 0.00110587 | 2.90744629 | 0.00066149 | -0.00029287 | 0.00044438 |  |  |  | 18182.63949194 |
| ZEC | very_low_fee | okx_maker_hl_cross | 0.00024083 | 0.00072249 | 0.9912522 | 0.00027825 | -0.00003742 | 0.00044424 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | very_low_fee | both_cross | 0.00024083 | 0.00072249 | 1.11511115 | 0.00030302 | -0.00006219 | 0.00041947 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| BABY | low_fee | okx_maker_hl_cross | 0.00036862 | 0.00110587 | 2.90744629 | 0.00078149 | -0.00041287 | 0.00032438 |  |  |  | 18182.63949194 |
| ZEC | low_fee | okx_maker_hl_cross | 0.00024083 | 0.00072249 | 0.9912522 | 0.00039825 | -0.00015742 | 0.00032424 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | one_bps_each | both_maker | 0.00024083 | 0.00072249 | 0 | 0.0004 | -0.00015917 | 0.00032249 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | low_fee | both_cross | 0.00024083 | 0.00072249 | 1.11511115 | 0.00042302 | -0.00018219 | 0.00029947 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | one_bps_each | okx_cross_hl_maker | 0.00024083 | 0.00072249 | 0.12385895 | 0.00042477 | -0.00018394 | 0.00029772 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| BTC | very_low_fee | both_maker | 0.00010582 | 0.00031747 | 0 | 0.00008 | 0.00002582 | 0.00023747 | 0 | 0.6 | 0.2 | 422448.80855333 |
| BTC | very_low_fee | okx_cross_hl_maker | 0.00010582 | 0.00031747 | 0.00811186 | 0.00008162 | 0.0000242 | 0.00023585 | 0 | 0.6 | 0.2 | 422448.80855333 |
| BTC | very_low_fee | okx_maker_hl_cross | 0.00010582 | 0.00031747 | 0.08109511 | 0.00009622 | 0.0000096 | 0.00022125 | 0 | 0.6 | 0.2 | 422448.80855333 |
| BTC | very_low_fee | both_cross | 0.00010582 | 0.00031747 | 0.08920697 | 0.00009784 | 0.00000798 | 0.00021963 | 0 | 0.6 | 0.2 | 422448.80855333 |
| BABY | one_bps_each | okx_maker_hl_cross | 0.00036862 | 0.00110587 | 2.90744629 | 0.00098149 | -0.00061287 | 0.00012438 |  |  |  | 18182.63949194 |
| ZEC | one_bps_each | okx_maker_hl_cross | 0.00024083 | 0.00072249 | 0.9912522 | 0.00059825 | -0.00035742 | 0.00012424 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| BTC | low_fee | both_maker | 0.00010582 | 0.00031747 | 0 | 0.0002 | -0.00009418 | 0.00011747 | 0 | 0.6 | 0.2 | 422448.80855333 |
| BTC | low_fee | okx_cross_hl_maker | 0.00010582 | 0.00031747 | 0.00811186 | 0.00020162 | -0.0000958 | 0.00011585 | 0 | 0.6 | 0.2 | 422448.80855333 |
| BTC | low_fee | okx_maker_hl_cross | 0.00010582 | 0.00031747 | 0.08109511 | 0.00021622 | -0.0001104 | 0.00010125 | 0 | 0.6 | 0.2 | 422448.80855333 |

## Interpretation

If only both_maker survives, the candidate depends on maker availability. If a one-leg-cross mode survives, execution may be easier, but real fees and adverse selection still need account-level validation. Missing touch rates mean the maker-touch probe has not been run for that asset yet.
