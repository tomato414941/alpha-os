# Current Hyperliquid OI Shift Candidates

This reads accumulated Hyperliquid dislocation monitor samples and looks for short-window open-interest notional shifts. It is a crowding/unwind candidate screen, not a trade instruction.

| asset | status | side | score | obs | OI change | ret24 | funding ann | OI/vol | impact | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PURR | paper_oi_funding_crowding_watch | long_perp | 38.9924 | 17 | 0.0290 | 0.1070 | 0.1095 | 9.1428 | 0.012507 | OI notional is rising into an up move; test crowded continuation versus late-long squeeze risk |
| IMX | paper_oi_funding_crowding_watch | long_perp | 33.9798 | 15 | 0.0292 | 0.0769 | 0.1095 | 2.3200 | 0.002793 | OI notional is rising into an up move; test crowded continuation versus late-long squeeze risk |
| ZEC | paper_oi_unwind_watch | context_only | 30.3644 | 17 | -0.0168 | 0.1826 | -0.3458 | 0.4256 | 0.000376 | OI notional is falling into an up move; test short-cover exhaustion versus cleaner continuation |
| NEAR | paper_oi_funding_crowding_watch | long_perp | 28.3350 | 17 | 0.0211 | 0.1019 | 0.1095 | 1.1965 | 0.000439 | OI notional is rising into an up move; test crowded continuation versus late-long squeeze risk |
| DASH | paper_oi_funding_crowding_watch | long_perp | 27.8703 | 17 | 0.0193 | 0.1301 | 0.1095 | 0.3489 | 0.001232 | OI notional is rising into an up move; test crowded continuation versus late-long squeeze risk |
| PENGU | paper_oi_unwind_watch | context_only | 25.0121 | 17 | -0.0182 | 0.0931 | -0.2518 | 1.1172 | 0.000727 | OI notional is falling into an up move; test short-cover exhaustion versus cleaner continuation |
| HYPE | paper_oi_funding_crowding_watch | long_perp | 22.4961 | 8 | 0.0157 | 0.0784 | 0.1095 | 2.0060 | 0.000040 | OI notional is rising into an up move; test crowded continuation versus late-long squeeze risk |
| SPX | paper_oi_unwind_watch | context_only | 20.8197 | 13 | -0.0148 | 0.0710 | -0.2933 | 2.0737 | 0.001924 | OI notional is falling into an up move; test short-cover exhaustion versus cleaner continuation |
| ATOM | paper_oi_funding_crowding_watch | long_perp | 20.4741 | 17 | 0.0115 | 0.0515 | 0.0652 | 6.4130 | 0.001520 | OI notional is rising into an up move; test crowded continuation versus late-long squeeze risk |
| FARTCOIN | paper_oi_unwind_watch | context_only | 18.4274 | 17 | -0.0103 | 0.0804 | 0.1095 | 2.8683 | 0.000867 | OI notional is falling into an up move; test short-cover exhaustion versus cleaner continuation |

## Interpretation

OI rising into a strong move can mean crowded continuation or a late squeeze setup. OI falling into a strong move can mean short covering, long liquidation, or crowded-risk decay. Both need forward labels, depth, and funding/fee costs before paper action.
