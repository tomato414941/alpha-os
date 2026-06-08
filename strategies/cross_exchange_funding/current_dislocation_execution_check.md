# Current Dislocation Execution Check

This checks whether the current monitor candidate still has 24-hour headroom after fee scenarios and visible public-book taker slippage. It is not a trade instruction and does not use account-specific fee tiers.

| asset | fee bps/fill/venue | mean net24 | fee round trip | taker slippage bps | fee-only net24 | conservative taker net24 | sizes valid | book filled | action | reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| MON | 0.000000 | 0.001026 | 0.000000 | 9.275463 | 0.001026 | 0.000098 | True | True | conservative_taker_monitor | survives fees and visible taker slippage in the conservative check |
| MON | 0.250000 | 0.001026 | 0.000100 | 9.275463 | 0.000926 | -0.000002 | True | True | fee_only_monitor | survives fees, but visible taker slippage consumes the conservative edge |
| MON | 0.500000 | 0.001026 | 0.000200 | 9.275463 | 0.000826 | -0.000102 | True | True | fee_only_monitor | survives fees, but visible taker slippage consumes the conservative edge |
| MON | 1.000000 | 0.001026 | 0.000400 | 9.275463 | 0.000626 | -0.000302 | True | True | fee_only_monitor | survives fees, but visible taker slippage consumes the conservative edge |
| MON | 2.000000 | 0.001026 | 0.000800 | 9.275463 | 0.000226 | -0.000702 | True | True | fee_only_monitor | survives fees, but visible taker slippage consumes the conservative edge |

## Interpretation

`fee_only_monitor` means the 24-hour monitor survives the configured fee scenario but may need maker execution or lower slippage. `conservative_taker_monitor` means it also survives subtracting visible public-book taker slippage from the already friction-adjusted proxy.
