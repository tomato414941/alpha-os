# Current Dislocation Execution Check

This checks whether the current monitor candidate still has 24-hour headroom after fee scenarios and visible public-book taker slippage. It is not a trade instruction and does not use account-specific fee tiers.

| asset | fee bps/fill/venue | mean net24 | fee round trip | taker slippage bps | fee-only net24 | conservative taker net24 | sizes valid | book filled | action | reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| STABLE | 0.000000 | 0.002214 | 0.000000 | 11.294032 | 0.002214 | 0.001084 | True | True | conservative_taker_monitor | survives fees and visible taker slippage in the conservative check |
| STABLE | 0.250000 | 0.002214 | 0.000100 | 11.294032 | 0.002114 | 0.000984 | True | True | conservative_taker_monitor | survives fees and visible taker slippage in the conservative check |
| STABLE | 0.500000 | 0.002214 | 0.000200 | 11.294032 | 0.002014 | 0.000884 | True | True | conservative_taker_monitor | survives fees and visible taker slippage in the conservative check |
| STABLE | 1.000000 | 0.002214 | 0.000400 | 11.294032 | 0.001814 | 0.000684 | True | True | conservative_taker_monitor | survives fees and visible taker slippage in the conservative check |
| STABLE | 2.000000 | 0.002214 | 0.000800 | 11.294032 | 0.001414 | 0.000284 | True | True | conservative_taker_monitor | survives fees and visible taker slippage in the conservative check |

## Interpretation

`fee_only_monitor` means the 24-hour monitor survives the configured fee scenario but may need maker execution or lower slippage. `conservative_taker_monitor` means it also survives subtracting visible public-book taker slippage from the already friction-adjusted proxy.
