# Current Dislocation Execution Check

This checks whether the current monitor candidate still has 24-hour headroom after fee scenarios and visible public-book taker slippage. It is not a trade instruction and does not use account-specific fee tiers.

| asset | fee bps/fill/venue | mean net24 | fee round trip | taker slippage bps | fee-only net24 | conservative taker net24 | sizes valid | book filled | action | reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| STABLE | 0.000000 | 0.002103 | 0.000000 | 11.294032 | 0.002103 | 0.000974 | True | True | conservative_taker_monitor | survives fees and visible taker slippage in the conservative check |
| STABLE | 0.250000 | 0.002103 | 0.000100 | 11.294032 | 0.002003 | 0.000874 | True | True | conservative_taker_monitor | survives fees and visible taker slippage in the conservative check |
| STABLE | 0.500000 | 0.002103 | 0.000200 | 11.294032 | 0.001903 | 0.000774 | True | True | conservative_taker_monitor | survives fees and visible taker slippage in the conservative check |
| STABLE | 1.000000 | 0.002103 | 0.000400 | 11.294032 | 0.001703 | 0.000574 | True | True | conservative_taker_monitor | survives fees and visible taker slippage in the conservative check |
| STABLE | 2.000000 | 0.002103 | 0.000800 | 11.294032 | 0.001303 | 0.000174 | True | True | conservative_taker_monitor | survives fees and visible taker slippage in the conservative check |

## Interpretation

`fee_only_monitor` means the 24-hour monitor survives the configured fee scenario but may need maker execution or lower slippage. `conservative_taker_monitor` means it also survives subtracting visible public-book taker slippage from the already friction-adjusted proxy.
