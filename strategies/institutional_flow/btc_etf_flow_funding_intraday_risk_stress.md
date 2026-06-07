# BTC ETF Flow Funding Intraday Risk Stress

This measures 1h mark-to-market adverse excursion for the BTC ETF-flow/funding short candidate. For a short, adverse excursion is the largest high above the entry close during the 120-hour hold. Liquidation columns are rough flags using 50%, 33.3%, and 20% adverse moves for 2x, 3x, and 5x leverage.

| group | trades | mean price net | hit | mean adverse | max adverse | liq 2x | liq 3x | liq 5x | action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| entry_offset_hours_0 | 21 | 0.02047865 | 0.5238 | 0.04357517 | 0.12139435 | 0 | 0 | 0 | survives_5x_buffer |
| entry_offset_hours_8 | 21 | 0.02351709 | 0.6667 | 0.04264361 | 0.14130097 | 0 | 0 | 0 | survives_5x_buffer |
| entry_offset_hours_16 | 21 | 0.02837573 | 0.6667 | 0.03725806 | 0.12457860 | 0 | 0 | 0 | survives_5x_buffer |
| entry_offset_hours_24 | 21 | 0.02635495 | 0.6190 | 0.03441678 | 0.12743422 | 0 | 0 | 0 | survives_5x_buffer |
| entry_offset_hours_32 | 21 | 0.02348725 | 0.6667 | 0.03473943 | 0.20283906 | 0 | 0 | 1 | survives_3x_buffer |
| entry_offset_hours_48 | 21 | 0.01405478 | 0.6667 | 0.04123588 | 0.13929317 | 0 | 0 | 0 | survives_5x_buffer |

## Caveat

This is not an exchange liquidation model. Maintenance margin, mark/index divergence, funding timestamps, and stop fills are still ignored. The purpose is to see whether the candidate requires obviously unsafe leverage.
