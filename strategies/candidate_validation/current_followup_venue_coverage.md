# Current Follow-Up Venue Coverage

This checks whether follow-up queue assets exist on major perp venues. It prevents Hyperliquid-only execution context from silently dropping otherwise testable candidates.

| asset | priority | source | HL | OKX | Binance | venues | action | reason |
| --- | ---: | --- | --- | --- | --- | ---: | --- | --- |
| WLD | 10.0571 | hl_candidate;okx_pressure;liquidation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| ETH | 4.5510 | okx_pressure;liquidation;l2_imbalance | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| MEGA | 3.8916 | hl_candidate | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| BTC | 3.6217 | liquidation;l2_imbalance | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| ONDO | 3.6106 | liquidation;sector_rotation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| XRP | 3.4627 | okx_pressure;liquidation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| JTO | 3.4579 | liquidation;l2_imbalance | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| XPL | 3.4493 | l2_imbalance;sector_rotation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| IP | 3.3166 | hl_candidate | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| LTC | 3.2959 | okx_pressure;liquidation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| ZORA | 3.2743 | hl_candidate | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| KAITO | 3.1882 | hl_candidate | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| AIXBT | 3.1603 | hl_candidate | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| SOL | 3.1187 | okx_pressure;liquidation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| PUMP | 2.9792 | liquidation;sector_rotation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| XLM | 2.9178 | okx_pressure;l2_imbalance | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| ZRO | 2.5173 | hl_candidate | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| TON | 2.1872 | okx_pressure;liquidation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| PYTH | 1.9270 | sector_rotation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| AI | 1.8310 | okx_pressure | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| DOGE | 1.8179 | liquidation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| HYPE | 1.7045 | liquidation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| TURBO | 1.6661 | sector_rotation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| FIL | 1.4957 | okx_pressure | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| ENA | 1.4836 | okx_pressure | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| POL | 1.4202 | sector_rotation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| BNB | 1.3932 | liquidation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| ICP | 1.3465 | okx_pressure | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| AAVE | 1.2914 | okx_pressure | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| LINK | 1.2570 | sector_rotation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| BABY | 1.2532 | okx_pressure | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| RENDER | 1.2400 | okx_pressure | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| JUP | 1.2318 | sector_rotation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| AVAX | 1.2204 | okx_pressure | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| PEPE | 3.7269 | okx_pressure;liquidation | False | True | False | 1 | okx_only_followup | candidate is missing from Hyperliquid but exists on OKX USDT swap |
| XMR | 3.6004 | hl_candidate | True | False | False | 1 | hyperliquid_only_followup | candidate is currently visible on Hyperliquid only in this check |
| APEX | 3.1524 | hl_candidate | True | False | False | 1 | hyperliquid_only_followup | candidate is currently visible on Hyperliquid only in this check |
| ALLO | 3.0965 | liquidation | False | True | False | 1 | okx_only_followup | candidate is missing from Hyperliquid but exists on OKX USDT swap |
| BSV | 3.0874 | hl_candidate | True | False | False | 1 | hyperliquid_only_followup | candidate is currently visible on Hyperliquid only in this check |
| SAGA | 3.0365 | hl_candidate | True | False | False | 1 | hyperliquid_only_followup | candidate is currently visible on Hyperliquid only in this check |
| HOME | 2.8356 | okx_pressure | False | True | False | 1 | okx_only_followup | candidate is missing from Hyperliquid but exists on OKX USDT swap |
| H | 2.7846 | liquidation | False | True | False | 1 | okx_only_followup | candidate is missing from Hyperliquid but exists on OKX USDT swap |
| BILL | 2.0283 | okx_pressure | False | True | False | 1 | okx_only_followup | candidate is missing from Hyperliquid but exists on OKX USDT swap |
| EDEN | 1.9688 | okx_pressure | False | True | False | 1 | okx_only_followup | candidate is missing from Hyperliquid but exists on OKX USDT swap |
| LAB | 1.8363 | liquidation | False | True | False | 1 | okx_only_followup | candidate is missing from Hyperliquid but exists on OKX USDT swap |
| BEAT | 1.7502 | liquidation | False | True | False | 1 | okx_only_followup | candidate is missing from Hyperliquid but exists on OKX USDT swap |
| USELESS | 1.2713 | okx_pressure | False | True | False | 1 | okx_only_followup | candidate is missing from Hyperliquid but exists on OKX USDT swap |
| UP | 1.2711 | okx_pressure | False | True | False | 1 | okx_only_followup | candidate is missing from Hyperliquid but exists on OKX USDT swap |
| BASED | 1.2596 | okx_pressure | False | True | False | 1 | okx_only_followup | candidate is missing from Hyperliquid but exists on OKX USDT swap |
| CHZ | 1.2442 | okx_pressure | False | True | False | 1 | okx_only_followup | candidate is missing from Hyperliquid but exists on OKX USDT swap |

## Interpretation

Multi-venue coverage improves observability and execution optionality. Single-venue candidates can still be useful, but venue-specific data, fees, and depth must be checked instead of assuming Hyperliquid is the only route.
