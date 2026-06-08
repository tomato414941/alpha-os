# Current Broad Alpha Fill Audit Outcomes

These outcomes check fresh broad paper fill-audit tickets against public 1m candle path. They do not prove live fill quality.

| ticket | horizon | status | outcome | asset | venue | side | close | adverse | stop | candles | next step |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| broad-fill-audit-pol-short-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | POL |  | short | -72.0379 | -75.5499 | stop_triggered | 14 | do not promote POL short; fresh fill audit hit the stop |
| broad-fill-audit-pol-short-50bps-stop | 1h | pending | pending | POL |  | short | -98.4421 | -104.5538 | stop_triggered | 38 | wait for 1h fill-audit checkpoint for POL short |
| broad-fill-audit-zec-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | ZEC |  | long | -25.8696 | -53.6634 | stop_triggered | 14 | do not promote ZEC long; fresh fill audit hit the stop |
| broad-fill-audit-zec-long-50bps-stop | 1h | pending | pending | ZEC |  | long | 121.4376 | -53.6634 | stop_triggered | 26 | wait for 1h fill-audit checkpoint for ZEC long |
| broad-fill-audit-zro-short-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | ZRO |  | short | -54.5720 | -65.0470 | stop_triggered | 14 | do not promote ZRO short; fresh fill audit hit the stop |
| broad-fill-audit-zro-short-50bps-stop | 1h | pending | pending | ZRO |  | short | -145.6846 | -145.6846 | stop_triggered | 16 | wait for 1h fill-audit checkpoint for ZRO short |
| broad-fill-audit-wld-short-50bps-stop | 15m | pending | pending | WLD |  | short | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for WLD |
| broad-fill-audit-wld-short-50bps-stop | 1h | pending | pending | WLD |  | short | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for WLD |
| broad-fill-audit-chip-long-50bps-stop | 15m | ready | paper_fill_audit_win | CHIP |  | long | 97.6187 | -11.8326 | stop_survived | 14 | compare CHIP long fill audit against prior broad paper path before promotion |
| broad-fill-audit-chip-long-50bps-stop | 1h | pending | pending | CHIP |  | long | 141.9908 | -11.8326 | stop_survived | 26 | wait for 1h fill-audit checkpoint for CHIP long |
| broad-fill-audit-hype-long-50bps-stop | 15m | ready | paper_fill_audit_win | HYPE |  | long | 7.3713 | -11.4490 | stop_survived | 14 | compare HYPE long fill audit against prior broad paper path before promotion |
| broad-fill-audit-hype-long-50bps-stop | 1h | pending | pending | HYPE |  | long | 37.4837 | -11.4490 | stop_survived | 26 | wait for 1h fill-audit checkpoint for HYPE long |
| broad-fill-audit-inj-long-50bps-stop | 15m | pending | pending | INJ |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for INJ |
| broad-fill-audit-inj-long-50bps-stop | 1h | pending | pending | INJ |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for INJ |
| broad-fill-audit-wld-long-50bps-stop | 15m | pending | pending | WLD |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for WLD |
| broad-fill-audit-wld-long-50bps-stop | 1h | pending | pending | WLD |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for WLD |
| broad-fill-audit-fet-long-50bps-stop | 15m | pending | pending | FET |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for FET |
| broad-fill-audit-fet-long-50bps-stop | 1h | pending | pending | FET |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for FET |
| broad-fill-audit-near-long-50bps-stop | 15m | pending | pending | NEAR |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for NEAR |
| broad-fill-audit-near-long-50bps-stop | 1h | pending | pending | NEAR |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for NEAR |
| broad-fill-audit-btc-short-50bps-stop | 15m | ready | paper_fill_audit_loss | BTC |  | short | -19.6884 | -24.0880 | stop_survived | 14 | do not promote BTC short; fresh fill audit failed at 15m |
| broad-fill-audit-btc-short-50bps-stop | 1h | pending | pending | BTC |  | short | -52.5911 | -66.4702 | stop_triggered | 26 | wait for 1h fill-audit checkpoint for BTC short |
| broad-fill-audit-eigen-short-50bps-stop | 15m | ready | paper_fill_audit_loss | EIGEN |  | short | -21.8221 | -43.5493 | stop_survived | 14 | do not promote EIGEN short; fresh fill audit failed at 15m |
| broad-fill-audit-eigen-short-50bps-stop | 1h | pending | pending | EIGEN |  | short | -92.0910 | -92.0910 | stop_triggered | 26 | wait for 1h fill-audit checkpoint for EIGEN short |
| broad-fill-audit-beat-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | BEAT | OKX | long | -123.3158 | -220.3699 | stop_triggered | 14 | do not promote BEAT long; fresh fill audit hit the stop |
| broad-fill-audit-beat-long-50bps-stop | 1h | pending | pending | BEAT | OKX | long | -30.8290 | -277.4606 | stop_triggered | 43 | wait for 1h fill-audit checkpoint for BEAT long |
