# Current Broad Alpha Fill Audit Outcomes

These outcomes check fresh broad paper fill-audit tickets against public 1m candle path. They do not prove live fill quality.

| ticket | horizon | status | outcome | asset | venue | side | close | adverse | stop | candles | next step |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| broad-fill-audit-pol-short-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | POL |  | short | -72.0379 | -75.5499 | stop_triggered | 14 | do not promote POL short; fresh fill audit hit the stop |
| broad-fill-audit-pol-short-50bps-stop | 1h | pending | pending | POL |  | short | -48.2670 | -75.5499 | stop_triggered | 22 | wait for 1h fill-audit checkpoint for POL short |
| broad-fill-audit-zro-short-50bps-stop | 15m | pending | pending | ZRO |  | short | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for ZRO |
| broad-fill-audit-zro-short-50bps-stop | 1h | pending | pending | ZRO |  | short | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for ZRO |
| broad-fill-audit-zec-long-50bps-stop | 15m | pending | pending | ZEC |  | long | -3.2070 | -53.6634 | stop_triggered | 10 | wait for 15m fill-audit checkpoint for ZEC long |
| broad-fill-audit-zec-long-50bps-stop | 1h | pending | pending | ZEC |  | long | -3.2070 | -53.6634 | stop_triggered | 10 | wait for 1h fill-audit checkpoint for ZEC long |
| broad-fill-audit-chip-long-50bps-stop | 15m | pending | pending | CHIP |  | long | 63.6001 | -11.8326 | stop_survived | 10 | wait for 15m fill-audit checkpoint for CHIP long |
| broad-fill-audit-chip-long-50bps-stop | 1h | pending | pending | CHIP |  | long | 63.6001 | -11.8326 | stop_survived | 10 | wait for 1h fill-audit checkpoint for CHIP long |
| broad-fill-audit-eigen-short-50bps-stop | 15m | pending | pending | EIGEN |  | short | -21.8221 | -43.5493 | stop_survived | 10 | wait for 15m fill-audit checkpoint for EIGEN short |
| broad-fill-audit-eigen-short-50bps-stop | 1h | pending | pending | EIGEN |  | short | -21.8221 | -43.5493 | stop_survived | 10 | wait for 1h fill-audit checkpoint for EIGEN short |
| broad-fill-audit-hype-long-50bps-stop | 15m | pending | pending | HYPE |  | long | 11.2922 | -11.4490 | stop_survived | 10 | wait for 15m fill-audit checkpoint for HYPE long |
| broad-fill-audit-hype-long-50bps-stop | 1h | pending | pending | HYPE |  | long | 11.2922 | -11.4490 | stop_survived | 10 | wait for 1h fill-audit checkpoint for HYPE long |
| broad-fill-audit-beat-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | BEAT | OKX | long | -123.3158 | -220.3699 | stop_triggered | 14 | do not promote BEAT long; fresh fill audit hit the stop |
| broad-fill-audit-beat-long-50bps-stop | 1h | pending | pending | BEAT | OKX | long | -171.2720 | -277.4606 | stop_triggered | 27 | wait for 1h fill-audit checkpoint for BEAT long |
| broad-fill-audit-btc-short-50bps-stop | 15m | pending | pending | BTC |  | short | -9.3026 | -12.6095 | stop_survived | 10 | wait for 15m fill-audit checkpoint for BTC short |
| broad-fill-audit-btc-short-50bps-stop | 1h | pending | pending | BTC |  | short | -9.3026 | -12.6095 | stop_survived | 10 | wait for 1h fill-audit checkpoint for BTC short |
