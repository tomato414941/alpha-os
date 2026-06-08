# Current Broad Alpha Fill Audit Outcomes

These outcomes check fresh broad paper fill-audit tickets against public 1m candle path. They do not prove live fill quality.

| ticket | horizon | status | outcome | asset | venue | side | close | adverse | stop | candles | next step |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| broad-fill-audit-pol-short-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | POL |  | short | -72.0379 | -75.5499 | stop_triggered | 14 | do not promote POL short; fresh fill audit hit the stop |
| broad-fill-audit-pol-short-50bps-stop | 1h | pending | pending | POL |  | short | -70.4065 | -75.5499 | stop_triggered | 15 | wait for 1h fill-audit checkpoint for POL short |
| broad-fill-audit-zec-long-50bps-stop | 15m | pending | pending | ZEC |  | long | -0.8552 | -23.3040 | stop_survived | 3 | wait for 15m fill-audit checkpoint for ZEC long |
| broad-fill-audit-zec-long-50bps-stop | 1h | pending | pending | ZEC |  | long | -0.8552 | -23.3040 | stop_survived | 3 | wait for 1h fill-audit checkpoint for ZEC long |
| broad-fill-audit-chip-long-50bps-stop | 15m | pending | pending | CHIP |  | long | 2.6623 | -11.8326 | stop_survived | 3 | wait for 15m fill-audit checkpoint for CHIP long |
| broad-fill-audit-chip-long-50bps-stop | 1h | pending | pending | CHIP |  | long | 2.6623 | -11.8326 | stop_survived | 3 | wait for 1h fill-audit checkpoint for CHIP long |
| broad-fill-audit-eigen-short-50bps-stop | 15m | pending | pending | EIGEN |  | short | 0.0000 | -10.9230 | stop_survived | 3 | wait for 15m fill-audit checkpoint for EIGEN short |
| broad-fill-audit-eigen-short-50bps-stop | 1h | pending | pending | EIGEN |  | short | 0.0000 | -10.9230 | stop_survived | 3 | wait for 1h fill-audit checkpoint for EIGEN short |
| broad-fill-audit-hype-long-50bps-stop | 15m | pending | pending | HYPE |  | long | -2.0389 | -3.4504 | stop_survived | 3 | wait for 15m fill-audit checkpoint for HYPE long |
| broad-fill-audit-hype-long-50bps-stop | 1h | pending | pending | HYPE |  | long | -2.0389 | -3.4504 | stop_survived | 3 | wait for 1h fill-audit checkpoint for HYPE long |
| broad-fill-audit-beat-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | BEAT | OKX | long | -123.3158 | -220.3699 | stop_triggered | 14 | do not promote BEAT long; fresh fill audit hit the stop |
| broad-fill-audit-beat-long-50bps-stop | 1h | pending | pending | BEAT | OKX | long | -232.4732 | -240.9226 | stop_triggered | 20 | wait for 1h fill-audit checkpoint for BEAT long |
| broad-fill-audit-btc-short-50bps-stop | 15m | pending | pending | BTC |  | short | -1.7357 | -5.5206 | stop_survived | 3 | wait for 15m fill-audit checkpoint for BTC short |
| broad-fill-audit-btc-short-50bps-stop | 1h | pending | pending | BTC |  | short | -1.7357 | -5.5206 | stop_survived | 3 | wait for 1h fill-audit checkpoint for BTC short |
