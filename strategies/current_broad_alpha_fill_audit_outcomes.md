# Current Broad Alpha Fill Audit Outcomes

These outcomes check fresh broad paper fill-audit tickets against public 1m candle path. They do not prove live fill quality.

| ticket | horizon | status | outcome | asset | venue | side | close | adverse | stop | candles | next step |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| broad-fill-audit-pol-short-50bps-stop | 15m | pending | pending | POL |  | short | -75.5499 | -75.5499 | stop_triggered | 10 | wait for 15m fill-audit checkpoint for POL short |
| broad-fill-audit-pol-short-50bps-stop | 1h | pending | pending | POL |  | short | -75.5499 | -75.5499 | stop_triggered | 10 | wait for 1h fill-audit checkpoint for POL short |
| broad-fill-audit-zec-long-50bps-stop | 15m | pending | pending | ZEC |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for ZEC |
| broad-fill-audit-zec-long-50bps-stop | 1h | pending | pending | ZEC |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for ZEC |
| broad-fill-audit-chip-long-50bps-stop | 15m | pending | pending | CHIP |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for CHIP |
| broad-fill-audit-chip-long-50bps-stop | 1h | pending | pending | CHIP |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for CHIP |
| broad-fill-audit-eigen-short-50bps-stop | 15m | pending | pending | EIGEN |  | short | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for EIGEN |
| broad-fill-audit-eigen-short-50bps-stop | 1h | pending | pending | EIGEN |  | short | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for EIGEN |
| broad-fill-audit-hype-long-50bps-stop | 15m | pending | pending | HYPE |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for HYPE |
| broad-fill-audit-hype-long-50bps-stop | 1h | pending | pending | HYPE |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for HYPE |
| broad-fill-audit-beat-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | BEAT | OKX | long | -123.3158 | -220.3699 | stop_triggered | 14 | do not promote BEAT long; fresh fill audit hit the stop |
| broad-fill-audit-beat-long-50bps-stop | 1h | pending | pending | BEAT | OKX | long | -61.6579 | -220.3699 | stop_triggered | 16 | wait for 1h fill-audit checkpoint for BEAT long |
| broad-fill-audit-btc-short-50bps-stop | 15m | pending | pending | BTC |  | short | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for BTC |
| broad-fill-audit-btc-short-50bps-stop | 1h | pending | pending | BTC |  | short | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for BTC |
