# Current Broad Alpha Fill Audit Outcomes

These outcomes check fresh broad paper fill-audit tickets against public 1m candle path. They do not prove live fill quality.

| ticket | horizon | status | outcome | asset | venue | side | close | adverse | stop | candles | next step |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| broad-fill-audit-pol-short-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | POL |  | short | -72.0379 | -75.5499 | stop_triggered | 14 | do not promote POL short; fresh fill audit hit the stop |
| broad-fill-audit-pol-short-50bps-stop | 1h | pending | pending | POL |  | short | -67.2677 | -75.5499 | stop_triggered | 26 | wait for 1h fill-audit checkpoint for POL short |
| broad-fill-audit-zro-short-50bps-stop | 15m | pending | pending | ZRO |  | short | -8.9625 | -23.0139 | stop_survived | 4 | wait for 15m fill-audit checkpoint for ZRO short |
| broad-fill-audit-zro-short-50bps-stop | 1h | pending | pending | ZRO |  | short | -8.9625 | -23.0139 | stop_survived | 4 | wait for 1h fill-audit checkpoint for ZRO short |
| broad-fill-audit-zec-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | ZEC |  | long | -25.8696 | -53.6634 | stop_triggered | 14 | do not promote ZEC long; fresh fill audit hit the stop |
| broad-fill-audit-zec-long-50bps-stop | 1h | pending | pending | ZEC |  | long | -25.8696 | -53.6634 | stop_triggered | 14 | wait for 1h fill-audit checkpoint for ZEC long |
| broad-fill-audit-chip-long-50bps-stop | 15m | ready | paper_fill_audit_win | CHIP |  | long | 97.6187 | -11.8326 | stop_survived | 14 | compare CHIP long fill audit against prior broad paper path before promotion |
| broad-fill-audit-chip-long-50bps-stop | 1h | pending | pending | CHIP |  | long | 97.6187 | -11.8326 | stop_survived | 14 | wait for 1h fill-audit checkpoint for CHIP long |
| broad-fill-audit-eigen-short-50bps-stop | 15m | ready | paper_fill_audit_loss | EIGEN |  | short | -21.8221 | -43.5493 | stop_survived | 14 | do not promote EIGEN short; fresh fill audit failed at 15m |
| broad-fill-audit-eigen-short-50bps-stop | 1h | pending | pending | EIGEN |  | short | -21.8221 | -43.5493 | stop_survived | 14 | wait for 1h fill-audit checkpoint for EIGEN short |
| broad-fill-audit-hype-long-50bps-stop | 15m | ready | paper_fill_audit_win | HYPE |  | long | 7.3713 | -11.4490 | stop_survived | 14 | compare HYPE long fill audit against prior broad paper path before promotion |
| broad-fill-audit-hype-long-50bps-stop | 1h | pending | pending | HYPE |  | long | 7.3713 | -11.4490 | stop_survived | 14 | wait for 1h fill-audit checkpoint for HYPE long |
| broad-fill-audit-beat-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | BEAT | OKX | long | -123.3158 | -220.3699 | stop_triggered | 14 | do not promote BEAT long; fresh fill audit hit the stop |
| broad-fill-audit-beat-long-50bps-stop | 1h | pending | pending | BEAT | OKX | long | -132.9071 | -277.4606 | stop_triggered | 31 | wait for 1h fill-audit checkpoint for BEAT long |
| broad-fill-audit-btc-short-50bps-stop | 15m | ready | paper_fill_audit_loss | BTC |  | short | -19.6884 | -24.0880 | stop_survived | 14 | do not promote BTC short; fresh fill audit failed at 15m |
| broad-fill-audit-btc-short-50bps-stop | 1h | pending | pending | BTC |  | short | -19.6884 | -24.0880 | stop_survived | 14 | wait for 1h fill-audit checkpoint for BTC short |
