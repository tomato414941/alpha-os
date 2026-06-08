# Current Broad Alpha Fill Audit Outcomes

These outcomes check fresh broad paper fill-audit tickets against public 1m candle path. They do not prove live fill quality.

| ticket | horizon | status | outcome | asset | venue | side | close | adverse | stop | candles | next step |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| broad-fill-audit-pol-short-50bps-stop | 15m | pending | pending | POL |  | short | -57.3360 | -61.3613 | stop_triggered | 9 | wait for 15m fill-audit checkpoint for POL short |
| broad-fill-audit-pol-short-50bps-stop | 1h | pending | pending | POL |  | short | -57.3360 | -61.3613 | stop_triggered | 9 | wait for 1h fill-audit checkpoint for POL short |
| broad-fill-audit-beat-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | BEAT | OKX | long | -123.3158 | -220.3699 | stop_triggered | 14 | do not promote BEAT long; fresh fill audit hit the stop |
| broad-fill-audit-beat-long-50bps-stop | 1h | pending | pending | BEAT | OKX | long | -123.3158 | -220.3699 | stop_triggered | 14 | wait for 1h fill-audit checkpoint for BEAT long |
