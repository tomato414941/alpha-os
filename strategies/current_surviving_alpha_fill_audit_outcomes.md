# Current Surviving Alpha Fill Audit Outcomes

These outcomes check fresh fill-audit paper tickets against public 1m candle path. They do not prove live fill quality.

| ticket | horizon | status | outcome | asset | side | close | adverse | stop | candles | next step |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| fill-audit-bera-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | BERA | long | -40.2670 | -62.2646 | stop_triggered | 14 | do not promote BERA long; fresh fill audit hit the stop |
| fill-audit-bera-long-50bps-stop | 1h | pending | pending | BERA | long | -40.2670 | -62.2646 | stop_triggered | 14 | wait for 1h fill-audit checkpoint for BERA long |
