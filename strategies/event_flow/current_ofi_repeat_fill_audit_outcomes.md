# Current Broad Alpha Fill Audit Outcomes

These outcomes check fresh broad paper fill-audit tickets against public 1m candle path. They do not prove live fill quality.

| ticket | horizon | status | outcome | asset | venue | side | close | adverse | stop | candles | next step |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| broad-fill-audit-sui-short-50bps-stop | 15m | ready | paper_fill_audit_win | SUI |  | short | 24.8342 | -8.8208 | stop_survived | 14 | compare SUI short fill audit against prior broad paper path before promotion |
| broad-fill-audit-sui-short-50bps-stop | 1h | pending | pending | SUI |  | short | 24.0397 | -8.8208 | stop_survived | 15 | wait for 1h fill-audit checkpoint for SUI short |
| broad-fill-audit-bnb-short-50bps-stop | 15m | ready | paper_fill_audit_win | BNB |  | short | 16.0524 | -4.1289 | stop_survived | 14 | compare BNB short fill audit against prior broad paper path before promotion |
| broad-fill-audit-bnb-short-50bps-stop | 1h | pending | pending | BNB |  | short | 16.3839 | -4.1289 | stop_survived | 15 | wait for 1h fill-audit checkpoint for BNB short |
| broad-fill-audit-eth-short-50bps-stop | 15m | ready | paper_fill_audit_loss | ETH |  | short | -16.5153 | -20.6356 | stop_survived | 14 | do not promote ETH short; fresh fill audit failed at 15m |
| broad-fill-audit-eth-short-50bps-stop | 1h | pending | pending | ETH |  | short | -14.7484 | -20.6356 | stop_survived | 15 | wait for 1h fill-audit checkpoint for ETH short |
