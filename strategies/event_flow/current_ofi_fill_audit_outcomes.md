# Current Broad Alpha Fill Audit Outcomes

These outcomes check fresh broad paper fill-audit tickets against public 1m candle path. They do not prove live fill quality.

| ticket | horizon | status | outcome | asset | venue | side | close | adverse | stop | candles | next step |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| broad-fill-audit-eth-short-50bps-stop | 15m | ready | paper_fill_audit_win | ETH |  | short | 2.9504 | -5.3063 | stop_survived | 14 | compare ETH short fill audit against prior broad paper path before promotion |
| broad-fill-audit-eth-short-50bps-stop | 1h | pending | pending | ETH |  | short | 4.1310 | -5.3063 | stop_survived | 15 | wait for 1h fill-audit checkpoint for ETH short |
| broad-fill-audit-sui-short-50bps-stop | 15m | ready | paper_fill_audit_win | SUI |  | short | 3.4163 | 0.0000 | stop_survived | 14 | compare SUI short fill audit against prior broad paper path before promotion |
| broad-fill-audit-sui-short-50bps-stop | 1h | pending | pending | SUI |  | short | 6.3088 | -5.6449 | stop_survived | 15 | wait for 1h fill-audit checkpoint for SUI short |
| broad-fill-audit-bnb-short-50bps-stop | 15m | ready | paper_fill_audit_win | BNB |  | short | 13.3732 | 0.0000 | stop_survived | 14 | compare BNB short fill audit against prior broad paper path before promotion |
| broad-fill-audit-bnb-short-50bps-stop | 1h | pending | pending | BNB |  | short | 14.8613 | 0.0000 | stop_survived | 15 | wait for 1h fill-audit checkpoint for BNB short |
