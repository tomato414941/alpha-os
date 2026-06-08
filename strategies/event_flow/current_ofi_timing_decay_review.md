# Current OFI Timing Decay Review

This reviews the OFI short-horizon lifecycle as a timing/state problem. It is not a trading rule and not a promotion list.

| asset | decision | paper | first audit 15m | repeat 5m | repeat audit 15m | second repeat 5m | status | next step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| BNB | paper_short | 50.68980790 | 13.37317770 | 20.77493817 | 16.05242690 | -8.10909211 | second_repeat_decay | treat OFI as timing-sensitive; learn entry cooldown/state filters before opening another repeat |
| ETH | paper_short | 96.39539639 | 2.95037470 | 15.33742331 | -16.51527663 |  | repeat_fill_audit_decay | do not promote; require a fresh independent OFI state before any new label |
| SUI | paper_short | 84.39697838 | 3.41628781 | 31.78689644 | 24.83421838 | -9.90465122 | second_repeat_decay | treat OFI as timing-sensitive; learn entry cooldown/state filters before opening another repeat |

## Interpretation

- BNB: The signal survived paper, repeat, and fill audit, then failed when immediately chased again.
- ETH: The first mark wins did not survive the later fill-audit window.
- SUI: The signal survived paper, repeat, and fill audit, then failed when immediately chased again.
