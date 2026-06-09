# Current Token Unlock Event Window Risk Check

This checks matured token-unlock event-window labels against rough spread, taker fee, and funding. It is not a live order list.

| ticket | asset | decision | outcome | dir bps | cost bps | funding bps | net bps | action | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| unlock-event-zro-paper-short | ZRO | paper_short | paper_mark_win | 251.98 | 14.11 | 0.16 | 238.03 | cost_adjusted_event_window_probe | the first event-window label survives rough trading costs |
| unlock-event-hype-paper-short | HYPE | paper_short | paper_mark_win | 163.13 | 10.16 | 0.16 | 153.14 | cost_adjusted_event_window_probe | the first event-window label survives rough trading costs |
| unlock-event-eigen-paper-short | EIGEN | paper_short | paper_mark_loss | -55.37 | 26.32 | 0.16 | -81.53 | event_window_label_not_supported | the first event-window label did not move in the intended direction |
| unlock-event-me-paper-long | ME | paper_long | paper_mark_loss | -176.98 | 34.17 | 0.22 | -210.93 | event_window_label_not_supported | the first event-window label did not move in the intended direction |
