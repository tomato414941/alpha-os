# Current Token Unlock Event Window Risk Check

This checks matured token-unlock event-window labels against rough spread, taker fee, and funding. It is not a live order list.

| ticket | asset | decision | outcome | dir bps | cost bps | funding bps | net bps | action | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| unlock-event-hype-paper-short | HYPE | paper_short | paper_mark_win | 0.63 | 11.14 | 0.08 | -10.43 | cost_adjusted_event_window_failed | the first directional mark does not survive rough spread, taker-fee, and funding |
| unlock-event-zro-paper-short | ZRO | paper_short | paper_mark_flat | -0.00 | 14.56 | 0.03 | -14.53 | event_window_label_not_supported | the first event-window label did not move in the intended direction |
| unlock-event-kaito-paper-short | KAITO | paper_short | paper_mark_flat | -0.00 | 29.05 | 0.00 | -29.05 | event_window_label_not_supported | the first event-window label did not move in the intended direction |
| unlock-event-eigen-paper-short | EIGEN | paper_short | paper_mark_flat | -0.00 | 31.54 | 0.03 | -31.51 | event_window_label_not_supported | the first event-window label did not move in the intended direction |
| unlock-event-me-paper-long | ME | paper_long | paper_mark_flat | 0.00 | 45.18 | 0.11 | -45.07 | event_window_label_not_supported | the first event-window label did not move in the intended direction |
