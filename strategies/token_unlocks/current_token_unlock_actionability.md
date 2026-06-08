# Current Token Unlock Actionability

This separates scheduled supply events from tradable candidates. Without event-window labels, an unlock is context, not an alpha candidate.

| symbol | status | action | score | ticket | in | value USD | % supply | funding | volume | impact | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HYPE | unlock_event_label_pending | create_event_window_label | 49.5899 | paper_short_candidate 32.62 | 28 | 594900000 | 4.5000 | 0.10950000 | 638826720 | 0.00051418 | unlock short thesis has no event-window forward label yet |
| ZRO | unlock_event_label_pending | create_event_window_label | 47.5604 | paper_short_candidate 31.47 | 12 | 34450000 | 10.2000 | 0.11050828 | 2972259 | 0.00085054 | unlock short thesis has no event-window forward label yet |
| KAITO | unlock_event_label_pending | create_event_window_label | 45.0427 | paper_short_candidate 27.90 | 12 | 7950000 | 7.3000 | 0.10950000 | 154063 | 0.00204240 | unlock short thesis has no event-window forward label yet |
| EIGEN | unlock_event_label_pending | create_event_window_label | 42.7240 | paper_short_candidate 25.30 | 23 | 7750000 | 5.0000 | 0.10950000 | 1036763 | 0.00222717 | unlock short thesis has no event-window forward label yet |
| ME | unlock_event_execution_blocked | wait_for_tighter_depth | 32.9347 | wide_impact_watch 31.51 | 2 | 10430000 | 30.8000 | 0.02297573 | 157905 | 0.00367955 | visible impact spread is too wide |
| PIXEL | unlock_event_not_tradeable | do_not_probe | 16.2091 | too_illiquid 12.44 | 11 | 630970 | 11.8000 | 0.00000000 | 0 | 0.00000000 | perp venue volume is too low for a paper probe |
| CYBER | unlock_event_not_tradeable | do_not_probe | 15.1564 | too_illiquid 6.69 | 7 | 2780000 | 5.9000 | 0.00000000 | 0 | 0.00000000 | perp venue volume is too low for a paper probe |
| AI | unlock_event_not_tradeable | do_not_probe | 14.8116 | too_illiquid 12.94 | 23 | 444360 | 12.7000 | 0.00000000 | 0 | 0.00000000 | perp venue volume is too low for a paper probe |
| SOPH | unlock_event_not_tradeable | do_not_probe | 10.1604 | too_illiquid 5.33 | 21 | 1210000 | 5.2000 | 0.10950000 | 27253 | 0.00294010 | perp venue volume is too low for a paper probe |
