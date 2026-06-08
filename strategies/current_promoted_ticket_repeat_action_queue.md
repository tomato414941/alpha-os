# Current Paper Ticket Action Queue

This turns paper-ticket mark outcomes into the next observation work. It is not a trade instruction and does not promote a candidate without fill, funding, stop, and repeated-label evidence.

| priority | ticket | action | asset | decision | dir bps | outcome | reason | next step |
| ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| 156.3034 | repeat-paper-09-sol-volume-dislocation | promote_to_fill_and_risk_check | SOL | paper_long | 56.30340097 | paper_mark_win | paper mark moved in the ticket direction | check fill assumption, funding, stop, adverse excursion, then repeat the label |
| 141.6000 | repeat-paper-26-sui-liquidation-intensity | promote_to_fill_and_risk_check | SUI | paper_long | 41.60000000 | paper_mark_win | paper mark moved in the ticket direction | check fill assumption, funding, stop, adverse excursion, then repeat the label |
| 141.6000 | repeat-paper-02-sui-repeat-execution | promote_to_fill_and_risk_check | SUI | paper_long | 41.60000000 | paper_mark_win | paper mark moved in the ticket direction | check fill assumption, funding, stop, adverse excursion, then repeat the label |
| 133.5731 | repeat-paper-01-sui-repeat-execution | promote_to_fill_and_risk_check | SUI | paper_long | 33.57314149 | paper_mark_win | paper mark moved in the ticket direction | check fill assumption, funding, stop, adverse excursion, then repeat the label |
| 133.5731 | repeat-paper-19-sui-microstructure-flow | promote_to_fill_and_risk_check | SUI | paper_long | 33.57314149 | paper_mark_win | paper mark moved in the ticket direction | check fill assumption, funding, stop, adverse excursion, then repeat the label |
| 50.0000 | repeat-paper-20-mon-microstructure-flow | wait_for_checkpoint | MON | paper_long |  | pending | ticket checkpoint has not matured | wait for the first checkpoint and refresh marks |
| 50.0000 | repeat-paper-50-hype-token-unlock | wait_for_checkpoint | HYPE | paper_short |  | pending | ticket checkpoint has not matured | wait for the first checkpoint and refresh marks |
| 30.0000 | repeat-paper-27-pepe-liquidation-intensity | fill_missing_observation | PEPE | paper_long |  | missing_current_mark | entry or current mark is invalid | fill missing current mark before judging the ticket |
| -156.5978 | repeat-paper-05-mega-microstructure-flow | deprioritize_or_repeat_once | MEGA | paper_long | -181.59782630 | paper_mark_loss | paper mark moved against the ticket direction | repeat only if the original hypothesis has independent support; otherwise deprioritize |
