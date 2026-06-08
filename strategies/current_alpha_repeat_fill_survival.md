# Current Alpha Repeat Fill Survival

This checks the top repeat/fill-risk worklist items against first and second repeat evidence. Rows still lack real fill, stop, and adverse-excursion records, so this is not a promotion report.

| work | asset | status | score | best net | first net | second net | first outcome | second outcome | decay | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| repeat_fill_risk_probe-02-paper-cost-survival-watchlist-link-paper-long | LINK | second_repeat_pending | 729.8142 | 129.8142 | 129.8142 | 0.0000 | paper_mark_win | pending | 0.0000 | wait for the second repeat checkpoint, then rerun repeat fill survival for LINK |
| repeat_fill_risk_probe-03-paper-cost-survival-watchlist-fartcoin-paper-long | FARTCOIN | second_repeat_pending | 713.3003 | 113.3003 | 113.3003 | 0.0000 | paper_mark_win | pending | 0.0000 | wait for the second repeat checkpoint, then rerun repeat fill survival for FARTCOIN |
| repeat_fill_risk_probe-04-paper-cost-survival-watchlist-aave-paper-long | AAVE | second_repeat_pending | 697.9135 | 97.9135 | 97.9135 | 0.0000 | paper_mark_win | pending | 0.0000 | wait for the second repeat checkpoint, then rerun repeat fill survival for AAVE |
| repeat_fill_risk_probe-05-paper-cost-survival-watchlist-sei-paper-long | SEI | second_repeat_pending | 653.1748 | 53.1748 | 53.1748 | 0.0000 | paper_mark_win | pending | 0.0000 | wait for the second repeat checkpoint, then rerun repeat fill survival for SEI |
| repeat_fill_risk_probe-01-paper-cost-survival-watchlist-chip-paper-long | CHIP | repeat_edge_collapsed | 261.4043 | 346.9475 | 346.9475 | 1.4043 | paper_mark_win | paper_mark_win | 0.9960 | do not promote CHIP paper_long; require a fresh independent repeat before keeping it alive |
