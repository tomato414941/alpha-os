# Current Event Probability Execution Queue

This queue tracks pure prediction-market probability probes that need quote refresh, fill/queue notes, max-loss handling, and resolution-risk records. It is not a live order list.

| queue | market | side | ask | edge | depth 5c | action | checkpoints | next step |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| event-probability-execution-1962237-yes | US x Iran permanent peace deal by June 30, 2026? | Yes | 0.170000 | 0.170000 | 74427.620000 | paper_check_pure_probability | 15m,1h,4h | paper-check US x Iran permanent peace deal by June 30, 2026? as a pure event-probability trade with max-loss and resolution-risk notes |
| event-probability-execution-947269-no | Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | 0.120000 | 0.140000 | 208484.050000 | paper_check_pure_probability | 15m,1h,4h | paper-check Will Keiko Fujimori win the 2026 Peruvian presidential election? as a pure event-probability trade with max-loss and resolution-risk notes |
| event-probability-execution-2270330-yes | US x Iran permanent peace deal by June 15, 2026? | Yes | 0.060000 | 0.180000 | 273946.710000 | restart_quote_survival_probe | 15m,1h,4h | restart paper ticket for US x Iran permanent peace deal by June 15, 2026? and require quote refresh survival |
| event-probability-execution-1971905-yes | Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.100000 | 0.180000 | 181416.570000 | restart_quote_survival_probe | 15m,1h,4h | restart paper ticket for Strait of Hormuz traffic returns to normal by end of June? and require quote refresh survival |
