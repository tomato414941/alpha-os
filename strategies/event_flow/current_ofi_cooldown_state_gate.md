# Current OFI Cooldown State Gate

This blocks immediate OFI repeats after timing decay and records what fresh state is needed. It is a paper-observation gate, not a live trading rule.

Generated at: 2026-06-08T23:03:35+00:00

| asset | lifecycle | mark | spread | depth 10bps | funding | gate | next step |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| BNB | second_repeat_decay | 604.75000000 | 0.16535895 | 105912.62032000 | 0.10243594 | cooldown_until_fresh_state | require a fresh OFI imbalance state plus a cooldown before another paper short |
| ETH | repeat_fill_audit_decay | 1699.60000000 | 0.58835643 | 11769550.71822000 | -0.07747081 | block_ofi_repeat | wait for a new independent OFI state and do not reuse the existing paper/repeat chain |
| SUI | second_repeat_decay | 0.75797000 | 0.65962626 | 70765.75498950 | 0.10950000 | cooldown_until_fresh_state | require a fresh OFI imbalance state plus a cooldown before another paper short |
