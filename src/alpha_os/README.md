# alpha_os

`alpha_os` is intentionally small right now.

The maintained package responsibility is to define the common trading strategy
contract:

```text
TradingStrategy.decide(input) -> output
```

A trading strategy is treated as a black-box decision component. The package
does not standardize the strategy's internal structure, market data shape,
backtest loop, portfolio allocator, execution model, eligibility rule, or
diagnostic report.

Repository roles:

- `src/alpha_os/`: maintained package contracts
- `examples/`: possible strategy shapes, not public API

Do not promote strategy-local code into `alpha_os` just because it is useful
once. Promote only when the shared shape is clear enough to be a package
contract rather than one strategy's implementation detail.
