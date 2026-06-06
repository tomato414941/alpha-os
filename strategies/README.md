# Strategies

This directory is for concrete strategy implementations that use alpha-os.

It is not package API and not a usage example directory:

- library contracts belong in `src/alpha_os/`
- usage sketches belong in `examples/`
- frozen historical research belongs in `experiments/`
- profit-seeking strategy candidates belong here

A maintained strategy should eventually include:

- a short hypothesis
- the market data it expects
- a concrete `TradingStrategy`-style boundary when that fits
- a local backtest or smoke path
- result notes for iteration decisions

Do not move shared code into `src/alpha_os/` just because one strategy uses it.
Promote code to the library only after multiple strategies need the same shape.
