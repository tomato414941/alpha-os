# Architecture

## Producer-Consumer Separation

The long-horizon target architecture should separate production from
consumption.

Producers should generate predictions without knowing anything about runtime
selection, allocation, or execution.

Consumers should evaluate predictions and make runtime decisions without
depending on producer internals.

```text
producer -> prediction store -> consumer
```

This makes it possible to add:

- DSL producers
- classical indicator producers
- ML producers
- external prediction ingests

without changing the evaluation pipeline each time.

## Prediction Store

The shared contract between producers and the evaluation pipeline should be a
simple prediction record.

Metadata such as model path, expression source, or producer type may exist for
auditability, but evaluation should depend on outputs, not implementation
internals.

## Diversity

Diversity should come from two places:

- upstream search breadth
- downstream marginal usefulness

Structural diversity may come from:

- horizons
- targets
- assets
- generator types

The runtime should not need brittle semantic duplicate detection in order to
benefit from diversity.

## Pipeline Validation

The pipeline itself must be validated, not just individual hypotheses.

Useful validation questions include:

- does the runtime behave sensibly when many hypotheses are present
- do selection and metrics behave coherently over time
- does portfolio construction improve over a naive equal-weight basket

This is mechanism validation rather than indicator-level optimization.
