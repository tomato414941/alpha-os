# Migration Map

## Purpose

This is a temporary inventory of components that may need later review.

Do not use this file to choose the next task. The next task should be selected
from an investment hypothesis, an experiment blocker, or a concrete runtime
failure.

## Current Inventory

### Observation

- [`observation_adapters.py`](../../src/alpha_os/observation_adapters.py)
- [`observables.py`](../../src/alpha_os/observables.py)
- [`signal_client.py`](../../src/alpha_os/signal_client.py)
- [`portfolio_decision.py`](../../src/alpha_os/portfolio_decision.py)
- [`store.py`](../../src/alpha_os/store.py)
- raw backend `signal_name` in public contracts

### Representation

- [`feature_plane.py`](../../src/alpha_os/feature_plane.py)
- [`evaluation_generation.py`](../../src/alpha_os/evaluation_generation.py)
- [`evaluation_inputs.py`](../../src/alpha_os/evaluation_inputs.py)
- [`evaluation_runtime.py`](../../src/alpha_os/evaluation_runtime.py)
- one-row-at-a-time evaluation persistence

### Signal Discovery

- [`signal_compiler.py`](../../src/alpha_os/signal_compiler.py)
- [`signal_registry.py`](../../src/alpha_os/signal_registry.py)
- [`targets.py`](../../src/alpha_os/targets.py)
- executable signal IDs as conceptual source of truth

### Selection

- [`screening.py`](../../src/alpha_os/screening.py)
- [`signal_discovery_screening_service.py`](../../src/alpha_os/signal_discovery_screening_service.py)
- [`pre_screening.py`](../../src/alpha_os/pre_screening.py)
- [`probe_screening.py`](../../src/alpha_os/probe_screening.py)
- [`survivor_screening.py`](../../src/alpha_os/survivor_screening.py)
- [`metrics_service.py`](../../src/alpha_os/metrics_service.py)
- [`meta_aggregation_service.py`](../../src/alpha_os/meta_aggregation_service.py)
- [`meta_metrics_service.py`](../../src/alpha_os/meta_metrics_service.py)
- [`scoring.py`](../../src/alpha_os/scoring.py)
- broad full-run validation as the first filter
- cheap pre-screen deciding final stability, novelty, or portfolio value

### Compression

- [`compression.py`](../../src/alpha_os/compression.py)
- [`signal_discovery_compression_service.py`](../../src/alpha_os/signal_discovery_compression_service.py)
- [`belief_synthesis.py`](../../src/alpha_os/belief_synthesis.py)
- [`meta_aggregation_service.py`](../../src/alpha_os/meta_aggregation_service.py)
- [`portfolio_decision_inputs.py`](../../src/alpha_os/portfolio_decision_inputs.py)
- direct portfolio use of large raw signal populations

### Decision

- [`portfolio_decision.py`](../../src/alpha_os/portfolio_decision.py)
- [`portfolio_decision_inputs.py`](../../src/alpha_os/portfolio_decision_inputs.py)
- [`portfolio_execution_policy.py`](../../src/alpha_os/portfolio_execution_policy.py)
- [`portfolio_decision_service.py`](../../src/alpha_os/portfolio_decision_service.py)
- [`decision_backtest.py`](../../src/alpha_os/decision_backtest.py)
- [`portfolio_sizing_policy.py`](../../src/alpha_os/portfolio_sizing_policy.py)
- uncertainty inputs
- decision layer as a substitute for missing upstream screening or compression

### Evaluation

- [`validation_engine.py`](../../src/alpha_os/validation_engine.py)
- [`validation_service.py`](../../src/alpha_os/validation_service.py)
- [`validation_spec.py`](../../src/alpha_os/validation_spec.py)
- winner-picking interpretations of validation output

### Control Surface

- [`cli/`](../../src/alpha_os/cli/)
- [`cli_output.py`](../../src/alpha_os/cli_output.py)
- [`config.py`](../../src/alpha_os/config.py)
- [`store.py`](../../src/alpha_os/store.py)
- debug commands drifting into public runtime truth
