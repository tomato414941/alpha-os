# Anomaly Stress

This lane looks for current cross-market anomaly states across existing probes.

It does not claim that an anomaly is a trade. It ranks unusual states that
deserve a falsification test: peg stress, lending stress, yield/peg conflicts,
cheap volatility quotes, prediction-market probability gaps, and execution
spread dislocations.

## Commands

```bash
uv run python -m strategies.anomaly_stress.current_cross_market_stress_anomaly
```
