# Liquidation Flow

This lane watches forced-liquidation flow as a candidate alpha source.

It is not a trading strategy. It looks for liquidation bursts and side
imbalance that may point to cascade continuation or squeeze/reversal setups.

## Commands

```bash
uv run python -m strategies.liquidation_flow.current_okx_liquidation_flow
uv run python -m strategies.liquidation_flow.current_okx_liquidation_forward_labels
uv run python -m strategies.liquidation_flow.current_okx_liquidation_monitor
uv run python -m strategies.liquidation_flow.current_okx_liquidation_monitor_forward_labels
uv run python -m strategies.liquidation_flow.current_okx_liquidation_depth_check
uv run python -m strategies.liquidation_flow.current_okx_liquidation_actionability_review
uv run python -m strategies.liquidation_flow.current_okx_liquidation_paper_gate
```
