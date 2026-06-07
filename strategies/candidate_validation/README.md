# Candidate Validation

This lane checks current candidates from other research lanes against market
context.

It does not prove alpha. It prevents candidate screens from staying detached
from realized market behavior.

## Commands

```bash
uv run python -m strategies.candidate_validation.current_hl_candidate_return_context
uv run python -m strategies.candidate_validation.current_hl_signal_forward_labels
```
