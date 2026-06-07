# Candidate Validation

This lane checks current candidates from other research lanes against market
context.

It does not prove alpha. It prevents candidate screens from staying detached
from realized market behavior.

## Commands

```bash
uv run python -m strategies.candidate_validation.current_hl_candidate_return_context
uv run python -m strategies.candidate_validation.current_hl_signal_forward_labels
uv run python -m strategies.candidate_validation.current_cross_lane_candidate_review
uv run python -m strategies.candidate_validation.current_signal_family_review
uv run python -m strategies.candidate_validation.current_source_conflict_review
uv run python -m strategies.candidate_validation.current_followup_queue
uv run python -m strategies.candidate_validation.current_followup_execution_context
uv run python -m strategies.candidate_validation.current_followup_repeat_observations
uv run python -m strategies.candidate_validation.current_followup_repeat_forward_labels
uv run python -m strategies.candidate_validation.current_followup_venue_coverage
uv run python -m strategies.candidate_validation.current_followup_okx_execution_context
uv run python -m strategies.candidate_validation.current_followup_okx_repeat_observations
uv run python -m strategies.candidate_validation.current_followup_okx_repeat_forward_labels
uv run python -m strategies.candidate_validation.current_followup_repeat_history
uv run python -m strategies.candidate_validation.current_followup_repeat_history_labels
uv run python -m strategies.candidate_validation.current_followup_repeat_history_summary
```
