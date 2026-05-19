from __future__ import annotations


def test_strategy_evaluation_request_carries_evaluation_execution_inputs():
    from alpha_os.evaluation_spec import EvaluationDateRange
    from alpha_os.strategy_engine import (
        StrategyEvaluationContext,
        StrategyEvaluationRequest,
    )

    execution_range = EvaluationDateRange(
        label="train",
        start_date="2025-01-01",
        end_date="2025-12-31",
    )
    evaluation_date_ranges = (
        EvaluationDateRange(
            label="test",
            start_date="2026-01-01",
            end_date="2026-03-31",
        ),
    )
    request = StrategyEvaluationRequest(
        evaluation_task_id="case:test",
        evaluation_spec_id="protocol:test",
        fold_label="fold_2025",
        context=StrategyEvaluationContext(
            strategy_id="strategy:test",
            target_id="target:test",
            base_url="http://example.com",
        ),
        execution_range=execution_range,
        evaluation_date_ranges=evaluation_date_ranges,
        metric_group_names=("decision_quality",),
    )

    assert request.evaluation_task_id == "case:test"
    assert request.evaluation_spec_id == "protocol:test"
    assert request.fold_label == "fold_2025"
    assert request.context.strategy_id == "strategy:test"
    assert request.evaluation_date_ranges[0].label == "test"

def test_evaluation_spec_reads_legacy_dimensions_but_writes_metric_group_names():
    from alpha_os.evaluation_metric_config import EvaluationMetricConfig
    from alpha_os.evaluation_spec import EvaluationSpec

    protocol = EvaluationSpec.from_document(
        {
            "execution_range": {
                "label": "eval",
                "start_date": "2026-01-01",
                "end_date": "2026-01-31",
            },
            "dimensions": ["decision_quality"],
            "metric_windows": [20],
        }
    )

    document = protocol.to_document()

    assert protocol.metric_group_names == ("decision_quality",)
    assert protocol.metric_config == EvaluationMetricConfig(
        metric_group_names=("decision_quality",),
        metric_windows=(20,),
    )
    assert document["metric_group_names"] == ["decision_quality"]
    assert "dimensions" not in document


def test_evaluation_spec_rejects_mixed_metric_group_keys():
    import pytest

    from alpha_os.evaluation_spec import EvaluationSpec

    with pytest.raises(ValueError, match="both metric_group_names and legacy dimensions"):
        EvaluationSpec.from_document(
            {
                "execution_range": {
                    "label": "eval",
                    "start_date": "2026-01-01",
                    "end_date": "2026-01-31",
                },
                "metric_group_names": ["decision_quality"],
                "dimensions": ["robustness"],
                "metric_windows": [20],
            }
        )


def test_requires_decision_evaluation_classifies_metric_group_names():
    from alpha_os.evaluation_metric_config import (
        DECISION_EVALUATION_METRIC_GROUP_NAMES,
        requires_decision_evaluation,
    )

    assert requires_decision_evaluation(("decision_quality",))
    assert requires_decision_evaluation(("robustness",))
    assert not requires_decision_evaluation(())
    assert "decision_quality" in DECISION_EVALUATION_METRIC_GROUP_NAMES
