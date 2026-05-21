from __future__ import annotations


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
