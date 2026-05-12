from __future__ import annotations


def _evaluation_policy_parts(
    *,
    sizing_method: str = "signal_weighted",
    sizing_engine: str | None = None,
):
    from alpha_os.evaluation_cost_config import (
        EvaluationRebalanceFrictionPolicySpec,
        ExecutionCostAssumptionsSpec,
        HoldingCostAssumptionsSpec,
    )
    from alpha_os.portfolio_construction_config import (
        PortfolioConstructionSizingSpec,
        PortfolioConstructionSpec,
    )

    return {
        "portfolio_construction": PortfolioConstructionSpec(
            sizing_policy=PortfolioConstructionSizingSpec(
                sizing_method=sizing_method,
                sizing_engine=sizing_engine,
            )
        ),
        "rebalance_friction_policy": EvaluationRebalanceFrictionPolicySpec(),
        "execution_cost_assumptions": ExecutionCostAssumptionsSpec(),
        "holding_cost_assumptions": HoldingCostAssumptionsSpec(),
    }


def test_strategy_evaluation_request_carries_evaluation_execution_inputs():
    from alpha_os.evaluation_spec import EvaluationDateRange
    from alpha_os.strategy_engine import (
        StrategyEvaluationDiagnosticRefs,
        StrategyEvaluationContext,
        StrategyEvaluationInputRefs,
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
    policy_parts = _evaluation_policy_parts(
        sizing_method="equal_weight",
        sizing_engine="history_based",
    )

    request = StrategyEvaluationRequest(
        evaluation_task_id="case:test",
        evaluation_spec_id="protocol:test",
        fold_label="fold_2025",
        context=StrategyEvaluationContext(
            strategy_id="strategy:test",
            signal_discovery_id="discovery:test",
            subject_set_id="subject-set:test",
            target_id="target:test",
            base_url="http://example.com",
            selection_kind="all_assets",
            top_k=None,
            portfolio_construction=policy_parts["portfolio_construction"],
            rebalance_friction_policy=policy_parts["rebalance_friction_policy"],
            execution_cost_assumptions=policy_parts["execution_cost_assumptions"],
            holding_cost_assumptions=policy_parts["holding_cost_assumptions"],
        ),
        input_refs=StrategyEvaluationInputRefs(
            initial_strategy_state_id="state:test",
            snapshot_set_id="snapshot-set:test",
            screening_result_id="screening:test",
            compressed_belief_id="belief:test",
            prepared_start_date="2025-01-01",
            prepared_end_date="2025-12-31",
        ),
        diagnostic_refs=StrategyEvaluationDiagnosticRefs(
            signal_discovery_run_id="signal-discovery-run:test",
        ),
        execution_range=execution_range,
        evaluation_date_ranges=evaluation_date_ranges,
        metric_group_names=("decision_quality",),
    )

    assert request.evaluation_task_id == "case:test"
    assert request.evaluation_spec_id == "protocol:test"
    assert request.fold_label == "fold_2025"
    assert request.context.strategy_id == "strategy:test"
    assert request.context.signal_discovery_id == "discovery:test"
    assert request.context.subject_set_id == "subject-set:test"
    assert request.input_refs.initial_strategy_state_id == "state:test"
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
        PREPARATION_STATS_METRIC_GROUP_NAMES,
        requires_decision_evaluation,
    )

    mixed_metric_group_names = (
        "signal_discovery_run_stats",
        "robustness",
        "signal_discovery_result_stats",
    )
    assert requires_decision_evaluation(("decision_quality",))
    assert requires_decision_evaluation(mixed_metric_group_names)
    assert not requires_decision_evaluation(
        ("signal_discovery_run_stats", "signal_discovery_result_stats")
    )
    assert PREPARATION_STATS_METRIC_GROUP_NAMES == (
        "signal_discovery_run_stats",
        "signal_discovery_result_stats",
    )
    assert set(DECISION_EVALUATION_METRIC_GROUP_NAMES).isdisjoint(
        PREPARATION_STATS_METRIC_GROUP_NAMES
    )


def test_evaluation_job_spec_defaults_run_mode_to_backtest_oos():
    from alpha_os.evaluation_job_spec import EvaluationJobSpec

    job_spec = EvaluationJobSpec(evaluation_task_id="case:test")

    assert job_spec.run_mode == "backtest_oos"


def test_fixed_state_replay_requires_fixed_initial_state():
    import pytest

    from alpha_os.evaluation_job_spec import EvaluationJobSpec

    with pytest.raises(
        ValueError,
        match="fixed_state_replay evaluation job requires fixed_initial_strategy_state_id",
    ):
        EvaluationJobSpec(
            evaluation_task_id="case:test",
            run_mode="fixed_state_replay",
        )


def test_backtest_oos_rejects_fixed_initial_state():
    import pytest

    from alpha_os.evaluation_job_spec import EvaluationJobSpec

    with pytest.raises(
        ValueError,
        match="only fixed_state_replay evaluation jobs may define fixed_initial_strategy_state_id",
    ):
        EvaluationJobSpec(
            evaluation_task_id="case:test",
            run_mode="backtest_oos",
            fixed_initial_strategy_state_id="state:test",
        )
