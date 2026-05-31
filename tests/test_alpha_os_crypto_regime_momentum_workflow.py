from __future__ import annotations


_REQUIRED_COMPARISON_METRICS = (
    ("decision_quality", "mean_decision_net_return"),
    ("decision_quality", "mean_decision_drawdown"),
    ("decision_quality", "annualized_step_sharpe"),
    ("decision_quality", "mean_decision_turnover"),
)


def _metric_group_metrics(task_result, metric_group_name: str) -> dict[str, object]:
    for group in task_result.metric_group_results:
        if group.metric_group_name == metric_group_name:
            return group.metrics
    raise AssertionError(f"missing metric group: {metric_group_name}")


def _metric(task_result, metric_group_name: str, metric_name: str) -> float:
    metrics = _metric_group_metrics(task_result, metric_group_name)
    assert metric_name in metrics
    assert isinstance(metrics[metric_name], (int, float))
    return float(metrics[metric_name])


def _assert_common_strategy_comparison_contract(candidate, comparison_target) -> None:
    for task_result in (candidate, comparison_target):
        for metric_group_name, metric_name in _REQUIRED_COMPARISON_METRICS:
            _metric(task_result, metric_group_name, metric_name)


def test_common_strategy_comparison_contract_rejects_missing_required_metric():
    from alpha_os.evaluation_result import (
        EvaluationMetricGroupResult,
        EvaluationResult,
    )

    candidate = EvaluationResult(
        strategy_id="strategy:candidate",
        metric_group_results=(
            EvaluationMetricGroupResult(
                metric_group_name="decision_quality",
                source="test",
                metrics={
                    "mean_decision_net_return": 0.1,
                    "mean_decision_drawdown": 0.02,
                    "mean_decision_turnover": 0.3,
                },
            ),
        ),
    )
    comparison_target = EvaluationResult(
        strategy_id="strategy:comparison",
        metric_group_results=(
            EvaluationMetricGroupResult(
                metric_group_name="decision_quality",
                source="test",
                metrics={
                    "mean_decision_net_return": 0.1,
                    "mean_decision_drawdown": 0.02,
                    "annualized_step_sharpe": 1.0,
                    "mean_decision_turnover": 0.3,
                },
            ),
        ),
    )
    try:
        _assert_common_strategy_comparison_contract(
            candidate,
            comparison_target,
        )
    except AssertionError:
        return
    raise AssertionError("comparison contract accepted a missing required metric")
