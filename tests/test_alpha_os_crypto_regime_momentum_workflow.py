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


def _assert_numeric_metrics(metrics: dict[str, object], names: tuple[str, ...]) -> None:
    for name in names:
        assert name in metrics
        assert isinstance(metrics[name], (int, float))


def _assert_common_strategy_comparison_contract(candidate, comparison_target) -> None:
    for task_result in (candidate, comparison_target):
        for metric_group_name, metric_name in _REQUIRED_COMPARISON_METRICS:
            _metric(task_result, metric_group_name, metric_name)


def _strategy_document(
    *,
    strategy_id: str,
    subject_set_id: str = "crypto_regime_pair",
) -> dict[str, object]:
    return {
        "trading_strategy": {
            "strategy_id": strategy_id,
            "label": strategy_id.removeprefix("strategy:"),
            "subject_set_id": subject_set_id,
            "target_id": "residual_return_1d",
            "portfolio_construction": {
                "sizing_policy": {
                    "sizing_method": "equal_weight",
                },
                "direction_mode": "long_only",
                "gross_exposure_cap": 1.0,
                "gross_leverage_cap": 1.0,
                "net_exposure_target": 1.0,
            },
            "created_at": "2026-05-01T00:00:00+00:00",
        }
    }


def _manifest_document(
    *,
    subject_set_id: str = "crypto_regime_pair",
    observation_spec_id: str = "crypto_regime_daily",
    source_id: str = "tests/fixtures/crypto_regime_momentum/{asset}.csv",
    subjects: tuple[tuple[str, str], ...] = (
        ("BTC_fixture", "BTC"),
        ("ETH_fixture", "ETH"),
    ),
    evaluation_spec_id: str = "crypto_regime_momentum_eval",
    execution_start: str = "2026-01-01",
    execution_end: str = "2026-01-31",
    evaluation_start: str = "2026-02-01",
    evaluation_end: str = "2026-03-02",
) -> dict[str, object]:
    return {
        "observables": [
            {
                "observable_id": "daily_close",
                "family": "price",
                "value_kind": "real_value",
                "default_resolution": "1d",
                "params": {},
            }
        ],
        "subject_sets": [
            {
                "subject_set_id": subject_set_id,
                "universe_policy": {
                    "base_currency": "USD",
                    "trading_calendar": "fixture_daily",
                    "benchmark_id": subject_set_id,
                },
                "observation_specs": [
                    {
                        "observation_spec_id": observation_spec_id,
                        "observable_id": "daily_close",
                        "adapter_kind": "fixture_csv",
                        "source_id": source_id,
                        "resolution": "1d",
                    }
                ],
                "bindings": [
                    {
                        "subject_id": subject_id,
                        "subject_kind": "crypto_perp",
                        "asset": asset,
                        "observation_spec_id": observation_spec_id,
                    }
                    for subject_id, asset in subjects
                ],
            }
        ],
        "strategy_specs": [
            _strategy_document(
                strategy_id="strategy:crypto_regime_momentum_candidate",
                subject_set_id=subject_set_id,
            ),
            _strategy_document(
                strategy_id="strategy:crypto_regime_momentum_baseline",
                subject_set_id=subject_set_id,
            ),
        ],
        "evaluation_specs": [
            {
                "evaluation_spec_id": evaluation_spec_id,
                "strategy_ids": [
                    "strategy:crypto_regime_momentum_candidate",
                    "strategy:crypto_regime_momentum_baseline",
                ],
                "execution_range": {
                    "label": "crypto_regime_train",
                    "start_date": execution_start,
                    "end_date": execution_end,
                },
                "evaluation_folds": [
                    {
                        "label": "crypto_regime_train_to_test",
                        "execution_range": {
                            "label": "crypto_regime_train",
                            "start_date": execution_start,
                            "end_date": execution_end,
                        },
                        "evaluation_date_ranges": [
                            {
                                "label": "crypto_regime_test",
                                "start_date": evaluation_start,
                                "end_date": evaluation_end,
                            }
                        ],
                    }
                ],
                "metric_group_names": [
                    "portfolio_target_return_alignment",
                    "decision_quality",
                    "portfolio_concentration",
                    "robustness",
                ],
                "metric_windows": [5],
                "rigor_level": "diagnostic",
                "oos_contract": {
                    "enforcement": "warn",
                    "require_non_overlapping_ranges": True,
                    "require_evaluation_after_execution": True,
                },
            }
        ],
    }


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
            EvaluationMetricGroupResult(
                metric_group_name="robustness",
                source="test",
                metrics={"worst_decision_net_return": 0.1},
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
            EvaluationMetricGroupResult(
                metric_group_name="robustness",
                source="test",
                metrics={"worst_decision_net_return": 0.1},
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
