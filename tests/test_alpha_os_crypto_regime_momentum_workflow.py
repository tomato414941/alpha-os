from __future__ import annotations

import json
from pathlib import Path


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


def _assert_common_strategy_comparison_contract(
    candidate,
    comparison_target,
    *,
    require_same_subject_set: bool = False,
) -> None:
    for task_result in (candidate, comparison_target):
        for metric_group_name, metric_name in _REQUIRED_COMPARISON_METRICS:
            _metric(task_result, metric_group_name, metric_name)

    if require_same_subject_set:
        assert candidate.universe_policy_fields == (
            comparison_target.universe_policy_fields
        )


def _strategy_document(
    *,
    strategy_id: str,
    position_rule_id: str,
    subject_set_id: str = "crypto_regime_pair",
) -> dict[str, object]:
    return {
        "trading_strategy": {
            "strategy_id": strategy_id,
            "label": strategy_id.removeprefix("strategy:"),
            "scope": {
                "subject_set_id": subject_set_id,
                "target_id": "residual_return_1d",
            },
            "signal_discovery_id": None,
            "position_rule_id": position_rule_id,
            "family_mix": None,
            "portfolio": {
                "portfolio_construction": {
                    "sizing_policy": {
                        "sizing_method": "equal_weight",
                    },
                    "direction_mode": "long_only",
                    "gross_exposure_cap": 1.0,
                    "gross_leverage_cap": 1.0,
                    "net_exposure_target": 1.0,
                },
                "rebalance_friction_policy": {
                    "turnover_friction": 0.0,
                    "no_trade_band": 0.0,
                    "execution_cost_aversion": 1.0,
                    "execution_mode": "utility_priority",
                    "turnover_budget": None,
                    "benefit_scale": 1.0,
                    "min_trade_utility": 0.0,
                    "uncertainty_aversion": 1.0,
                    "risk_aversion": 0.0,
                    "partial_fill_enabled": True,
                },
                "execution_policy": {
                    "market_impact_bps": 0.0,
                    "fee_bps": 5.0,
                    "bid_ask_spread_bps": 0.0,
                },
                "holding_cost_policy": {
                    "funding_bps_per_step": 0.0,
                    "borrow_fee_bps_per_step": 0.0,
                },
                "rebalance_interval_steps": 1,
                "selection_kind": "all_assets",
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
                position_rule_id="crypto_regime_momentum_hold",
                subject_set_id=subject_set_id,
            ),
            _strategy_document(
                strategy_id="strategy:crypto_regime_momentum_baseline",
                position_rule_id="constant_hold",
                subject_set_id=subject_set_id,
            ),
        ],
        "evaluation_specs": [
            {
                "evaluation_spec_id": evaluation_spec_id,
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
        "evaluation_tasks": [
            {
                "evaluation_task_id": "crypto_regime_momentum_candidate_case",
                "evaluation_spec_id": evaluation_spec_id,
                "strategy_id": "strategy:crypto_regime_momentum_candidate",
                "base_url": "fixture://local",
            },
            {
                "evaluation_task_id": "crypto_regime_momentum_baseline_case",
                "evaluation_spec_id": evaluation_spec_id,
                "strategy_id": "strategy:crypto_regime_momentum_baseline",
                "base_url": "fixture://local",
            },
        ],
    }


def test_crypto_regime_momentum_strategy_backtest_workflow(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "alpha-os.db"
    manifest_path = tmp_path / "crypto-regime-manifest.json"
    manifest_path.write_text(json.dumps(_manifest_document()))

    assert (
        main(
            [
                "apply-manifest",
                "--manifest",
                str(manifest_path),
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "run-walk-forward-evaluation",
                "--evaluation-spec-id",
                "crypto_regime_momentum_eval",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    store = EvaluationStore(Path(db_path))
    try:
        report_state = store.get_latest_evaluation_report()
        assert report_state is not None
        report = report_state.report
        assert report.evaluation_spec_id == "crypto_regime_momentum_eval"
        assert len(report.task_results) == 2
        task_results = {item.evaluation_task_id: item for item in report.task_results}
        candidate = task_results["crypto_regime_momentum_candidate_case"]
        baseline = task_results["crypto_regime_momentum_baseline_case"]
        assert candidate.strategy_id == "strategy:crypto_regime_momentum_candidate"
        assert baseline.strategy_id == "strategy:crypto_regime_momentum_baseline"

        candidate_strategy = store.get_trading_strategy(candidate.strategy_id)
        baseline_strategy = store.get_trading_strategy(baseline.strategy_id)
        assert candidate_strategy is not None
        assert baseline_strategy is not None
        assert (
            candidate_strategy.trading_strategy.position_rule_id
            == "crypto_regime_momentum_hold"
        )
        assert baseline_strategy.trading_strategy.position_rule_id == "constant_hold"
        assert candidate_strategy.trading_strategy.subject_set_id == "crypto_regime_pair"
        assert baseline_strategy.trading_strategy.subject_set_id == "crypto_regime_pair"
        assert (
            candidate_strategy.trading_strategy.target_id
            == baseline_strategy.trading_strategy.target_id
            == "residual_return_1d"
        )
        assert (
            candidate_strategy.trading_strategy.selection_kind
            == baseline_strategy.trading_strategy.selection_kind
            == "all_assets"
        )
        assert (
            candidate_strategy.trading_strategy.portfolio.execution_policy.fee_bps
            == baseline_strategy.trading_strategy.portfolio.execution_policy.fee_bps
            == 5.0
        )
        assert (
            candidate_strategy.trading_strategy.portfolio.holding_cost_policy.funding_bps_per_step
            == baseline_strategy.trading_strategy.portfolio.holding_cost_policy.funding_bps_per_step
            == 0.0
        )
        assert (
            candidate_strategy.trading_strategy.portfolio.holding_cost_policy.borrow_fee_bps_per_step
            == baseline_strategy.trading_strategy.portfolio.holding_cost_policy.borrow_fee_bps_per_step
            == 0.0
        )

        for task_result in (candidate, baseline):
            decision_quality = _metric_group_metrics(
                task_result,
                "decision_quality",
            )
            robustness = _metric_group_metrics(task_result, "robustness")
            _assert_numeric_metrics(
                decision_quality,
                (
                    "mean_decision_net_return",
                    "mean_decision_drawdown",
                    "annualized_step_sharpe",
                    "mean_decision_turnover",
                ),
            )
            _assert_numeric_metrics(
                robustness,
                ("worst_decision_net_return",),
            )
        _assert_common_strategy_comparison_contract(
            candidate,
            baseline,
            require_same_subject_set=True,
        )
    finally:
        store.close()


def test_crypto_regime_momentum_real_dataset_backtest_reproduces_direction(
    tmp_path,
    capsys,
):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "alpha-os-real-data.db"
    manifest_path = tmp_path / "crypto-regime-real-data-manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest_document(
                source_id=(
                    "experiments/datasets/"
                    "ds_crypto_btc_eth_daily_2024_2025/{asset}.csv"
                ),
                subjects=(
                    ("BTCUSDT_fixture", "BTCUSDT"),
                    ("ETHUSDT_fixture", "ETHUSDT"),
                ),
                evaluation_spec_id="crypto_regime_momentum_real_data_eval",
                execution_start="2024-01-01",
                execution_end="2024-03-31",
                evaluation_start="2024-04-01",
                evaluation_end="2025-12-31",
            )
        )
    )

    assert (
        main(
            [
                "apply-manifest",
                "--manifest",
                str(manifest_path),
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "run-walk-forward-evaluation",
                "--evaluation-spec-id",
                "crypto_regime_momentum_real_data_eval",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    store = EvaluationStore(Path(db_path))
    try:
        report_state = store.get_latest_evaluation_report()
        assert report_state is not None
        report = report_state.report
        task_results = {item.evaluation_task_id: item for item in report.task_results}
        candidate = task_results["crypto_regime_momentum_candidate_case"]
        baseline = task_results["crypto_regime_momentum_baseline_case"]

        candidate_mean_net = _metric(
            candidate,
            "decision_quality",
            "mean_decision_net_return",
        )
        baseline_mean_net = _metric(
            baseline,
            "decision_quality",
            "mean_decision_net_return",
        )
        candidate_worst_net = _metric(
            candidate,
            "robustness",
            "worst_decision_net_return",
        )
        baseline_worst_net = _metric(
            baseline,
            "robustness",
            "worst_decision_net_return",
        )

        assert candidate_mean_net > baseline_mean_net
        assert candidate_worst_net >= baseline_worst_net
    finally:
        store.close()


def test_common_strategy_comparison_contract_rejects_missing_required_metric():
    from alpha_os.evaluation_result import (
        EvaluationMetricGroupResult,
        EvaluationTaskResult,
    )

    candidate = EvaluationTaskResult(
        evaluation_task_id="candidate",
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
    comparison_target = EvaluationTaskResult(
        evaluation_task_id="comparison",
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
