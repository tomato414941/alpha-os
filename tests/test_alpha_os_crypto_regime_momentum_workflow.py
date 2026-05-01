from __future__ import annotations

import json
from pathlib import Path


def _metric_group_metrics(task_result, metric_group_name: str) -> dict[str, object]:
    for group in task_result.metric_group_results:
        if group.metric_group_name == metric_group_name:
            return group.metrics
    raise AssertionError(f"missing metric group: {metric_group_name}")


def _assert_numeric_metrics(metrics: dict[str, object], names: tuple[str, ...]) -> None:
    for name in names:
        assert name in metrics
        assert isinstance(metrics[name], (int, float))


def _strategy_document(*, strategy_id: str, signal_kind: str) -> dict[str, object]:
    return {
        "trading_strategy": {
            "strategy_id": strategy_id,
            "label": strategy_id.removeprefix("strategy:"),
            "scope": {
                "subject_set_id": "crypto_regime_pair",
                "target_id": "residual_return_1d",
            },
            "signal_policy": {
                "definition_policy": {
                    "signal_discovery_id": None,
                    "signal_kind": signal_kind,
                    "family_mix": None,
                },
                "update_policy": {
                    "execution_kind": "trainless",
                },
            },
            "portfolio_policy": {
                "selection_policy": {
                    "selection_kind": "all_assets",
                    "top_k": None,
                },
                "sizing_policy": {
                    "sizing_method": "equal_weight",
                },
                "rebalance_policy": {
                    "rebalance": "every_1_steps",
                    "rebalance_interval_steps": 1,
                },
                "risk_policy": {
                    "long_only": True,
                    "gross_exposure_cap": 1.0,
                    "gross_leverage_cap": 1.0,
                    "net_exposure_target": 1.0,
                },
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
            "created_at": "2026-05-01T00:00:00+00:00",
        }
    }


def _manifest_document() -> dict[str, object]:
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
                "subject_set_id": "crypto_regime_pair",
                "universe_policy": {
                    "base_currency": "USD",
                    "trading_calendar": "fixture_daily",
                    "benchmark_id": "crypto_regime_pair",
                },
                "observation_specs": [
                    {
                        "observation_spec_id": "crypto_regime_daily",
                        "observable_id": "daily_close",
                        "adapter_kind": "fixture_csv",
                        "source_id": (
                            "tests/fixtures/crypto_regime_momentum/{asset}.csv"
                        ),
                        "resolution": "1d",
                    }
                ],
                "bindings": [
                    {
                        "subject_id": "BTC_fixture",
                        "subject_kind": "crypto_perp",
                        "asset": "BTC",
                        "observation_spec_id": "crypto_regime_daily",
                    },
                    {
                        "subject_id": "ETH_fixture",
                        "subject_kind": "crypto_perp",
                        "asset": "ETH",
                        "observation_spec_id": "crypto_regime_daily",
                    },
                ],
            }
        ],
        "strategy_specs": [
            _strategy_document(
                strategy_id="strategy:crypto_regime_momentum_candidate",
                signal_kind="crypto_regime_momentum_hold",
            ),
            _strategy_document(
                strategy_id="strategy:crypto_regime_momentum_baseline",
                signal_kind="constant_hold",
            ),
        ],
        "evaluation_specs": [
            {
                "evaluation_spec_id": "crypto_regime_momentum_eval",
                "execution_range": {
                    "label": "crypto_regime_train",
                    "start_date": "2026-01-01",
                    "end_date": "2026-01-31",
                },
                "evaluation_folds": [
                    {
                        "label": "crypto_regime_train_to_test",
                        "execution_range": {
                            "label": "crypto_regime_train",
                            "start_date": "2026-01-01",
                            "end_date": "2026-01-31",
                        },
                        "evaluation_date_ranges": [
                            {
                                "label": "crypto_regime_test",
                                "start_date": "2026-02-01",
                                "end_date": "2026-03-02",
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
                    "require_frozen_state_for_trained_strategy": False,
                },
            }
        ],
        "evaluation_tasks": [
            {
                "evaluation_task_id": "crypto_regime_momentum_candidate_case",
                "evaluation_spec_id": "crypto_regime_momentum_eval",
                "strategy_id": "strategy:crypto_regime_momentum_candidate",
                "base_url": "fixture://local",
            },
            {
                "evaluation_task_id": "crypto_regime_momentum_baseline_case",
                "evaluation_spec_id": "crypto_regime_momentum_eval",
                "strategy_id": "strategy:crypto_regime_momentum_baseline",
                "base_url": "fixture://local",
            },
        ],
    }


def test_crypto_regime_momentum_candidate_backtest_workflow(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "alpha-os.db"
    manifest_path = tmp_path / "crypto-regime-manifest.json"
    manifest_path.write_text(json.dumps(_manifest_document()))

    assert (
        main(
            [
                "apply-runtime-manifest",
                "--manifest",
                str(manifest_path),
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    apply_output = capsys.readouterr().out
    assert "crypto_regime_momentum_eval" in apply_output

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
    run_output = capsys.readouterr().out
    assert "alpha-os evaluation report" in run_output

    assert main(["show-evaluation-report", "--db", str(db_path)]) == 0
    report_output = capsys.readouterr().out
    assert "TaskResults: 2" in report_output
    assert "strategy:crypto_regime_momentum_candidate" in report_output
    assert "subject_set=crypto_regime_pair" in report_output
    assert "target_id=residual_return_1d" in report_output

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
        assert candidate.strategy_contract_fields["target_id"] == "residual_return_1d"
        assert baseline.strategy_contract_fields["target_id"] == "residual_return_1d"

        candidate_strategy = store.get_trading_strategy(candidate.strategy_id)
        baseline_strategy = store.get_trading_strategy(baseline.strategy_id)
        assert candidate_strategy is not None
        assert baseline_strategy is not None
        assert (
            candidate_strategy.trading_strategy.signal_kind
            == "crypto_regime_momentum_hold"
        )
        assert baseline_strategy.trading_strategy.signal_kind == "constant_hold"
        assert candidate_strategy.trading_strategy.subject_set_id == "crypto_regime_pair"
        assert baseline_strategy.trading_strategy.subject_set_id == "crypto_regime_pair"

        shared_contract_keys = (
            "target_id",
            "selection",
            "sizing",
            "rebalance",
            "long_only",
            "fee_bps",
            "funding_bps_per_step",
            "borrow_fee_bps_per_step",
        )
        for key in shared_contract_keys:
            assert candidate.strategy_contract_fields[key] == (
                baseline.strategy_contract_fields[key]
            )

        assert candidate.subject_set_facts == baseline.subject_set_facts
        assert candidate.universe_policy_fields == baseline.universe_policy_fields

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
                    "mean_decision_turnover",
                ),
            )
            _assert_numeric_metrics(
                robustness,
                ("worst_decision_net_return",),
            )
            trace_steps = store.list_evaluation_decision_trace_steps(
                evaluation_report_id=report.evaluation_report_id,
                evaluation_task_id=task_result.evaluation_task_id,
                evaluation_fold_label="crypto_regime_train_to_test",
                evaluation_range_label="crypto_regime_test",
                limit=1000,
            )
            assert trace_steps
            assert {item.target_id for item in trace_steps} == {"residual_return_1d"}
            assert {item.subject_set_id for item in trace_steps} == {
                "crypto_regime_pair"
            }
            assert min(item.step_as_of for item in trace_steps) >= "2026-02-01"
            assert max(item.step_as_of for item in trace_steps) <= "2026-03-02"
    finally:
        store.close()
