from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _evaluation_policy_parts(
    *,
    sizing_method: str = "signal_weighted",
    sizing_engine: str | None = None,
    rebalance_interval_steps: int = 1,
    long_only: bool = False,
    top_k: int | None = None,
    gross_exposure_cap: float | None = None,
    target_vol: float | None = None,
    gross_leverage_cap: float | None = None,
    net_exposure_target: float | None = None,
):
    return {}


def _strategy_portfolio_document(
    *,
    sizing_method: str,
    direction_mode: str | None,
    gross_exposure_cap: float | None,
    selection_kind: str = "all_assets",
    top_k: int | None = None,
    rebalance_interval_steps: int = 1,
    rebalance_friction_policy: dict[str, object] | None = None,
    execution_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    portfolio: dict[str, object] = {
        "portfolio_construction": {
            "sizing_policy": {"sizing_method": sizing_method},
            "direction_mode": direction_mode,
            "gross_exposure_cap": gross_exposure_cap,
        },
        "rebalance_friction_policy": (
            {} if rebalance_friction_policy is None else rebalance_friction_policy
        ),
        "execution_policy": {} if execution_policy is None else execution_policy,
        "rebalance_interval_steps": rebalance_interval_steps,
        "selection_kind": selection_kind,
    }
    if top_k is not None:
        portfolio["top_k"] = top_k
    return {"portfolio": portfolio}


def _build_trading_strategy(
    *,
    strategy_id: str,
    label: str,
    subject_set_id: str | None = None,
    target_id: str | None = None,
    signal_discovery_id: str | None = None,
    position_rule_id: str = "constant_hold",
    family_mix: str | None = None,
    sizing_method: str | None = None,
    rebalance: str | None = None,
    long_only: bool | None = None,
    top_k: int | None = None,
    gross_exposure_cap: float | None = None,
    market_impact_bps: float | None = None,
    fee_bps: float | None = None,
    bid_ask_spread_bps: float | None = None,
    turnover_friction: float | None = None,
    no_trade_band: float | None = None,
    created_at: str = "2026-04-05T00:00:00Z",
):
    from alpha_os.trading_strategy import (
        ExecutionPolicySpec,
        StrategyPortfolioSpec,
        TradingStrategyScopeSpec,
        TradingStrategySpec,
        RebalanceFrictionPolicySpec,
        HoldingCostPolicySpec,
    )
    from alpha_os.portfolio_construction_config import (
        PortfolioConstructionSizingSpec,
        PortfolioConstructionSpec,
    )

    return TradingStrategySpec(
        strategy_id=strategy_id,
        label=label,
        scope=TradingStrategyScopeSpec(
            subject_set_id=subject_set_id,
            target_id=target_id,
        ),
        signal_discovery_id=signal_discovery_id,
        position_rule_id=position_rule_id,
        family_mix=family_mix,
        portfolio=StrategyPortfolioSpec(
            portfolio_construction=PortfolioConstructionSpec(
                sizing_policy=PortfolioConstructionSizingSpec(
                    sizing_method=sizing_method or "equal_weight",
                ),
                direction_mode=(
                    "long_only"
                    if long_only is True
                    else "long_short"
                    if long_only is False
                    else None
                ),
                gross_exposure_cap=gross_exposure_cap,
            ),
            rebalance_friction_policy=RebalanceFrictionPolicySpec(
                turnover_friction=turnover_friction,
                no_trade_band=no_trade_band,
            ),
            execution_policy=ExecutionPolicySpec(
                market_impact_bps=market_impact_bps,
                fee_bps=fee_bps,
                bid_ask_spread_bps=bid_ask_spread_bps,
            ),
            holding_cost_policy=HoldingCostPolicySpec(),
            rebalance_interval_steps=(
                int(rebalance[len("every_") : -len("_steps")])
                if isinstance(rebalance, str)
                and rebalance.startswith("every_")
                and rebalance.endswith("_steps")
                else 1
            ),
            selection_kind="all_assets",
            top_k=top_k,
        ),
        created_at=created_at,
    )


def test_evaluation_trading_config_from_args_accepts_direction_mode():
    from alpha_os.cli import _evaluation_trading_config_from_args

    config = _evaluation_trading_config_from_args(
        Namespace(
            sizing_method="signal_weighted",
            sizing_engine=None,
            rebalance_step=None,
            long_only=False,
            direction_mode="short_only",
            top_k=None,
            gross_exposure_cap=None,
            turnover_friction=None,
            no_trade_band=None,
            market_impact_bps=None,
            fee_bps=None,
            bid_ask_spread_bps=None,
            funding_bps_per_step=None,
            borrow_fee_bps_per_step=None,
        )
    )

    assert config.portfolio_construction.direction_mode == "short_only"
    assert config.portfolio_construction.long_only is False


def test_evaluation_task_manifest_strategy_override_takes_precedence():
    from alpha_os.cli import _evaluation_strategy_override_from_document

    config, has_override = _evaluation_strategy_override_from_document(
        {
            "portfolio_construction": {
                "sizing_policy": {
                    "sizing_method": "equal_weight",
                    "sizing_engine": "history_based",
                }
            },
            "strategy_override": {
                "portfolio_construction": {
                    "sizing_policy": {
                        "sizing_method": "signed_mean_variance",
                        "sizing_engine": "optimizer",
                    }
                },
                "rebalance_friction_policy": {
                    "turnover_budget": 0.25,
                },
            },
        }
    )

    assert has_override is True
    assert config.portfolio_construction.sizing_method == "signed_mean_variance"
    assert config.portfolio_construction.sizing_engine == "optimizer"
    assert config.rebalance_friction_policy.turnover_budget == 0.25


def test_evaluation_task_manifest_legacy_trading_config_still_loads():
    from alpha_os.cli import _evaluation_strategy_override_from_document

    config, has_override = _evaluation_strategy_override_from_document(
        {
            "portfolio_construction": {
                "sizing_policy": {
                    "sizing_method": "equal_weight",
                    "sizing_engine": "history_based",
                }
            }
        }
    )

    assert has_override is True
    assert config.portfolio_construction.sizing_method == "equal_weight"
    assert config.portfolio_construction.sizing_engine == "history_based"


def test_dual_momentum_signal_lags_trailing_returns_to_avoid_lookahead():
    from alpha_os.strategy_backtest import dual_momentum_signal_series_by_subject

    signals = dual_momentum_signal_series_by_subject(
        subject_return_series_by_subject={
            "AAA": pd.Series(
                {
                    "2026-01-01": 0.10,
                    "2026-01-02": 0.10,
                    "2026-01-03": -0.50,
                    "2026-01-04": 0.10,
                },
                dtype=float,
            )
        },
        family_mix="lookback=2",
    )

    signal = signals["AAA"]

    assert signal.loc["2026-01-01"] == pytest.approx(0.0)
    assert signal.loc["2026-01-02"] == pytest.approx(0.0)
    assert signal.loc["2026-01-03"] == pytest.approx(0.21)
    assert signal.loc["2026-01-04"] == pytest.approx(0.0)


def test_crypto_regime_momentum_eligibility_requires_trend_confirmation_and_funding_filter():
    from alpha_os.position_rules import (
        crypto_regime_momentum_eligibility_series_by_subject,
    )

    index = pd.date_range("2026-01-01", periods=66, freq="D").strftime("%Y-%m-%d")
    returns = pd.Series(0.01, index=index, dtype=float)
    returns.loc["2026-02-05":"2026-02-10"] = -0.01
    funding_rate = pd.Series(0.001, index=index, dtype=float)
    funding_rate.loc["2026-03-06"] = 0.01

    signals = crypto_regime_momentum_eligibility_series_by_subject(
        subject_return_series_by_subject={"BTC": returns},
        funding_rate_series_by_subject={"BTC": funding_rate},
    )

    signal = signals["BTC"]

    assert signal.loc["2026-01-29"] == pytest.approx(0.0)
    assert signal.loc["2026-01-30"] == pytest.approx(1.0)
    assert signal.loc["2026-02-10"] == pytest.approx(0.0)
    assert signal.loc["2026-03-05"] == pytest.approx(1.0)
    assert signal.loc["2026-03-06"] == pytest.approx(0.0)


def test_crypto_regime_momentum_eligibility_matches_experiment_reference():
    from alpha_os.position_rules import (
        crypto_regime_momentum_eligibility_series_by_subject,
    )

    fixture_dir = Path(__file__).parent / "fixtures" / "crypto_regime_momentum"
    returns_by_subject: dict[str, pd.Series] = {}
    funding_by_subject: dict[str, pd.Series] = {}
    expected_by_subject: dict[str, pd.Series] = {}
    for subject_id in ("BTC", "ETH"):
        frame = pd.read_csv(
            fixture_dir / f"{subject_id}.csv",
            parse_dates=["timestamp"],
        ).sort_values("timestamp")
        frame = frame.set_index("timestamp")
        frame.index = frame.index.tz_convert(None)
        close = frame["close"].astype(float)
        frame["return_7d"] = close / close.shift(7) - 1.0
        frame["return_30d"] = close / close.shift(30) - 1.0
        frame["funding_60d_median"] = frame["funding_rate"].rolling(60).median()
        funding_overheated = (frame["funding_rate"] > 0.0) & (
            frame["funding_rate"] > frame["funding_60d_median"]
        )
        expected_by_subject[subject_id] = (
            (
                (frame["return_7d"] > 0.0)
                & (frame["return_30d"] > 0.0)
                & ~funding_overheated
            )
            .fillna(False)
            .astype(float)
        )
        returns_by_subject[subject_id] = close.pct_change().dropna()
        funding_by_subject[subject_id] = frame["funding_rate"].astype(float)

    actual_by_subject = crypto_regime_momentum_eligibility_series_by_subject(
        subject_return_series_by_subject=returns_by_subject,
        funding_rate_series_by_subject=funding_by_subject,
    )

    for subject_id, actual in actual_by_subject.items():
        expected = expected_by_subject[subject_id].reindex(actual.index)
        pd.testing.assert_series_equal(
            actual,
            expected,
            check_names=False,
        )


def test_crypto_regime_momentum_eligibility_requires_funding_rate():
    from alpha_os.position_rules import (
        crypto_regime_momentum_eligibility_series_by_subject,
    )

    with pytest.raises(
        ValueError,
        match="crypto regime momentum requires funding_rate series: BTC",
    ):
        crypto_regime_momentum_eligibility_series_by_subject(
            subject_return_series_by_subject={
                "BTC": pd.Series(
                    {"2026-01-01": 0.01},
                    dtype=float,
                )
            },
            funding_rate_series_by_subject={},
        )


def test_direct_strategy_backtest_routes_crypto_regime_momentum_eligibility(
    monkeypatch,
):
    import alpha_os.strategy_backtest as strategy_backtest
    from alpha_os.evaluation_cost_config import (
        EvaluationRebalanceFrictionPolicySpec,
        ExecutionCostAssumptionsSpec,
        HoldingCostAssumptionsSpec,
    )
    from alpha_os.evaluation_spec import EvaluationDateRange
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
    )
    from alpha_os.subject_set_feature_plane import SubjectPlaneKey

    strategy = _build_trading_strategy(
        strategy_id="strategy:crypto_regime_momentum",
        label="Crypto regime momentum",
        subject_set_id="crypto",
        target_id="residual_return_1d",
        position_rule_id="crypto_regime_momentum_hold",
        long_only=True,
    )
    subject_set = SubjectSet(
        subject_set_id="crypto",
        observation_specs=(
            ObservationSpec(
                observation_spec_id="btc_daily",
                observable_id="daily_close",
            ),
        ),
        bindings=(
            SubjectObservationBinding(
                subject_id="BTC",
                asset="BTC",
                observation_spec_id="btc_daily",
            ),
        ),
    )
    index = pd.date_range("2026-01-01", periods=61, freq="D").strftime("%Y-%m-%d")
    returns = pd.Series(0.01, index=index, dtype=float)
    funding_rate = pd.Series(0.001, index=index, dtype=float)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        strategy_backtest,
        "build_subject_set_feature_planes",
        lambda **_: {
            SubjectPlaneKey(asset="BTC", observation_spec_id="btc_daily"): SimpleNamespace(
                daily_returns=returns,
                extra_observables={"funding_rate": funding_rate},
            )
        },
    )

    def capture_metric_group_results(**kwargs):
        captured.update(kwargs)
        return ((), ())

    monkeypatch.setattr(
        strategy_backtest,
        "build_direct_strategy_evaluation_metric_group_results",
        capture_metric_group_results,
    )

    strategy_backtest.run_strategy_backtest_from_store(
        store=SimpleNamespace(
            get_trading_strategy=lambda strategy_id: SimpleNamespace(
                trading_strategy=strategy
            ),
            get_subject_set=lambda subject_set_id: SimpleNamespace(
                definition=subject_set
            ),
        ),
        strategy_id="strategy:crypto_regime_momentum",
        subject_set_id="crypto",
        target_id="residual_return_1d",
        evaluation_date_ranges=(
            EvaluationDateRange(
                label="eval",
                start_date="2026-01-01",
                end_date="2026-03-02",
            ),
        ),
        base_url="fixture://",
        portfolio_construction=strategy.portfolio_construction,
        rebalance_friction_policy=EvaluationRebalanceFrictionPolicySpec(),
        execution_cost_assumptions=ExecutionCostAssumptionsSpec(),
        holding_cost_assumptions=HoldingCostAssumptionsSpec(),
        feature_plane_repository=None,
    )

    signal_series_by_subject = captured["signal_series_by_subject"]
    assert signal_series_by_subject["BTC"].loc["2026-01-29"] == pytest.approx(0.0)
    assert signal_series_by_subject["BTC"].loc["2026-01-30"] == pytest.approx(1.0)
    assert captured["funding_cost_bps_series_by_subject"]["BTC"].iloc[0] == (
        pytest.approx(10.0)
    )


def test_run_evaluation_uses_archived_prepared_snapshots(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "observables": [
                    {
                        "observable_id": "daily_close",
                        "family": "price",
                        "value_kind": "real_value",
                        "default_resolution": "1d",
                    }
                ],
                "signal_specs": [
                    {
                        "signal_id": "reversal_1d",
                        "kind": "reversal",
                        "required_observable_id": "daily_close",
                        "target_definition": {
                            "target_id": "residual_return_3d",
                            "family": "residual_return",
                            "observation_kind": "realized_return",
                            "subject_kind": "asset",
                            "output_kind": "real_value",
                            "scoring_kind": "corr",
                            "horizon_days": 3,
                            "params": {},
                        },
                        "params": {"lookback": 1},
                    },
                    {
                        "signal_id": "reversal_3d",
                        "kind": "reversal",
                        "required_observable_id": "daily_close",
                        "target_definition": {
                            "target_id": "residual_return_3d",
                            "family": "residual_return",
                            "observation_kind": "realized_return",
                            "subject_kind": "asset",
                            "output_kind": "real_value",
                            "scoring_kind": "corr",
                            "horizon_days": 3,
                            "params": {},
                        },
                        "params": {"lookback": 3},
                    },
                    {
                        "signal_id": "average_gap_3d",
                        "kind": "average_gap",
                        "required_observable_id": "daily_close",
                        "target_definition": {
                            "target_id": "residual_return_3d",
                            "family": "residual_return",
                            "observation_kind": "realized_return",
                            "subject_kind": "asset",
                            "output_kind": "real_value",
                            "scoring_kind": "corr",
                            "horizon_days": 3,
                            "params": {},
                        },
                        "params": {"lookback": 3},
                    },
                ],
                "subject_sets": [
                    {
                        "subject_set_id": "core_crypto",
                        "observation_specs": [
                            {
                                "observation_spec_id": "btc_close",
                                "observable_id": "daily_close",
                            }
                        ],
                        "bindings": [
                            {
                                "subject_id": "BTC_spot",
                                "subject_kind": "asset",
                                "asset": "BTC",
                                "observation_spec_id": "btc_close",
                            }
                        ],
                    }
                ],
                "signal_discoveries": [
                    {
                        "signal_discovery_id": "core_crypto_search",
                        "subject_set_id": "core_crypto",
                        "selection_policy": {
                            "min_sample_count": 1,
                            "min_abs_corr": 0.0,
                            "probe_max_dates": 3,
                            "probe_min_sample_count": 2,
                            "probe_min_abs_corr": 0.0,
                            "probe_max_family_survivors_per_subject": 1,
                            "survivor_min_sample_count": 2,
                            "survivor_min_abs_corr": 0.0,
                            "survivor_max_family_survivors_per_subject": 1,
                            "snapshot_retention": "latest_per_survivor",
                            "max_family_survivors_per_subject": 2,
                        },
                        "families": [
                            {
                                "family_id": "reversal_family",
                                "kind": "reversal",
                                "parameter_space": {
                                    "lookback": [1, 3],
                                },
                                "required_observable_id": "daily_close",
                                "target_id": "residual_return_3d",
                                "survivor_budget": 1,
                            },
                            {
                                "family_id": "average_gap_family",
                                "kind": "average_gap",
                                "parameter_space": {
                                    "lookback": [3],
                                },
                                "required_observable_id": "daily_close",
                                "target_id": "residual_return_3d",
                            },
                        ],
                        "target_id": "residual_return_3d",
                    }
                ],
                "evaluation_specs": [
                    {
                        "evaluation_spec_id": "core_crypto_eval",
                        "execution_range": {
                            "label": "exec_window",
                            "start_date": "2026-03-23",
                            "end_date": "2026-03-24",
                        },
                        "evaluation_date_ranges": [
                            {
                                "label": "eval_window",
                                "start_date": "2026-03-23",
                                "end_date": "2026-03-24",
                            }
                        ],
                        "metric_windows": [2],
                        "metric_group_names": [
                            "signed_belief_quality",
                            "portfolio_target_return_alignment",
                            "sizing_policy_quality",
                            "rebalance_policy_quality",
                            "decision_quality",
                            "robustness",
                        ],
                    }
                ],
                "evaluation_tasks": [
                    {
                        "evaluation_spec_id": "core_crypto_eval",
                        "signal_discovery_id": "core_crypto_search",
                        "base_url": "http://example.com",
                        "portfolio_construction": {
                            "sizing_policy": {"sizing_method": "signal_weighted", "sizing_engine": "rule_based"}
                        },
                        "rebalance_friction_policy": {},
                        "execution_cost_assumptions": {},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    def _fake_loader(observation_spec, *, asset: str, base_url: str, client=None):
        import pandas as pd

        assert asset == "BTC"
        return pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-20T00:00:00Z",
                    "2026-03-21T00:00:00Z",
                    "2026-03-22T00:00:00Z",
                    "2026-03-23T00:00:00Z",
                    "2026-03-24T00:00:00Z",
                    "2026-03-25T00:00:00Z",
                    "2026-03-26T00:00:00Z",
                    "2026-03-27T00:00:00Z",
                ],
                "value": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0, 107.0, 106.0],
            }
        )

    import alpha_os.data_repositories as data_repositories

    original_loader = data_repositories.load_observation_frame
    data_repositories.load_observation_frame = _fake_loader
    try:
        assert (
            main(
                [
                    "run-walk-forward",
                    "--db",
                    str(db_path),
                    "--evaluation-spec-id",
                    "core_crypto_eval",
                ]
            )
            == 0
        )
    finally:
        data_repositories.load_observation_frame = original_loader
    capsys.readouterr()

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        report_state = store.get_latest_evaluation_report()
        assert report_state is not None
        task_result = report_state.report.task_results[0]
        decision_metric_group_result = next(
            item for item in task_result.metric_group_results if item.metric_group_name == "decision_quality"
        )
        assert decision_metric_group_result.metrics["mean_decision_step_count"] > 1.0
        assert decision_metric_group_result.metrics["total_decision_step_count"] > 1
        assert "annualized_step_sharpe" in decision_metric_group_result.metrics
        assert "pooled_step_max_drawdown" in decision_metric_group_result.metrics
    finally:
        store.close()


def test_apply_runtime_manifest_accepts_explicit_strategy_specs(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "observables": [
                    {
                        "observable_id": "daily_close",
                        "family": "price",
                        "value_kind": "real_value",
                        "default_resolution": "1d",
                    }
                ],
                "signal_specs": [
                    {
                        "signal_id": "reversal_1d",
                        "kind": "reversal",
                        "required_observable_id": "daily_close",
                        "target_definition": {
                            "target_id": "residual_return_3d",
                            "family": "residual_return",
                            "observation_kind": "realized_return",
                            "subject_kind": "asset",
                            "output_kind": "real_value",
                            "scoring_kind": "corr",
                            "horizon_days": 3,
                            "params": {},
                        },
                        "params": {"lookback": 1},
                    }
                ],
                "subject_sets": [
                    {
                        "subject_set_id": "core_crypto",
                        "observation_specs": [
                            {
                                "observation_spec_id": "btc_close",
                                "observable_id": "daily_close",
                            }
                        ],
                        "bindings": [
                            {
                                "subject_id": "BTC_spot",
                                "subject_kind": "asset",
                                "asset": "BTC",
                                "observation_spec_id": "btc_close",
                            }
                        ],
                    }
                ],
                "signal_discoveries": [
                    {
                        "signal_discovery_id": "core_crypto_search",
                        "subject_set_id": "core_crypto",
                        "signal_spec_ids": [
                            "reversal_1d"
                        ],
                        "selection_policy": {
                            "min_sample_count": 1,
                            "min_abs_corr": 0.0,
                        },
                        "target_id": "residual_return_3d",
                    }
                ],
                "strategy_specs": [
                    {
                        "trading_strategy": {
                            "strategy_id": "strategy:core_crypto_rule",
                            "label": "Core Crypto Rule",
                            "scope": {
                                "subject_set_id": "core_crypto",
                                "target_id": "residual_return_3d",
                            },
                            "signal_discovery_id": "core_crypto_search",
                            "position_rule_id": "constant_hold",
                            "family_mix": "spec:-",
                            **_strategy_portfolio_document(
                                sizing_method="signal_weighted",
                                direction_mode="long_short",
                                gross_exposure_cap=None,
                                rebalance_friction_policy={
                                    "turnover_friction": 0.0,
                                    "no_trade_band": 0.0,
                                },
                                execution_policy={
                                    "market_impact_bps": 0.0,
                                    "fee_bps": 0.0,
                                    "bid_ask_spread_bps": 0.0,
                                },
                            ),
                            "created_at": "2026-04-05T00:00:00+00:00",
                        }
                    }
                ],
                "evaluation_specs": [
                    {
                        "evaluation_spec_id": "core_crypto_eval",
                        "execution_range": {
                            "label": "exec_window",
                            "start_date": "2026-03-23",
                            "end_date": "2026-03-24",
                        },
                        "evaluation_date_ranges": [
                            {
                                "label": "eval_window",
                                "start_date": "2026-03-23",
                                "end_date": "2026-03-24",
                            }
                        ],
                        "metric_windows": [2],
                        "metric_group_names": [
                            "signed_belief_quality",
                            "portfolio_target_return_alignment",
                            "sizing_policy_quality",
                            "rebalance_policy_quality",
                            "decision_quality",
                            "robustness",
                        ],
                    }
                ],
                "evaluation_tasks": [
                    {
                        "evaluation_spec_id": "core_crypto_eval",
                        "strategy_id": "strategy:core_crypto_rule",
                        "base_url": "http://example.com",
                        "portfolio_construction": {
                            "sizing_policy": {"sizing_method": "signal_weighted", "sizing_engine": "rule_based"},
                        },
                        "rebalance_friction_policy": {},
                        "execution_cost_assumptions": {},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        strategy_state = store.get_trading_strategy("strategy:core_crypto_rule")
        assert strategy_state is not None
        trading_strategy = strategy_state.trading_strategy
        assert trading_strategy.signal_discovery_id == "core_crypto_search"
        evaluation_task_states = store.list_evaluation_tasks(limit=10)
        assert len(evaluation_task_states) == 1
        assert evaluation_task_states[0].task.strategy_id == "strategy:core_crypto_rule"
    finally:
        store.close()


def test_apply_runtime_manifest_accepts_trading_strategy_specs(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "observables": [
                    {
                        "observable_id": "daily_close",
                        "family": "price",
                        "value_kind": "real_value",
                        "default_resolution": "1d",
                    }
                ],
                "signal_specs": [
                    {
                        "signal_id": "reversal_1d",
                        "kind": "reversal",
                        "required_observable_id": "daily_close",
                        "target_definition": {
                            "target_id": "residual_return_3d",
                            "family": "residual_return",
                            "observation_kind": "realized_return",
                            "subject_kind": "asset",
                            "output_kind": "real_value",
                            "scoring_kind": "corr",
                            "horizon_days": 3,
                            "params": {},
                        },
                        "params": {"lookback": 1},
                    }
                ],
                "subject_sets": [
                    {
                        "subject_set_id": "core_crypto",
                        "observation_specs": [
                            {
                                "observation_spec_id": "btc_close",
                                "observable_id": "daily_close",
                            }
                        ],
                        "bindings": [
                            {
                                "subject_id": "BTC_spot",
                                "subject_kind": "asset",
                                "asset": "BTC",
                                "observation_spec_id": "btc_close",
                            }
                        ],
                    }
                ],
                "signal_discoveries": [
                    {
                        "signal_discovery_id": "core_crypto_search",
                        "subject_set_id": "core_crypto",
                        "signal_spec_ids": [
                            "reversal_1d"
                        ],
                        "selection_policy": {
                            "min_sample_count": 1,
                            "min_abs_corr": 0.0,
                        },
                        "target_id": "residual_return_3d",
                    }
                ],
                "strategy_specs": [
                    {
                        "trading_strategy": {
                            "strategy_id": "strategy:core_crypto_rule",
                            "label": "Core Crypto Rule",
                            "scope": {
                                "subject_set_id": "core_crypto",
                                "target_id": "residual_return_3d",
                            },
                            "signal_discovery_id": "core_crypto_search",
                            "position_rule_id": "constant_hold",
                            "family_mix": "spec:-",
                            **_strategy_portfolio_document(
                                sizing_method="signal_weighted",
                                direction_mode="long_short",
                                gross_exposure_cap=None,
                                rebalance_friction_policy={
                                    "turnover_friction": 0.0,
                                    "no_trade_band": 0.0,
                                },
                                execution_policy={
                                    "market_impact_bps": 0.0,
                                    "fee_bps": 0.0,
                                    "bid_ask_spread_bps": 0.0,
                                },
                            ),
                            "created_at": "2026-04-05T00:00:00+00:00",
                        }
                    }
                ],
                "evaluation_specs": [
                    {
                        "evaluation_spec_id": "core_crypto_eval",
                        "execution_range": {
                            "label": "exec_window",
                            "start_date": "2026-03-23",
                            "end_date": "2026-03-24",
                        },
                        "evaluation_date_ranges": [
                            {
                                "label": "eval_window",
                                "start_date": "2026-03-23",
                                "end_date": "2026-03-24",
                            }
                        ],
                        "metric_windows": [2],
                        "metric_group_names": [
                            "signed_belief_quality",
                            "portfolio_target_return_alignment",
                            "sizing_policy_quality",
                            "rebalance_policy_quality",
                            "decision_quality",
                            "robustness",
                        ],
                    }
                ],
                "evaluation_tasks": [
                    {
                        "evaluation_spec_id": "core_crypto_eval",
                        "strategy_id": "strategy:core_crypto_rule",
                        "base_url": "http://example.com",
                        "portfolio_construction": {
                            "sizing_policy": {"sizing_method": "signal_weighted", "sizing_engine": "rule_based"},
                        },
                        "rebalance_friction_policy": {},
                        "execution_cost_assumptions": {},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        strategy_specs = store.list_trading_strategies(limit=10)
        assert len(strategy_specs) == 1
        trading_strategy = strategy_specs[0].trading_strategy
        assert trading_strategy.strategy_id == "strategy:core_crypto_rule"
        assert trading_strategy.scope.subject_set_id == "core_crypto"
        assert (
            trading_strategy.signal_discovery_id
            == "core_crypto_search"
        )
        assert trading_strategy.portfolio.portfolio_construction.sizing_method == (
            "signal_weighted"
        )
    finally:
        store.close()


def test_apply_runtime_manifest_accepts_search_free_evaluation_task(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "strategy_specs": [
                    {
                        "trading_strategy": {
                            "strategy_id": "strategy:buy_and_hold",
                            "label": "Buy And Hold",
                            "scope": {
                                "subject_set_id": "broad_9_etf",
                                "target_id": None,
                            },
                            "signal_discovery_id": None,
                            "position_rule_id": "constant_hold",
                            "family_mix": None,
                            **_strategy_portfolio_document(
                                sizing_method="equal_weight",
                                direction_mode=None,
                                gross_exposure_cap=None,
                                rebalance_friction_policy={
                                    "turnover_friction": None,
                                    "no_trade_band": None,
                                },
                                execution_policy={
                                    "market_impact_bps": None,
                                    "fee_bps": None,
                                    "bid_ask_spread_bps": None,
                                },
                            ),
                            "created_at": "2026-04-05T00:00:00+00:00",
                        }
                    }
                ],
                "evaluation_specs": [
                    {
                        "evaluation_spec_id": "buy_and_hold_eval",
                        "execution_range": {
                            "label": "exec_window",
                            "start_date": "2026-03-23",
                            "end_date": "2026-03-24",
                        },
                        "evaluation_date_ranges": [
                            {
                                "label": "eval_window",
                                "start_date": "2026-03-23",
                                "end_date": "2026-03-24",
                            }
                        ],
                        "metric_windows": [2],
                        "metric_group_names": ["decision_quality"],
                    }
                ],
                "evaluation_tasks": [
                    {
                        "evaluation_spec_id": "buy_and_hold_eval",
                        "strategy_id": "strategy:buy_and_hold",
                        "base_url": "http://example.com",
                        "portfolio_construction": {
                            "sizing_policy": {"sizing_method": "equal_weight", "sizing_engine": "history_based"},
                        },
                        "rebalance_friction_policy": {},
                        "execution_cost_assumptions": {},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        evaluation_task_states = store.list_evaluation_tasks(limit=10)
        assert len(evaluation_task_states) == 1
        case = evaluation_task_states[0].task
        assert case.strategy_id == "strategy:buy_and_hold"
    finally:
        store.close()


def test_run_walk_forward_evaluation_executes_search_free_strategy(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "observables": [
                    {
                        "observable_id": "daily_close",
                        "family": "price",
                        "value_kind": "real_value",
                        "default_resolution": "1d",
                    }
                ],
                "subject_sets": [
                    {
                        "subject_set_id": "core_crypto",
                        "observation_specs": [
                            {
                                "observation_spec_id": "btc_close",
                                "observable_id": "daily_close",
                            }
                        ],
                        "bindings": [
                            {
                                "subject_id": "BTC_spot",
                                "subject_kind": "asset",
                                "asset": "BTC",
                                "observation_spec_id": "btc_close",
                            }
                        ],
                    }
                ],
                "strategy_specs": [
                    {
                        "trading_strategy": {
                            "strategy_id": "strategy:buy_and_hold",
                            "label": "Buy And Hold",
                            "scope": {
                                "subject_set_id": "core_crypto",
                                "target_id": "residual_return_3d",
                            },
                            "signal_discovery_id": None,
                            "position_rule_id": "constant_hold",
                            "family_mix": None,
                            **_strategy_portfolio_document(
                                sizing_method="equal_weight",
                                direction_mode=None,
                                gross_exposure_cap=None,
                                rebalance_friction_policy={
                                    "turnover_friction": None,
                                    "no_trade_band": None,
                                },
                                execution_policy={
                                    "market_impact_bps": None,
                                    "fee_bps": None,
                                    "bid_ask_spread_bps": None,
                                },
                            ),
                            "created_at": "2026-04-05T00:00:00+00:00",
                        }
                    }
                ],
                "evaluation_specs": [
                    {
                        "evaluation_spec_id": "buy_and_hold_walk_forward",
                        "execution_range": {
                            "label": "compat_window",
                            "start_date": "2026-03-23",
                            "end_date": "2026-03-24",
                        },
                        "evaluation_folds": [
                            {
                                "label": "fold_a",
                                "execution_range": {
                                    "label": "train_a",
                                    "start_date": "2026-03-23",
                                    "end_date": "2026-03-24",
                                },
                                "evaluation_date_ranges": [
                                    {
                                        "label": "test_a",
                                        "start_date": "2026-03-25",
                                        "end_date": "2026-03-27",
                                    }
                                ],
                            }
                        ],
                        "metric_windows": [2],
                        "metric_group_names": [
                            "decision_quality",
                            "robustness",
                        ],
                    }
                ],
                "evaluation_tasks": [
                    {
                        "evaluation_spec_id": "buy_and_hold_walk_forward",
                        "strategy_id": "strategy:buy_and_hold",
                        "base_url": "http://example.com",
                        "portfolio_construction": {
                            "sizing_policy": {"sizing_method": "equal_weight", "sizing_engine": "history_based"}
                        },
                        "rebalance_friction_policy": {},
                        "execution_cost_assumptions": {},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    def _fake_loader(observation_spec, *, asset: str, base_url: str, client=None):
        import pandas as pd

        assert asset == "BTC"
        return pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-20T00:00:00Z",
                    "2026-03-21T00:00:00Z",
                    "2026-03-22T00:00:00Z",
                    "2026-03-23T00:00:00Z",
                    "2026-03-24T00:00:00Z",
                    "2026-03-25T00:00:00Z",
                    "2026-03-26T00:00:00Z",
                    "2026-03-27T00:00:00Z",
                ],
                "value": [
                    100.0,
                    101.0,
                    102.0,
                    104.0,
                    103.0,
                    105.0,
                    107.0,
                    108.0,
                ],
            }
        )

    import alpha_os.data_repositories as data_repositories

    original_loader = data_repositories.load_observation_frame
    data_repositories.load_observation_frame = _fake_loader
    try:
        assert (
            main(
                [
                    "run-walk-forward-evaluation",
                    "--db",
                    str(db_path),
                    "--evaluation-spec-id",
                    "buy_and_hold_walk_forward",
                    "--base-url",
                    "http://example.com",
                ]
            )
            == 0
        )
    finally:
        data_repositories.load_observation_frame = original_loader

    output = capsys.readouterr().out
    assert "alpha-os evaluation run" in output
    assert "TaskResults: 1" in output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        report_state = store.get_latest_evaluation_report()
        assert report_state is not None
        assert len(report_state.report.task_results) == 1
        task_result = report_state.report.task_results[0]
        assert task_result.signal_discovery_id is None
        decision_metric_group_result = next(
            item
            for item in task_result.metric_group_results
            if item.metric_group_name == "decision_quality"
        )
        assert decision_metric_group_result.metrics["total_decision_step_count"] > 0
    finally:
        store.close()


def test_run_walk_forward_evaluation_executes_search_free_top_k_strategy(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    manifest_path = tmp_path / "runtime-manifest-top-k.json"
    manifest_path.write_text(
        json.dumps(
            {
                "observables": [
                    {
                        "observable_id": "daily_close",
                        "family": "price",
                        "value_kind": "real_value",
                        "default_resolution": "1d",
                    }
                ],
                "subject_sets": [
                    {
                        "subject_set_id": "core_crypto_top_k",
                        "observation_specs": [
                            {
                                "observation_spec_id": "btc_close",
                                "observable_id": "daily_close",
                            },
                            {
                                "observation_spec_id": "eth_close",
                                "observable_id": "daily_close",
                            },
                        ],
                        "bindings": [
                            {
                                "subject_id": "BTC_spot",
                                "subject_kind": "asset",
                                "asset": "BTC",
                                "observation_spec_id": "btc_close",
                            },
                            {
                                "subject_id": "ETH_spot",
                                "subject_kind": "asset",
                                "asset": "ETH",
                                "observation_spec_id": "eth_close",
                            },
                        ],
                        "universe_policy": {
                            "base_currency": "USD",
                            "trading_calendar": "UTC",
                            "benchmark_id": "core_crypto_top_k",
                        },
                    }
                ],
                "strategy_specs": [
                    {
                        "trading_strategy": {
                            "strategy_id": "strategy:top_k_hold",
                            "label": "Top K Hold",
                            "scope": {
                                "subject_set_id": "core_crypto_top_k",
                                "target_id": "residual_return_3d",
                            },
                            "signal_discovery_id": None,
                            "position_rule_id": "constant_hold",
                            "family_mix": None,
                            **_strategy_portfolio_document(
                                sizing_method="equal_weight",
                                direction_mode="long_only",
                                gross_exposure_cap=1.0,
                                selection_kind="top_k",
                                top_k=1,
                                rebalance_friction_policy={
                                    "turnover_friction": None,
                                    "no_trade_band": None,
                                },
                                execution_policy={
                                    "market_impact_bps": None,
                                    "fee_bps": None,
                                    "bid_ask_spread_bps": None,
                                },
                            ),
                            "created_at": "2026-04-05T00:00:00+00:00",
                        }
                    }
                ],
                "evaluation_specs": [
                    {
                        "evaluation_spec_id": "top_k_hold_walk_forward",
                        "execution_range": {
                            "label": "compat_window",
                            "start_date": "2026-03-23",
                            "end_date": "2026-03-24",
                        },
                        "evaluation_folds": [
                            {
                                "label": "fold_a",
                                "execution_range": {
                                    "label": "train_a",
                                    "start_date": "2026-03-23",
                                    "end_date": "2026-03-24",
                                },
                                "evaluation_date_ranges": [
                                    {
                                        "label": "test_a",
                                        "start_date": "2026-03-25",
                                        "end_date": "2026-03-27",
                                    }
                                ],
                            }
                        ],
                        "metric_windows": [2],
                        "metric_group_names": [
                            "decision_quality",
                            "robustness",
                        ],
                    }
                ],
                "evaluation_tasks": [
                    {
                        "evaluation_spec_id": "top_k_hold_walk_forward",
                        "strategy_id": "strategy:top_k_hold",
                        "base_url": "http://example.com",
                        "portfolio_construction": {
                            "sizing_policy": {
                                "sizing_method": "equal_weight",
                                "sizing_engine": "history_based"
                            },
                            "direction_mode": "long_only",
                            "top_k": 1,
                            "gross_exposure_cap": 1.0
                        },
                        "rebalance_friction_policy": {},
                        "execution_cost_assumptions": {},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    def _fake_loader(observation_spec, *, asset: str, base_url: str, client=None):
        import pandas as pd

        frames = {
            "BTC": pd.DataFrame(
                {
                    "timestamp": [
                        "2026-03-20T00:00:00Z",
                        "2026-03-21T00:00:00Z",
                        "2026-03-22T00:00:00Z",
                        "2026-03-23T00:00:00Z",
                        "2026-03-24T00:00:00Z",
                        "2026-03-25T00:00:00Z",
                        "2026-03-26T00:00:00Z",
                        "2026-03-27T00:00:00Z",
                    ],
                    "value": [100.0, 101.0, 102.0, 104.0, 103.0, 105.0, 107.0, 108.0],
                }
            ),
            "ETH": pd.DataFrame(
                {
                    "timestamp": [
                        "2026-03-20T00:00:00Z",
                        "2026-03-21T00:00:00Z",
                        "2026-03-22T00:00:00Z",
                        "2026-03-23T00:00:00Z",
                        "2026-03-24T00:00:00Z",
                        "2026-03-25T00:00:00Z",
                        "2026-03-26T00:00:00Z",
                        "2026-03-27T00:00:00Z",
                    ],
                    "value": [50.0, 49.0, 48.0, 47.0, 46.0, 45.0, 44.0, 43.0],
                }
            ),
        }
        return frames[asset]

    import alpha_os.data_repositories as data_repositories

    original_loader = data_repositories.load_observation_frame
    data_repositories.load_observation_frame = _fake_loader
    try:
        assert (
            main(
                [
                    "run-walk-forward-evaluation",
                    "--db",
                    str(db_path),
                    "--evaluation-spec-id",
                    "top_k_hold_walk_forward",
                    "--base-url",
                    "http://example.com",
                ]
            )
            == 0
        )
    finally:
        data_repositories.load_observation_frame = original_loader

    output = capsys.readouterr().out
    assert "alpha-os evaluation run" in output
    assert "TaskResults: 1" in output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        report_state = store.get_latest_evaluation_report()
        assert report_state is not None
        task_result = report_state.report.task_results[0]
        decision_metric_group_result = next(
            item
            for item in task_result.metric_group_results
            if item.metric_group_name == "decision_quality"
        )
        assert decision_metric_group_result.metrics["total_decision_step_count"] > 0
    finally:
        store.close()


def test_run_walk_forward_evaluation_executes_trainless_dual_momentum_strategy(
    tmp_path, capsys
):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    manifest_path = tmp_path / "runtime-manifest-dual-momentum.json"
    manifest_path.write_text(
        json.dumps(
            {
                "observables": [
                    {
                        "observable_id": "daily_close",
                        "family": "price",
                        "value_kind": "real_value",
                        "default_resolution": "1d",
                    }
                ],
                "subject_sets": [
                    {
                        "subject_set_id": "core_crypto_dual_momentum",
                        "observation_specs": [
                            {
                                "observation_spec_id": "btc_close",
                                "observable_id": "daily_close",
                            },
                            {
                                "observation_spec_id": "eth_close",
                                "observable_id": "daily_close",
                            },
                        ],
                        "bindings": [
                            {
                                "subject_id": "BTC_spot",
                                "subject_kind": "asset",
                                "asset": "BTC",
                                "observation_spec_id": "btc_close",
                            },
                            {
                                "subject_id": "ETH_spot",
                                "subject_kind": "asset",
                                "asset": "ETH",
                                "observation_spec_id": "eth_close",
                            },
                        ],
                        "universe_policy": {
                            "base_currency": "USD",
                            "trading_calendar": "UTC",
                            "benchmark_id": "core_crypto_dual_momentum",
                        },
                    }
                ],
                "strategy_specs": [
                    {
                        "trading_strategy": {
                            "strategy_id": "strategy:dual_momentum_hold",
                            "label": "Dual Momentum Hold",
                            "scope": {
                                "subject_set_id": "core_crypto_dual_momentum",
                                "target_id": "residual_return_3d",
                            },
                            "signal_discovery_id": None,
                            "position_rule_id": "dual_momentum_hold",
                            "family_mix": "lookback=2",
                            **_strategy_portfolio_document(
                                sizing_method="signal_weighted",
                                direction_mode="long_only",
                                gross_exposure_cap=1.0,
                                selection_kind="top_k",
                                top_k=1,
                                rebalance_friction_policy={
                                    "turnover_friction": None,
                                    "no_trade_band": None,
                                },
                                execution_policy={
                                    "market_impact_bps": None,
                                    "fee_bps": None,
                                    "bid_ask_spread_bps": None,
                                },
                            ),
                            "created_at": "2026-04-05T00:00:00+00:00",
                        }
                    }
                ],
                "evaluation_specs": [
                    {
                        "evaluation_spec_id": "dual_momentum_hold_walk_forward",
                        "execution_range": {
                            "label": "compat_window",
                            "start_date": "2026-03-23",
                            "end_date": "2026-03-24",
                        },
                        "evaluation_folds": [
                            {
                                "label": "fold_a",
                                "execution_range": {
                                    "label": "train_a",
                                    "start_date": "2026-03-23",
                                    "end_date": "2026-03-24",
                                },
                                "evaluation_date_ranges": [
                                    {
                                        "label": "test_a",
                                        "start_date": "2026-03-25",
                                        "end_date": "2026-03-27",
                                    }
                                ],
                            }
                        ],
                        "metric_windows": [2],
                        "metric_group_names": [
                            "decision_quality",
                            "robustness",
                        ],
                    }
                ],
                "evaluation_tasks": [
                    {
                        "evaluation_spec_id": "dual_momentum_hold_walk_forward",
                        "strategy_id": "strategy:dual_momentum_hold",
                        "base_url": "http://example.com",
                        "portfolio_construction": {
                            "sizing_policy": {
                                "sizing_method": "signal_weighted",
                                "sizing_engine": "rule_based"
                            },
                            "rebalance_interval_steps": 1,
                            "direction_mode": "long_only",
                            "top_k": 1,
                            "gross_exposure_cap": 1.0
                        },
                        "rebalance_friction_policy": {},
                        "execution_cost_assumptions": {},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    def _fake_loader(observation_spec, *, asset: str, base_url: str, client=None):
        import pandas as pd

        frames = {
            "BTC": pd.DataFrame(
                {
                    "timestamp": [
                        "2026-03-20T00:00:00Z",
                        "2026-03-21T00:00:00Z",
                        "2026-03-22T00:00:00Z",
                        "2026-03-23T00:00:00Z",
                        "2026-03-24T00:00:00Z",
                        "2026-03-25T00:00:00Z",
                        "2026-03-26T00:00:00Z",
                        "2026-03-27T00:00:00Z",
                    ],
                    "value": [100.0, 101.0, 102.0, 104.0, 106.0, 108.0, 110.0, 111.0],
                }
            ),
            "ETH": pd.DataFrame(
                {
                    "timestamp": [
                        "2026-03-20T00:00:00Z",
                        "2026-03-21T00:00:00Z",
                        "2026-03-22T00:00:00Z",
                        "2026-03-23T00:00:00Z",
                        "2026-03-24T00:00:00Z",
                        "2026-03-25T00:00:00Z",
                        "2026-03-26T00:00:00Z",
                        "2026-03-27T00:00:00Z",
                    ],
                    "value": [50.0, 49.0, 48.0, 47.0, 46.0, 45.0, 44.0, 43.0],
                }
            ),
        }
        return frames[asset]

    import alpha_os.data_repositories as data_repositories

    original_loader = data_repositories.load_observation_frame
    data_repositories.load_observation_frame = _fake_loader
    try:
        assert (
            main(
                [
                    "run-walk-forward-evaluation",
                    "--db",
                    str(db_path),
                    "--evaluation-spec-id",
                    "dual_momentum_hold_walk_forward",
                    "--base-url",
                    "http://example.com",
                ]
            )
            == 0
        )
    finally:
        data_repositories.load_observation_frame = original_loader

    output = capsys.readouterr().out
    assert "alpha-os evaluation run" in output
    assert "TaskResults: 1" in output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        report_state = store.get_latest_evaluation_report()
        assert report_state is not None
        task_result = report_state.report.task_results[0]
        decision_metric_group_result = next(
            item
            for item in task_result.metric_group_results
            if item.metric_group_name == "decision_quality"
        )
        assert decision_metric_group_result.metrics["total_decision_step_count"] > 0
        assert decision_metric_group_result.metrics["mean_decision_net_return"] > 0.0
    finally:
        store.close()


def test_run_walk_forward_evaluation_supports_checked_in_global_macro_manifest(tmp_path, capsys):
    from pathlib import Path

    import json
    import numpy as np
    import pandas as pd

    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    source_manifest_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "runtime_manifests"
        / "global_macro_futures_daily_trend.json"
    )
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    keep_assets = {"ES", "ZN", "CL", "GC", "BTCUSDT", "ETHUSDT"}
    subject_set = manifest["subject_sets"][0]
    subject_set["instruments"] = [
        item for item in subject_set["instruments"] if item["asset"] in keep_assets
    ]
    subject_set["bindings"] = [
        item for item in subject_set["bindings"] if item["asset"] in keep_assets
    ]
    manifest["generated_signal_discoveries"][0]["parameter_space"] = {"lookback": [20]}
    manifest["generated_signal_discoveries"][0]["selection_policy"].update(
        {
            "min_sample_count": 10,
            "min_abs_corr": 0.0,
            "min_stability_score": 0.0,
            "pre_screen_top_k_per_kind": 1,
            "probe_max_dates": 10,
            "probe_min_sample_count": 5,
            "probe_min_abs_corr": 0.0,
            "survivor_min_sample_count": 5,
            "survivor_min_abs_corr": 0.0,
        }
    )
    manifest["evaluation_specs"][0]["execution_range"] = {
        "label": "train_2024h1",
        "start_date": "2024-01-01",
        "end_date": "2024-04-30",
    }
    manifest["evaluation_specs"][0]["evaluation_folds"] = [
        {
            "label": "fold_2024h1_to_2024m5",
            "execution_range": {
                "label": "train_2024h1",
                "start_date": "2024-01-01",
                "end_date": "2024-04-30",
            },
            "evaluation_date_ranges": [
                {
                    "label": "2024m5",
                    "start_date": "2024-05-01",
                    "end_date": "2024-05-31",
                }
            ],
        }
    ]
    manifest_path = tmp_path / "global-macro-trimmed.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    assert (
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    asset_phase = {
        asset: (index % 17) * 0.23
        for index, asset in enumerate(("ES", "ZN", "CL", "GC", "BTCUSDT", "ETHUSDT"))
    }

    def _fake_loader(observation_spec, *, asset: str, base_url: str, client=None):
        dates = pd.date_range("2023-11-01", "2024-05-31", freq="D", tz="UTC")
        index = np.arange(len(dates), dtype=float)
        phase = asset_phase[asset]
        latent = (
            0.0009
            + 0.0030 * np.sin(index / 11.0 + phase)
            + 0.0014 * np.cos(index / 27.0 + phase * 0.5)
        )
        latent = np.clip(latent, -0.01, 0.01)
        close = 100.0 * np.cumprod(1.0 + latent)
        frame = pd.DataFrame(
            {
                "timestamp": dates.strftime("%Y-%m-%dT00:00:00Z"),
                "value": close,
                "front_price": close,
                "next_price": close * (1.0 + latent * 0.8),
                "basis": latent * 1.2,
                "open_interest": 10000.0 + index * (1.0 + (phase % 1.0)),
            }
        )
        frame["funding_rate"] = latent * (0.8 if asset.endswith("USDT") else 0.2)
        return frame

    import alpha_os.data_repositories as data_repositories

    original_loader = data_repositories.load_observation_frame
    data_repositories.load_observation_frame = _fake_loader
    try:
        assert (
            main(
                [
                    "run-walk-forward-evaluation",
                    "--db",
                    str(db_path),
                    "--evaluation-spec-id",
                    "global_macro_futures_daily_trend_eval",
                    "--base-url",
                    "http://example.com",
                ]
            )
            == 0
        )
    finally:
        data_repositories.load_observation_frame = original_loader

    output = capsys.readouterr().out
    assert "alpha-os evaluation run" in output
    assert "TaskResults: 1" in output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        strategy_checkpoints = store.list_strategy_checkpoints(
            signal_discovery_id="global_macro_futures_daily_trend_search",
            limit=10,
        )
        assert len(strategy_checkpoints) >= 1
        assert {item.state.fold_label for item in strategy_checkpoints} >= {
            "fold_2024h1_to_2024m5",
        }
        report_state = store.get_latest_evaluation_report()
        assert report_state is not None
        assert len(report_state.report.task_results) == 1
        task_result = report_state.report.task_results[0]
        assert task_result.signal_discovery_id == "global_macro_futures_daily_trend_search"
        assert task_result.artifact_refs.get("strategy_checkpoint_ids")
        decision_metric_group_result = next(
            item
            for item in task_result.metric_group_results
            if item.metric_group_name == "decision_quality"
        )
        assert decision_metric_group_result.metrics["total_decision_step_count"] > 0
    finally:
        store.close()


def test_run_diagnostic_evaluation_applies_extended_manifest_and_prints_focus(
    tmp_path, capsys
):
    import numpy as np

    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    base_manifest_path = tmp_path / "base-runtime-manifest.json"
    diagnostic_manifest_path = tmp_path / "diagnostic-runtime-manifest.json"
    base_manifest_path.write_text(
        json.dumps(
            {
                "observables": [
                    {
                        "observable_id": "daily_close",
                        "family": "price",
                        "value_kind": "real_value",
                        "default_resolution": "1d",
                    }
                ],
                "signal_specs": [
                    {
                        "signal_id": "trend_2d",
                        "kind": "time_series_trend",
                        "required_observable_id": "daily_close",
                        "target_definition": {
                            "target_id": "residual_return_1d",
                            "family": "residual_return",
                            "observation_kind": "realized_return",
                            "subject_kind": "asset",
                            "output_kind": "real_value",
                            "scoring_kind": "corr",
                            "params": {"horizon_days": 1},
                        },
                        "params": {"lookback": 2},
                    }
                ],
                "subject_sets": [
                    {
                        "subject_set_id": "diagnostic_subjects",
                        "observation_specs": [
                            {
                                "observation_spec_id": "asset_close",
                                "observable_id": "daily_close",
                            }
                        ],
                        "bindings": [
                            {
                                "subject_id": "AAA_asset",
                                "subject_kind": "asset",
                                "asset": "AAA",
                                "observation_spec_id": "asset_close",
                            },
                            {
                                "subject_id": "BBB_asset",
                                "subject_kind": "asset",
                                "asset": "BBB",
                                "observation_spec_id": "asset_close",
                            },
                        ],
                        "universe_policy": {
                            "base_currency": "USD",
                            "trading_calendar": "daily",
                            "benchmark_id": "diagnostic_subjects",
                        },
                    }
                ],
                "signal_discoveries": [
                    {
                        "signal_discovery_id": "diagnostic_search",
                        "subject_set_id": "diagnostic_subjects",
                        "signal_spec_ids": ["trend_2d"],
                        "target_id": "residual_return_1d",
                        "selection_policy": {
                            "min_sample_count": 2,
                            "min_abs_corr": 0.0,
                            "probe_max_dates": 4,
                            "probe_min_sample_count": 2,
                            "probe_min_abs_corr": 0.0,
                            "probe_max_family_survivors_per_subject": 1,
                            "survivor_min_sample_count": 2,
                            "survivor_min_abs_corr": 0.0,
                            "survivor_max_family_survivors_per_subject": 1,
                            "snapshot_retention": "latest_per_survivor",
                            "max_family_survivors_per_subject": 1,
                        },
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    diagnostic_manifest_path.write_text(
        json.dumps(
            {
                "extends_manifest": base_manifest_path.name,
                "evaluation_specs": [
                    {
                        "evaluation_spec_id": "diagnostic_eval",
                        "execution_range": {
                            "label": "train",
                            "start_date": "2026-01-01",
                            "end_date": "2026-01-06",
                        },
                        "evaluation_folds": [
                            {
                                "label": "fold_train_to_test",
                                "execution_range": {
                                    "label": "train",
                                    "start_date": "2026-01-01",
                                    "end_date": "2026-01-06",
                                },
                                "evaluation_date_ranges": [
                                    {
                                        "label": "test",
                                        "start_date": "2026-01-07",
                                        "end_date": "2026-01-10",
                                    }
                                ],
                            }
                        ],
                        "metric_windows": [2],
                        "metric_group_names": [
                            "prediction_diagnostics",
                            "portfolio_target_return_alignment",
                        "decision_quality",
                        "portfolio_risk_budget",
                        "portfolio_construction_trace",
                        "execution_trace",
                        "cost_drag",
                        "signal_churn",
                        "portfolio_concentration",
                        "robustness",
                        ],
                    }
                ],
                "evaluation_tasks": [
                    {
                        "evaluation_task_id": "diagnostic_case",
                        "evaluation_spec_id": "diagnostic_eval",
                        "signal_discovery_id": "diagnostic_search",
                        "base_url": "http://manifest.example",
                        "portfolio_construction": {
                            "sizing_policy": {
                                "sizing_method": "signal_weighted",
                                "sizing_engine": "rule_based",
                            },
                            "rebalance_interval_steps": 1,
                            "direction_mode": "long_short",
                            "risk_budget": {
                                "risk_normalization_mode": "gross",
                                "target_gross_exposure": 0.5,
                            },
                        },
                        "rebalance_friction_policy": {},
                        "execution_cost_assumptions": {},
                        "holding_cost_assumptions": {},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def _fake_loader(observation_spec, *, asset: str, base_url: str, client=None):
        dates = pd.date_range("2025-12-25", "2026-01-12", freq="D", tz="UTC")
        index = np.arange(len(dates), dtype=float)
        slope = 0.002 if asset == "AAA" else -0.001
        close = 100.0 * np.cumprod(1.0 + slope + 0.0005 * np.sin(index))
        return pd.DataFrame(
            {
                "timestamp": dates.strftime("%Y-%m-%dT00:00:00Z"),
                "value": close,
            }
        )

    import alpha_os.data_repositories as data_repositories

    original_loader = data_repositories.load_observation_frame
    data_repositories.load_observation_frame = _fake_loader
    try:
        assert (
            main(
                [
                    "run-diagnostic-evaluation",
                    "--db",
                    str(db_path),
                    "--manifest",
                    str(diagnostic_manifest_path),
                    "--evaluation-spec-id",
                    "diagnostic_eval",
                    "--base-url",
                    "http://override.example",
                    "--details",
                ]
            )
            == 0
        )
    finally:
        data_repositories.load_observation_frame = original_loader

    output = capsys.readouterr().out
    assert "alpha-os diagnostic focus" in output
    assert "prediction_diagnostics:" in output
    assert "portfolio_construction_trace:" in output
    assert "execution_trace:" in output
    assert "cost_drag:" in output
    assert "signal_churn:" in output
    assert "portfolio_concentration:" in output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        assert store.get_subject_set("diagnostic_subjects") is not None
        report_state = store.get_latest_evaluation_report()
        assert report_state is not None
        assert report_state.report.evaluation_spec_id == "diagnostic_eval"
        task_result = report_state.report.task_results[0]
        assert {item.metric_group_name for item in task_result.metric_group_results} >= {
            "prediction_diagnostics",
            "portfolio_construction_trace",
            "execution_trace",
            "cost_drag",
            "signal_churn",
            "portfolio_concentration",
        }
    finally:
        store.close()


def test_run_diagnostic_evaluation_dry_run_validates_plan_without_report(
    tmp_path, capsys
):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"

    assert (
        main(
            [
                "run-diagnostic-evaluation",
                "--db",
                str(db_path),
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "alpha-os diagnostic dry run" in output
    assert "Cases:    15" in output
    assert "has_signal_discovery=false" in output
    assert "has_signal_discovery=true" in output
    assert "global_macro_tradeable_daily_diagnostic_equal_weight_hold_case" in output
    assert "global_macro_tradeable_daily_diagnostic_equal_weight_monthly_hold_case" in output
    assert "holding_style=equal_weight_hold" in output
    assert "construction=hold_baseline" in output
    assert "global_macro_tradeable_daily_diagnostic_utility_looser_benefit_case" in output
    assert "global_macro_tradeable_daily_diagnostic_mean_reversion_case" in output
    assert (
        "global_macro_tradeable_daily_diagnostic_mean_reversion_constrained_case"
        in output
    )
    assert (
        "global_macro_tradeable_daily_diagnostic_mean_reversion_optimizer_case"
        in output
    )
    assert "global_macro_tradeable_daily_diagnostic_no_risk_budget_case" in output
    assert "optimizer_backend=cvxpy_signed_mean_variance" in output
    assert "benefit_scale=2.0" in output
    assert "turnover_budget=0.025" in output
    assert "execution_mode=threshold" in output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        assert store.get_latest_evaluation_report() is None
    finally:
        store.close()


def test_run_diagnostic_evaluation_dry_run_does_not_create_db(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"

    assert (
        main(
            [
                "run-diagnostic-evaluation",
                "--db",
                str(db_path),
                "--dry-run",
                "--check",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "DryRunCheck: passed" in output
    assert not db_path.exists()


def test_run_fixture_diagnostic_evaluation_uses_local_csv_data(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"

    assert (
        main(
            [
                "run-diagnostic-evaluation",
                "--db",
                str(db_path),
                "--manifest",
                "fixture_daily_diagnostic",
                "--evaluation-spec-id",
                "fixture_daily_diagnostic_eval",
                "--details",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "fixture_daily_diagnostic.json" in output
    assert "fixture_daily_equal_weight_hold_case" in output
    assert "alpha-os diagnostic focus" in output
    assert "decision_quality:" in output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        report_state = store.get_latest_evaluation_report()
        assert report_state is not None
        assert report_state.report.evaluation_spec_id == "fixture_daily_diagnostic_eval"
        task_result = report_state.report.task_results[0]
        metric_names = {
            item.metric_group_name for item in task_result.metric_group_results
        }
        assert metric_names >= {
            "portfolio_target_return_alignment",
            "decision_quality",
            "portfolio_concentration",
            "robustness",
        }
    finally:
        store.close()


def test_run_diagnostic_evaluation_dry_run_does_not_apply_manifest(
    tmp_path,
    monkeypatch,
):
    import alpha_os.cli as cli

    def _fail_apply_runtime_manifest(_args):
        raise AssertionError("dry-run must not apply runtime manifests")

    monkeypatch.setattr(cli, "cmd_apply_runtime_manifest", _fail_apply_runtime_manifest)

    assert (
        cli.main(
            [
                "run-diagnostic-evaluation",
                "--db",
                str(tmp_path / "runtime.db"),
                "--dry-run",
                "--check",
            ]
        )
        == 0
    )


def test_run_diagnostic_evaluation_dry_run_check_passes_without_report(
    tmp_path, capsys
):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"

    assert (
        main(
            [
                "run-diagnostic-evaluation",
                "--db",
                str(db_path),
                "--dry-run",
                "--check",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "DryRunCheck: passed" in output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        assert store.get_latest_evaluation_report() is None
    finally:
        store.close()


def test_run_diagnostic_evaluation_dry_run_check_ignores_stale_tasks(
    tmp_path, capsys
):
    from alpha_os.cli import (
        _extended_runtime_manifest_paths,
        _resolve_runtime_manifest_path,
        main,
    )
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    manifest_path = _resolve_runtime_manifest_path(
        "global_macro_tradeable_daily_diagnostic"
    )

    for path in (*_extended_runtime_manifest_paths(manifest_path), manifest_path):
        assert (
            main(
                [
                    "apply-manifest",
                    "--db",
                    str(db_path),
                    "--manifest",
                    str(path),
                ]
            )
            == 0
        )
    capsys.readouterr()

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        source_state = store.get_evaluation_task(
            "global_macro_tradeable_daily_diagnostic_mean_reversion_case"
        )
        assert source_state is not None
        stale_task = replace(
            source_state.task,
            evaluation_task_id=(
                "global_macro_tradeable_daily_diagnostic_term_structure_carry_case"
            ),
        )
        store.upsert_evaluation_task(task=stale_task)
    finally:
        store.close()

    assert (
        main(
            [
                "run-diagnostic-evaluation",
                "--db",
                str(db_path),
                "--dry-run",
                "--check",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Cases:    15" in output
    assert "global_macro_tradeable_daily_diagnostic_term_structure_carry_case" not in output
    assert "DryRunCheck: passed" in output


def test_run_diagnostic_evaluation_check_requires_dry_run(tmp_path):
    from alpha_os.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "run-diagnostic-evaluation",
                "--db",
                str(tmp_path / "runtime.db"),
                "--check",
            ]
        )

    assert exc_info.value.code == 2


def test_run_diagnostic_evaluation_dry_run_check_rejects_finding_count_mismatch(
    tmp_path, monkeypatch
):
    import alpha_os.cli as cli

    monkeypatch.setattr(cli, "_DIAGNOSTIC_DRY_RUN_EXPECTED_CASE_COUNT", 12)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "run-diagnostic-evaluation",
                "--db",
                str(tmp_path / "runtime.db"),
                "--dry-run",
                "--check",
            ]
        )

    assert exc_info.value.code == 2


def test_build_evaluation_plan_supports_explicit_folds(tmp_path):
    from alpha_os.evaluation_task import EvaluationTask
    from alpha_os.evaluation_plan import build_evaluation_plan
    from alpha_os.evaluation_spec import (
        EvaluationDateRange,
        EvaluationFold,
        EvaluationSpec,
    )
    from alpha_os.strategy_checkpoint import StrategyCheckpoint
    from alpha_os.store import EvaluationStore
    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.upsert_strategy_checkpoint(
            state=StrategyCheckpoint(
                strategy_checkpoint_id="checkpoint_a",
                strategy_id="strategy_a",
                signal_discovery_id="discovery_a",
                subject_set_id="subject_set_a",
                target_id="residual_return_3d",
                fold_label="fold_2025",
                execution_start_date="2025-01-01",
                execution_end_date="2025-12-31",
                snapshot_set_id="snapshot_set_a",
                screening_result_id="screening_a",
                compressed_belief_id="belief_a",
                survivor_signal_ids=("h1", "h2"),
                created_at="2026-01-01T00:00:00Z",
            )
        )
        store.upsert_strategy_checkpoint(
            state=StrategyCheckpoint(
                strategy_checkpoint_id="checkpoint_b",
                strategy_id="strategy_a",
                signal_discovery_id="discovery_a",
                subject_set_id="subject_set_a",
                target_id="residual_return_3d",
                fold_label="fold_2026_q1",
                execution_start_date="2026-01-01",
                execution_end_date="2026-03-31",
                snapshot_set_id="snapshot_set_b",
                screening_result_id="screening_b",
                compressed_belief_id="belief_b",
                survivor_signal_ids=("h3", "h4"),
                created_at="2026-04-01T00:00:00Z",
            )
        )
        trading_strategy = _build_trading_strategy(
            strategy_id="strategy_a",
            label="Strategy A",
            signal_discovery_id="discovery_a",
            subject_set_id="subject_set_a",
            target_id="residual_return_3d",
            position_rule_id="signal_discovery",
            created_at="2026-01-01T00:00:00Z",
        )
        store.upsert_trading_strategy(trading_strategy=trading_strategy)

        evaluation_spec = EvaluationSpec(
            execution_range=EvaluationDateRange(
                label="compat_window",
                start_date="2025-01-01",
                end_date="2025-12-31",
            ),
            evaluation_folds=(
                EvaluationFold(
                    label="fold_2025",
                    execution_range=EvaluationDateRange(
                        label="train_2025",
                        start_date="2025-01-01",
                        end_date="2025-12-31",
                    ),
                    evaluation_date_ranges=(
                        EvaluationDateRange(
                            label="test_2026_q1",
                            start_date="2026-01-01",
                            end_date="2026-03-31",
                        ),
                    ),
                ),
                EvaluationFold(
                    label="fold_2026_q1",
                    execution_range=EvaluationDateRange(
                        label="train_2026_q1",
                        start_date="2026-01-01",
                        end_date="2026-03-31",
                    ),
                    evaluation_date_ranges=(
                        EvaluationDateRange(
                            label="test_2026_q2",
                            start_date="2026-04-01",
                            end_date="2026-06-30",
                        ),
                    ),
                ),
            ),
            metric_group_names=("decision_quality",),
            metric_windows=(20,),
        )
        evaluation_tasks = (
            EvaluationTask(
                evaluation_task_id="case_a",
                strategy_id="strategy_a",
                evaluation_spec_id="protocol_a",
                **_evaluation_policy_parts(),
            ),
        )

        plan = build_evaluation_plan(
            store,
            evaluation_spec_id="protocol_a",
            evaluation_spec=evaluation_spec,
            evaluation_tasks=evaluation_tasks,
            base_url="http://example.com",
        )

        assert len(plan.execution_requests) == 2
        assert tuple(item.fold_label for item in plan.execution_requests) == (
            "fold_2025",
            "fold_2026_q1",
        )
        assert tuple(item.input_refs.strategy_checkpoint_id for item in plan.execution_requests) == (
            "checkpoint_a",
            "checkpoint_b",
        )
        assert plan.execution_requests[0].evaluation_date_ranges[0].label == "test_2026_q1"
        assert plan.execution_requests[1].evaluation_date_ranges[0].label == "test_2026_q2"
    finally:
        store.close()


def test_build_evaluation_plan_uses_direct_strategy_without_discovery(
    tmp_path,
):
    from alpha_os.evaluation_task import EvaluationTask
    from alpha_os.evaluation_plan import build_evaluation_plan
    from alpha_os.evaluation_spec import (
        EvaluationDateRange,
        EvaluationFold,
        EvaluationSpec,
    )
    from alpha_os.store import EvaluationStore
    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        trading_strategy = _build_trading_strategy(
            strategy_id="strategy:nn_case",
            label="NN Case",
            subject_set_id="subject_set_a",
            target_id="residual_return_3d",
            position_rule_id="neural_model",
            created_at="2026-04-05T00:00:00Z",
        )
        store.upsert_trading_strategy(trading_strategy=trading_strategy)
        evaluation_spec = EvaluationSpec(
            execution_range=EvaluationDateRange(
                label="compat_window",
                start_date="2025-01-01",
                end_date="2025-12-31",
            ),
            evaluation_folds=(
                EvaluationFold(
                    label="fold_2025",
                    execution_range=EvaluationDateRange(
                        label="train_2025",
                        start_date="2025-01-01",
                        end_date="2025-12-31",
                    ),
                    evaluation_date_ranges=(
                        EvaluationDateRange(
                            label="test_2026_q1",
                            start_date="2026-01-01",
                            end_date="2026-03-31",
                        ),
                    ),
                ),
            ),
            metric_group_names=("decision_quality",),
            metric_windows=(20,),
        )
        evaluation_tasks = (
            EvaluationTask(
                evaluation_task_id="case_nn",
                strategy_id="strategy:nn_case",
                evaluation_spec_id="protocol_nn",
                **_evaluation_policy_parts(),
            ),
        )

        plan = build_evaluation_plan(
            store,
            evaluation_spec_id="protocol_nn",
            evaluation_spec=evaluation_spec,
            evaluation_tasks=evaluation_tasks,
            base_url="http://example.com",
        )

        assert len(plan.execution_requests) == 1
        request = plan.execution_requests[0]
        assert request.context.strategy_id == "strategy:nn_case"
        assert request.input_refs is None
    finally:
        store.close()


def test_build_evaluation_plan_rejects_direct_strategy_without_target(tmp_path):
    from alpha_os.evaluation_task import EvaluationTask
    from alpha_os.evaluation_plan import build_evaluation_plan
    from alpha_os.evaluation_spec import EvaluationDateRange, EvaluationSpec
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        trading_strategy = _build_trading_strategy(
            strategy_id="strategy:targetless",
            label="Targetless",
            subject_set_id="subject_set_a",
            target_id=None,
            position_rule_id="constant_hold",
            created_at="2026-04-05T00:00:00Z",
        )
        store.upsert_trading_strategy(trading_strategy=trading_strategy)
        evaluation_spec = EvaluationSpec(
            execution_range=EvaluationDateRange(
                label="eval",
                start_date="2026-01-01",
                end_date="2026-03-31",
            ),
            metric_group_names=("decision_quality",),
            metric_windows=(20,),
        )
        evaluation_task = EvaluationTask(
            evaluation_task_id="case:targetless",
            strategy_id="strategy:targetless",
            evaluation_spec_id="protocol_targetless",
            **_evaluation_policy_parts(),
        )

        with pytest.raises(
            ValueError,
            match="direct evaluation task requires strategy prediction target",
        ):
            build_evaluation_plan(
                store,
                evaluation_spec_id="protocol_targetless",
                evaluation_spec=evaluation_spec,
                evaluation_tasks=(evaluation_task,),
                base_url="http://example.com",
            )
    finally:
        store.close()


def test_build_evaluation_plan_keeps_strategy_portfolio_out_of_context(tmp_path):
    from alpha_os.evaluation_task import EvaluationTask
    from alpha_os.evaluation_plan import build_evaluation_plan
    from alpha_os.evaluation_spec import EvaluationDateRange, EvaluationSpec
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        trading_strategy = _build_trading_strategy(
            strategy_id="strategy:portfolio_source",
            label="Portfolio Source",
            subject_set_id="subject_set_a",
            target_id="residual_return_3d",
            sizing_method="equal_weight",
            rebalance="every_5_steps",
            long_only=True,
            top_k=3,
            gross_exposure_cap=0.8,
            created_at="2026-04-05T00:00:00Z",
        )
        store.upsert_trading_strategy(trading_strategy=trading_strategy)
        evaluation_spec = EvaluationSpec(
            execution_range=EvaluationDateRange(
                label="eval",
                start_date="2026-01-01",
                end_date="2026-03-31",
            ),
            metric_group_names=("decision_quality",),
            metric_windows=(20,),
        )
        evaluation_task = EvaluationTask(
            evaluation_task_id="case:portfolio_source",
            strategy_id="strategy:portfolio_source",
            evaluation_spec_id="protocol_portfolio_source",
            **_evaluation_policy_parts(
                sizing_method="equal_weight",
                sizing_engine="history_based",
                rebalance_interval_steps=1,
                long_only=False,
                top_k=1,
                gross_exposure_cap=1.5,
                target_vol=0.12,
                gross_leverage_cap=1.4,
                net_exposure_target=0.0,
            ),
        )

        plan = build_evaluation_plan(
            store,
            evaluation_spec_id="protocol_portfolio_source",
            evaluation_spec=evaluation_spec,
            evaluation_tasks=(evaluation_task,),
            base_url="http://example.com",
        )

        request = plan.execution_requests[0]
        assert request.context.strategy_id == "strategy:portfolio_source"
        assert request.context.target_id == "residual_return_3d"
        assert plan.execution_requests[0].input_refs is None
    finally:
        store.close()


def test_build_evaluation_plan_supports_strategy_checkpoint_replay(tmp_path):
    from alpha_os.evaluation_task import EvaluationTask
    from alpha_os.evaluation_plan import build_evaluation_plan
    from alpha_os.evaluation_spec import (
        EvaluationDateRange,
        EvaluationFold,
        EvaluationSpec,
    )
    from alpha_os.strategy_checkpoint import StrategyCheckpoint
    from alpha_os.store import EvaluationStore
    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        trading_strategy = _build_trading_strategy(
            strategy_id="strategy:checkpoint_case",
            label="Checkpoint Case",
            signal_discovery_id="discovery_a",
            subject_set_id="subject_set_a",
            target_id="residual_return_3d",
            created_at="2026-04-05T00:00:00Z",
        )
        store.upsert_trading_strategy(trading_strategy=trading_strategy)
        store.upsert_strategy_checkpoint(
            state=StrategyCheckpoint(
                strategy_checkpoint_id="checkpoint_a",
                strategy_id="strategy:checkpoint_case",
                signal_discovery_id="discovery_a",
                subject_set_id="subject_set_a",
                target_id="residual_return_3d",
                fold_label="source_fold",
                execution_start_date="2025-01-01",
                execution_end_date="2025-12-31",
                snapshot_set_id="snapshot_set_seed",
                screening_result_id="screening_seed",
                compressed_belief_id="belief_seed",
                survivor_signal_ids=("h1", "h2"),
                created_at="2026-04-05T00:00:00Z",
            )
        )
        evaluation_spec = EvaluationSpec(
            execution_range=EvaluationDateRange(
                label="compat_window",
                start_date="2025-01-01",
                end_date="2025-12-31",
            ),
            evaluation_folds=(
                EvaluationFold(
                    label="fold_2025",
                    execution_range=EvaluationDateRange(
                        label="train_2025",
                        start_date="2025-01-01",
                        end_date="2025-12-31",
                    ),
                    evaluation_date_ranges=(
                        EvaluationDateRange(
                            label="test_2026_q1",
                            start_date="2026-01-01",
                            end_date="2026-03-31",
                        ),
                    ),
                ),
            ),
            metric_group_names=("decision_quality",),
            metric_windows=(20,),
        )
        evaluation_tasks = (
            EvaluationTask(
                evaluation_task_id="case_checkpoint",
                strategy_id="strategy:checkpoint_case",
                evaluation_spec_id="protocol_checkpoint",
                **_evaluation_policy_parts(),
            ),
        )

        plan = build_evaluation_plan(
            store,
            evaluation_spec_id="protocol_checkpoint",
            evaluation_spec=evaluation_spec,
            evaluation_tasks=evaluation_tasks,
            base_url="http://example.com",
        )

        assert len(plan.execution_requests) == 1
        assert tuple(item.fold_label for item in plan.execution_requests) == ("fold_2025",)
        assert tuple(item.input_refs.strategy_checkpoint_id for item in plan.execution_requests) == (
            "checkpoint_a",
        )
    finally:
        store.close()


def test_run_walk_forward_evaluation_executes_fold_runs(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.evaluation_task import EvaluationTask, build_evaluation_task_id
    from alpha_os.store import EvaluationStore
    db_path = tmp_path / "runtime.db"
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "observables": [
                    {
                        "observable_id": "daily_close",
                        "family": "price",
                        "value_kind": "real_value",
                        "default_resolution": "1d",
                    }
                ],
                "signal_specs": [
                    {
                        "signal_id": "reversal_1d",
                        "kind": "reversal",
                        "required_observable_id": "daily_close",
                        "target_definition": {
                            "target_id": "residual_return_3d",
                            "family": "residual_return",
                            "observation_kind": "realized_return",
                            "subject_kind": "asset",
                            "output_kind": "real_value",
                            "scoring_kind": "corr",
                            "horizon_days": 3,
                            "params": {},
                        },
                        "params": {"lookback": 1},
                    },
                    {
                        "signal_id": "reversal_3d",
                        "kind": "reversal",
                        "required_observable_id": "daily_close",
                        "target_definition": {
                            "target_id": "residual_return_3d",
                            "family": "residual_return",
                            "observation_kind": "realized_return",
                            "subject_kind": "asset",
                            "output_kind": "real_value",
                            "scoring_kind": "corr",
                            "horizon_days": 3,
                            "params": {},
                        },
                        "params": {"lookback": 3},
                    },
                ],
                "subject_sets": [
                    {
                        "subject_set_id": "core_crypto",
                        "observation_specs": [
                            {
                                "observation_spec_id": "btc_close",
                                "observable_id": "daily_close",
                            }
                        ],
                        "bindings": [
                            {
                                "subject_id": "BTC_spot",
                                "subject_kind": "asset",
                                "asset": "BTC",
                                "observation_spec_id": "btc_close",
                            }
                        ],
                    }
                ],
                "signal_discoveries": [
                    {
                        "signal_discovery_id": "core_crypto_search",
                        "subject_set_id": "core_crypto",
                        "selection_policy": {
                            "min_sample_count": 1,
                            "min_abs_corr": 0.0,
                            "probe_max_dates": 3,
                            "probe_min_sample_count": 2,
                            "probe_min_abs_corr": 0.0,
                            "probe_max_family_survivors_per_subject": 1,
                            "survivor_min_sample_count": 2,
                            "survivor_min_abs_corr": 0.0,
                            "survivor_max_family_survivors_per_subject": 1,
                            "snapshot_retention": "latest_per_survivor",
                            "max_family_survivors_per_subject": 2,
                        },
                        "families": [
                            {
                                "family_id": "reversal_family",
                                "kind": "reversal",
                                "parameter_space": {
                                    "lookback": [1, 3],
                                },
                                "required_observable_id": "daily_close",
                                "target_id": "residual_return_3d",
                                "survivor_budget": 1,
                            }
                        ],
                        "target_id": "residual_return_3d",
                    }
                ],
                "evaluation_specs": [
                    {
                        "evaluation_spec_id": "core_crypto_walk_forward",
                        "execution_range": {
                            "label": "compat_window",
                            "start_date": "2026-03-23",
                            "end_date": "2026-03-24",
                        },
                        "evaluation_folds": [
                            {
                                "label": "fold_a",
                                "execution_range": {
                                    "label": "train_a",
                                    "start_date": "2026-03-23",
                                    "end_date": "2026-03-24",
                                },
                                "evaluation_date_ranges": [
                                    {
                                        "label": "test_a",
                                        "start_date": "2026-03-25",
                                        "end_date": "2026-03-26",
                                    }
                                ],
                            },
                            {
                                "label": "fold_b",
                                "execution_range": {
                                    "label": "train_b",
                                    "start_date": "2026-03-24",
                                    "end_date": "2026-03-25",
                                },
                                "evaluation_date_ranges": [
                                    {
                                        "label": "test_b",
                                        "start_date": "2026-03-26",
                                        "end_date": "2026-03-27",
                                    }
                                ],
                            },
                        ],
                        "metric_windows": [2],
                        "metric_group_names": [
                            "signed_belief_quality",
                            "portfolio_target_return_alignment",
                            "sizing_policy_quality",
                            "rebalance_policy_quality",
                            "decision_quality",
                            "robustness",
                        ],
                    }
                ],
                "evaluation_tasks": [
                    {
                        "evaluation_spec_id": "core_crypto_walk_forward",
                        "signal_discovery_id": "core_crypto_search",
                        "base_url": "http://example.com",
                        "portfolio_construction": {
                            "sizing_policy": {"sizing_method": "signal_weighted", "sizing_engine": "rule_based"}
                        },
                        "rebalance_friction_policy": {},
                        "execution_cost_assumptions": {},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    def _fake_loader(observation_spec, *, asset: str, base_url: str, client=None):
        import pandas as pd

        assert asset == "BTC"
        return pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-20T00:00:00Z",
                    "2026-03-21T00:00:00Z",
                    "2026-03-22T00:00:00Z",
                    "2026-03-23T00:00:00Z",
                    "2026-03-24T00:00:00Z",
                    "2026-03-25T00:00:00Z",
                    "2026-03-26T00:00:00Z",
                    "2026-03-27T00:00:00Z",
                    "2026-03-28T00:00:00Z",
                    "2026-03-29T00:00:00Z",
                    "2026-03-30T00:00:00Z",
                ],
                "value": [
                    100.0,
                    101.0,
                    103.0,
                    102.0,
                    104.0,
                    105.0,
                    107.0,
                    106.0,
                    108.0,
                    109.0,
                    111.0,
                ],
            }
        )

    import alpha_os.data_repositories as data_repositories

    original_loader = data_repositories.load_observation_frame
    data_repositories.load_observation_frame = _fake_loader
    try:
        assert (
            main(
                [
                    "run-walk-forward-evaluation",
                    "--db",
                    str(db_path),
                    "--evaluation-spec-id",
                    "core_crypto_walk_forward",
                    "--base-url",
                    "http://example.com",
                ]
            )
            == 0
        )
    finally:
        data_repositories.load_observation_frame = original_loader

    output = capsys.readouterr().out
    assert "alpha-os evaluation run" in output
    assert "TaskResults: 2" in output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        strategy_checkpoints = store.list_strategy_checkpoints(
            signal_discovery_id="core_crypto_search",
            limit=10,
        )
        assert len(strategy_checkpoints) >= 2
        assert {item.state.fold_label for item in strategy_checkpoints} >= {
            "fold_a",
            "fold_b",
        }
        report_state = store.get_latest_evaluation_report()
        assert report_state is not None
        assert len(report_state.report.task_results) == 2
        for task_result in report_state.report.task_results:
            assert task_result.artifact_refs.get("strategy_checkpoint_ids")
            decision_metric_group_result = next(
                item
                for item in task_result.metric_group_results
                if item.metric_group_name == "decision_quality"
            )
            assert decision_metric_group_result.metrics["total_decision_step_count"] > 0
        evaluation_tasks = store.list_evaluation_tasks(limit=10)
        assert len(evaluation_tasks) == 1
        evaluation_task_state = evaluation_tasks[0]
        strategy_specs = store.list_trading_strategies(limit=10)
        assert len(strategy_specs) == 1
        trading_strategy = strategy_specs[0].trading_strategy
        assert trading_strategy.strategy_id.startswith("strategy:")
        strategy_checkpoints_for_strategy = store.list_strategy_checkpoints(
            strategy_id=trading_strategy.strategy_id,
            limit=10,
        )
        assert len(strategy_checkpoints_for_strategy) >= 2
        assert all(
            item.state.strategy_id == trading_strategy.strategy_id
            for item in strategy_checkpoints_for_strategy
        )
        strategy_checkpoints_for_signal_discovery = store.list_strategy_checkpoints(
            strategy_id=trading_strategy.strategy_id,
            signal_discovery_id=trading_strategy.signal_discovery_id,
            limit=10,
        )
        assert len(strategy_checkpoints_for_signal_discovery) >= 2
        assert evaluation_task_state.task.strategy_id == trading_strategy.strategy_id
        assert (
            evaluation_task_state.task.evaluation_spec_id
            == "core_crypto_walk_forward"
        )
        assert trading_strategy.scope.subject_set_id == "core_crypto"
        assert trading_strategy.portfolio.portfolio_construction.sizing_method == (
            "signal_weighted"
        )
        assert trading_strategy.portfolio.rebalance_interval_steps == 1
        strategy_checkpoint_count = len(strategy_checkpoints)
    finally:
        store.close()

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        evaluation_task_state = store.list_evaluation_tasks(limit=10)[0]
        challenger_strategy = _build_trading_strategy(
            strategy_id="strategy:core_crypto_equal_weight",
            label="core_crypto_equal_weight",
            signal_discovery_id="core_crypto_search",
            subject_set_id="core_crypto",
            target_id="residual_return_3d",
            family_mix="reversal",
            sizing_method="equal_weight",
            rebalance="every_1_steps",
            long_only=True,
            gross_exposure_cap=1.0,
            market_impact_bps=0.0,
            turnover_friction=0.0,
            no_trade_band=0.0,
            created_at="2026-04-05T00:00:00+00:00",
        )
        store.upsert_trading_strategy(trading_strategy=challenger_strategy)
        challenger_task = EvaluationTask(
            evaluation_task_id=build_evaluation_task_id(
                strategy_id=challenger_strategy.strategy_id,
                evaluation_spec_id="core_crypto_walk_forward",
            ),
            strategy_id=challenger_strategy.strategy_id,
            evaluation_spec_id="core_crypto_walk_forward",
            **_evaluation_policy_parts(
                sizing_method="equal_weight",
                sizing_engine="history_based",
            ),
        )
        store.upsert_evaluation_task(task=challenger_task)
    finally:
        store.close()

    import alpha_os.data_repositories as data_repositories

    original_loader = data_repositories.load_observation_frame
    data_repositories.load_observation_frame = _fake_loader
    try:
        assert (
            main(
                [
                    "run-walk-forward-evaluation",
                    "--db",
                    str(db_path),
                    "--evaluation-spec-id",
                    "core_crypto_walk_forward",
                    "--base-url",
                    "http://example.com",
                ]
            )
            == 0
        )
    finally:
        data_repositories.load_observation_frame = original_loader

    capsys.readouterr()

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        rerun_strategy_checkpoints = store.list_strategy_checkpoints(
            signal_discovery_id="core_crypto_search",
            limit=10,
        )
        assert len(rerun_strategy_checkpoints) == strategy_checkpoint_count + 2
        challenger_strategy_checkpoints = store.list_strategy_checkpoints(
            strategy_id="strategy:core_crypto_equal_weight",
            limit=10,
        )
        assert len(challenger_strategy_checkpoints) == 2
    finally:
        store.close()
