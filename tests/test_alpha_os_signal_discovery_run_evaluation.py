from __future__ import annotations

import json
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
    trading_environment: dict[str, object] | None = None,
) -> dict[str, object]:
    document: dict[str, object] = {
        "portfolio_construction": {
            "sizing_policy": {"sizing_method": sizing_method},
            "direction_mode": direction_mode,
            "gross_exposure_cap": gross_exposure_cap,
        },
        "trading_environment": ({} if trading_environment is None else trading_environment),
        "rebalance_interval_steps": rebalance_interval_steps,
        "selection_kind": selection_kind,
    }
    if top_k is not None:
        document["top_k"] = top_k
    return document


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
    turnover_cost_rate: float | None = None,
    created_at: str = "2026-04-05T00:00:00Z",
):
    from alpha_os.evaluation_cost_config import TradingEnvironment
    from alpha_os.trading_strategy import TradingStrategySpec
    from alpha_os.portfolio_construction_config import (
        PortfolioConstructionSizingSpec,
        PortfolioConstructionSpec,
    )

    return TradingStrategySpec(
        strategy_id=strategy_id,
        label=label,
        subject_set_id=subject_set_id,
        target_id=target_id,
        signal_discovery_id=signal_discovery_id,
        position_rule_id=position_rule_id,
        family_mix=family_mix,
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
        trading_environment=TradingEnvironment(
            turnover_cost_rate=(0.0 if turnover_cost_rate is None else turnover_cost_rate),
            market_impact_bps=0.0 if market_impact_bps is None else market_impact_bps,
            fee_bps=0.0 if fee_bps is None else fee_bps,
            bid_ask_spread_bps=(0.0 if bid_ask_spread_bps is None else bid_ask_spread_bps),
        ),
        created_at=created_at,
        rebalance_interval_steps=(
            int(rebalance[len("every_") : -len("_steps")])
            if isinstance(rebalance, str)
            and rebalance.startswith("every_")
            and rebalance.endswith("_steps")
            else 1
        ),
        selection_kind="all_assets",
        top_k=top_k,
    )


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
            ((frame["return_7d"] > 0.0) & (frame["return_30d"] > 0.0) & ~funding_overheated)
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
        TradingEnvironment,
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
            get_trading_strategy=lambda strategy_id: SimpleNamespace(trading_strategy=strategy),
            get_subject_set=lambda subject_set_id: SimpleNamespace(definition=subject_set),
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
        trading_environment=TradingEnvironment(),
        feature_plane_repository=None,
    )

    signal_series_by_subject = captured["signal_series_by_subject"]
    assert signal_series_by_subject["BTC"].loc["2026-01-29"] == pytest.approx(0.0)
    assert signal_series_by_subject["BTC"].loc["2026-01-30"] == pytest.approx(1.0)
    assert captured["funding_cost_bps_series_by_subject"]["BTC"].iloc[0] == (pytest.approx(10.0))


def test_run_walk_forward_evaluates_signal_discovery_derived_direct_strategy(tmp_path, capsys):
    from alpha_os.cli import main

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
                        "strategy_ids": [
                            "strategy:core_crypto_rule",
                        ],
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
                "strategy_specs": [
                    {
                        "trading_strategy": {
                            "strategy_id": "strategy:core_crypto_rule",
                            "label": "Core Crypto Rule",
                            "subject_set_id": "core_crypto",
                            "target_id": "residual_return_3d",
                            "signal_discovery_id": "core_crypto_search",
                            "position_rule_id": "constant_hold",
                            "family_mix": None,
                            **_strategy_portfolio_document(
                                sizing_method="signal_weighted",
                                direction_mode="long_short",
                                gross_exposure_cap=None,
                                trading_environment={
                                    "market_impact_bps": 0.0,
                                    "fee_bps": 0.0,
                                    "bid_ask_spread_bps": 0.0,
                                },
                            ),
                            "created_at": "2026-04-05T00:00:00+00:00",
                        }
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

    output = capsys.readouterr().out
    assert "alpha-os evaluation run" in output
    assert "Results: 1" in output


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
                        "signal_spec_ids": ["reversal_1d"],
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
                            "subject_set_id": "core_crypto",
                            "target_id": "residual_return_3d",
                            "signal_discovery_id": "core_crypto_search",
                            "position_rule_id": "constant_hold",
                            "family_mix": "spec:-",
                            **_strategy_portfolio_document(
                                sizing_method="signal_weighted",
                                direction_mode="long_short",
                                gross_exposure_cap=None,
                                trading_environment={
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
                        "strategy_ids": [
                            "strategy:core_crypto_rule",
                        ],
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
                        "signal_spec_ids": ["reversal_1d"],
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
                            "subject_set_id": "core_crypto",
                            "target_id": "residual_return_3d",
                            "signal_discovery_id": "core_crypto_search",
                            "position_rule_id": "constant_hold",
                            "family_mix": "spec:-",
                            **_strategy_portfolio_document(
                                sizing_method="signal_weighted",
                                direction_mode="long_short",
                                gross_exposure_cap=None,
                                trading_environment={
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
                        "strategy_ids": [
                            "strategy:core_crypto_rule",
                        ],
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
        assert trading_strategy.subject_set_id == "core_crypto"
        assert trading_strategy.signal_discovery_id == "core_crypto_search"
        assert trading_strategy.portfolio_construction.sizing_method == (
            "signal_weighted"
        )
    finally:
        store.close()


def test_apply_runtime_manifest_accepts_search_free_evaluation_case(tmp_path, capsys):
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
                            "subject_set_id": "broad_9_etf",
                            "target_id": None,
                            "signal_discovery_id": None,
                            "position_rule_id": "constant_hold",
                            "family_mix": None,
                            **_strategy_portfolio_document(
                                sizing_method="equal_weight",
                                direction_mode=None,
                                gross_exposure_cap=None,
                                trading_environment={
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
                        "strategy_ids": [
                            "strategy:buy_and_hold",
                        ],
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
        strategy_state = store.get_trading_strategy("strategy:buy_and_hold")
        assert strategy_state is not None
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
                            "subject_set_id": "core_crypto",
                            "target_id": "residual_return_3d",
                            "signal_discovery_id": None,
                            "position_rule_id": "constant_hold",
                            "family_mix": None,
                            **_strategy_portfolio_document(
                                sizing_method="equal_weight",
                                direction_mode=None,
                                gross_exposure_cap=None,
                                trading_environment={
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
                        "strategy_ids": [
                            "strategy:buy_and_hold",
                        ],
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
    assert "Results: 1" in output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        run_result_state = store.get_latest_evaluation_run_result()
        assert run_result_state is not None
        assert len(run_result_state.run_result.results) == 1
        result = next(iter(run_result_state.run_result.results.values()))
        decision_metric_group_result = next(
            item
            for item in result.metric_group_results
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
                            "subject_set_id": "core_crypto_top_k",
                            "target_id": "residual_return_3d",
                            "signal_discovery_id": None,
                            "position_rule_id": "constant_hold",
                            "family_mix": None,
                            **_strategy_portfolio_document(
                                sizing_method="equal_weight",
                                direction_mode="long_only",
                                gross_exposure_cap=1.0,
                                selection_kind="top_k",
                                top_k=1,
                                trading_environment={
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
                        "strategy_ids": [
                            "strategy:top_k_hold",
                        ],
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
    assert "Results: 1" in output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        run_result_state = store.get_latest_evaluation_run_result()
        assert run_result_state is not None
        result = next(iter(run_result_state.run_result.results.values()))
        decision_metric_group_result = next(
            item
            for item in result.metric_group_results
            if item.metric_group_name == "decision_quality"
        )
        assert decision_metric_group_result.metrics["total_decision_step_count"] > 0
    finally:
        store.close()


def test_run_walk_forward_evaluation_executes_trainless_dual_momentum_strategy(tmp_path, capsys):
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
                            "subject_set_id": "core_crypto_dual_momentum",
                            "target_id": "residual_return_3d",
                            "signal_discovery_id": None,
                            "position_rule_id": "dual_momentum_hold",
                            "family_mix": "lookback=2",
                            **_strategy_portfolio_document(
                                sizing_method="signal_weighted",
                                direction_mode="long_only",
                                gross_exposure_cap=1.0,
                                selection_kind="top_k",
                                top_k=1,
                                trading_environment={
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
                        "strategy_ids": [
                            "strategy:dual_momentum_hold",
                        ],
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
    assert "Results: 1" in output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        run_result_state = store.get_latest_evaluation_run_result()
        assert run_result_state is not None
        result = next(iter(run_result_state.run_result.results.values()))
        decision_metric_group_result = next(
            item
            for item in result.metric_group_results
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
    assert "Results: 1" in output


def test_run_walk_forward_evaluation_executes_signal_discovery_derived_direct_strategy(
    tmp_path, capsys
):
    from alpha_os.cli import main

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
                        "strategy_ids": [
                            "strategy:core_crypto_rule",
                        ],
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
                "strategy_specs": [
                    {
                        "trading_strategy": {
                            "strategy_id": "strategy:core_crypto_rule",
                            "label": "Core Crypto Rule",
                            "subject_set_id": "core_crypto",
                            "target_id": "residual_return_3d",
                            "signal_discovery_id": "core_crypto_search",
                            "position_rule_id": "constant_hold",
                            "family_mix": None,
                            **_strategy_portfolio_document(
                                sizing_method="signal_weighted",
                                direction_mode="long_short",
                                gross_exposure_cap=None,
                                trading_environment={
                                    "market_impact_bps": 0.0,
                                    "fee_bps": 0.0,
                                    "bid_ask_spread_bps": 0.0,
                                },
                            ),
                            "created_at": "2026-04-05T00:00:00+00:00",
                        }
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
    assert "Results: 2" in output
