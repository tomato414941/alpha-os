from __future__ import annotations

import json
import pytest


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
    created_at: str = "2026-03-29T00:00:00+00:00",
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
                direction_mode=None,
                gross_exposure_cap=None,
            ),
            rebalance_friction_policy=RebalanceFrictionPolicySpec(
                turnover_friction=None,
                no_trade_band=None,
            ),
            execution_policy=ExecutionPolicySpec(
                market_impact_bps=None,
                fee_bps=None,
                bid_ask_spread_bps=None,
            ),
            holding_cost_policy=HoldingCostPolicySpec(),
            selection_kind="all_assets",
            top_k=None,
        ),
        created_at=created_at,
    )


def _register_singleton_subject_set(store, *, subject_set_id: str = "core_crypto") -> None:
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
    )

    store.upsert_subject_set(
        subject_set_id,
        definition=SubjectSet(
            subject_set_id=subject_set_id,
            observation_specs=(
                ObservationSpec(
                    observation_spec_id="btc_close",
                    observable_id="daily_close",
                ),
            ),
            bindings=(
                SubjectObservationBinding(
                    subject_id="BTC_spot",
                    asset="BTC",
                    observation_spec_id="btc_close",
                ),
            ),
        ),
    )


def _register_search_strategy(
    store,
    *,
    strategy_id: str,
    signal_discovery_id: str,
    subject_set_id: str = "core_crypto",
) -> None:
    spec = _build_trading_strategy(
        strategy_id=strategy_id,
        label=strategy_id,
        subject_set_id=subject_set_id,
        target_id=None,
        signal_discovery_id=signal_discovery_id,
        position_rule_id="signal_discovery",
    )
    store.upsert_trading_strategy(trading_strategy=spec)


def _register_default_validation_signal_specs(store) -> None:
    for signal_id in ("momentum_1d", "reversal_1d", "reversal_3d"):
        store.register_signal_spec(signal_id=signal_id)


def test_write_and_load_default_validation_spec(tmp_path):
    from alpha_os.validation_spec import (
        default_validation_spec,
        load_validation_spec,
        write_validation_spec,
    )

    path = tmp_path / "validation_spec.json"
    expected = default_validation_spec(subject_set_ids=("core_crypto",))
    write_validation_spec(path, expected)
    loaded = load_validation_spec(path)

    assert loaded == expected


def test_build_validation_plan_for_strategies_uses_strategy_scope(tmp_path):
    from alpha_os.signal_discovery import (
        SignalFamily,
        SignalParameterSpace,
        SignalDiscoverySpec,
    )
    from alpha_os.signal_discovery_execution import build_signal_discovery_execution_plan
    from alpha_os.store import EvaluationStore
    from alpha_os.validation_service import build_validation_plan_for_strategies
    from alpha_os.validation_spec import ValidationDateRange, ValidationSpec

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()
    _register_default_validation_signal_specs(store)
    _register_singleton_subject_set(store)
    store.upsert_signal_discovery_spec(
        "core_crypto_search",
        definition=SignalDiscoverySpec(
            signal_discovery_id="core_crypto_search",
            subject_set_id="core_crypto",
            families=(
                SignalFamily(
                    family_id="reversal_family",
                    kind="reversal",
                    parameter_space=SignalParameterSpace.from_document(
                        {"lookback": [1, 3]}
                    ),
                    target_id="residual_return_3d",
                ),
            ),
            target_id="residual_return_3d",
        )
    )
    _register_search_strategy(
        store,
        strategy_id="strategy:core_crypto_search",
        signal_discovery_id="core_crypto_search",
    )
    base_spec = ValidationSpec(
        signal_ids=("momentum_1d",),
        target_ids=("residual_return_1d",),
        date_ranges=(
            ValidationDateRange(
                label="mini",
                start_date="2026-03-22",
                end_date="2026-03-22",
            ),
        ),
        metric_windows=(20,),
        aggregation_kinds=("active_equal_mean",),
        subject_set_ids=("ignored_scope",),
        base_url="http://example.com",
    )

    plan = build_validation_plan_for_strategies(
        store,
        strategy_ids=("strategy:core_crypto_search",),
        base_spec=base_spec,
        base_url="http://override.example.com",
    )

    assert plan.strategy_ids == ("strategy:core_crypto_search",)
    assert len(plan.entries) == 1
    assert plan.entries[0].strategy_id == "strategy:core_crypto_search"
    assert plan.entries[0].signal_discovery_id == "core_crypto_search"
    assert plan.entries[0].subject_set_id == "core_crypto"
    assert set(plan.entries[0].signal_ids) == {"reversal_1d", "reversal_3d"}
    assert plan.spec.subject_set_ids == ("core_crypto",)
    assert set(plan.spec.signal_ids) == {"reversal_1d", "reversal_3d"}
    assert plan.spec.target_ids == ("residual_return_3d",)
    assert plan.spec.base_url == "http://override.example.com"

    execution_plan = build_signal_discovery_execution_plan(
        store,
        signal_discovery_id="core_crypto_search",
        default_target_id="residual_return_1d",
    )
    assert execution_plan.signal_discovery.signal_discovery_id == "core_crypto_search"
    assert execution_plan.subject_set.subject_set_id == "core_crypto"
    assert set(execution_plan.signal_spec_ids) == {
        "reversal_1d",
        "reversal_3d",
    }
    assert execution_plan.target_id == "residual_return_3d"


def test_signal_discovery_execution_plan_rejects_incomplete_universe_policy(tmp_path):
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
    )
    from alpha_os.signal_discovery import (
        SignalFamily,
        SignalParameterSpace,
        SignalDiscoverySpec,
    )
    from alpha_os.signal_discovery_execution import build_signal_discovery_execution_plan
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()
    _register_default_validation_signal_specs(store)
    store.upsert_subject_set(
        "core_crypto",
        definition=SubjectSet(
            subject_set_id="core_crypto",
            observation_specs=(
                ObservationSpec(
                    observation_spec_id="btc_close",
                    observable_id="daily_close",
                ),
                ObservationSpec(
                    observation_spec_id="eth_close",
                    observable_id="daily_close",
                ),
            ),
            bindings=(
                SubjectObservationBinding(
                    subject_id="BTC_spot",
                    asset="BTC",
                    observation_spec_id="btc_close",
                ),
                SubjectObservationBinding(
                    subject_id="ETH_spot",
                    asset="ETH",
                    observation_spec_id="eth_close",
                ),
            ),
        ),
    )
    store.upsert_signal_discovery_spec(
        "core_crypto_search",
        definition=SignalDiscoverySpec(
            signal_discovery_id="core_crypto_search",
            subject_set_id="core_crypto",
            families=(
                SignalFamily(
                    family_id="reversal_family",
                    kind="reversal",
                    parameter_space=SignalParameterSpace.from_document(
                        {"lookback": [1, 3]}
                    ),
                    target_id="residual_return_3d",
                ),
            ),
            target_id="residual_return_3d",
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        build_signal_discovery_execution_plan(
            store,
            signal_discovery_id="core_crypto_search",
            default_target_id="residual_return_1d",
        )

    assert (
        "subject set universe policy is incomplete for multi-subject validation/evaluation: "
        "core_crypto missing base_currency, trading_calendar, benchmark_id"
        in str(excinfo.value)
    )


def test_run_validation_persists_results(tmp_path):
    from alpha_os.store import EvaluationStore
    from alpha_os.validation_service import run_validation
    from alpha_os.validation_spec import ValidationDateRange, ValidationSpec
    import alpha_os.validation_service as validation_service

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()
    _register_default_validation_signal_specs(store)
    _register_singleton_subject_set(store)

    frame = {
        "timestamp": [
            "2026-03-20T00:00:00Z",
            "2026-03-21T00:00:00Z",
            "2026-03-22T00:00:00Z",
            "2026-03-23T00:00:00Z",
            "2026-03-24T00:00:00Z",
            "2026-03-25T00:00:00Z",
        ],
        "value": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
    }

    def _fake_loader(*, base_url: str, asset: str, observation_spec):
        import pandas as pd

        return pd.DataFrame(frame)

    original_loader = validation_service._load_price_frame_from_signal_noise
    validation_service._load_price_frame_from_signal_noise = _fake_loader
    try:
        spec = ValidationSpec(
            signal_ids=("momentum_1d", "reversal_1d"),
            target_ids=("residual_return_1d", "residual_return_3d"),
            date_ranges=(
                ValidationDateRange(
                    label="mini",
                    start_date="2026-03-22",
                    end_date="2026-03-22",
                ),
            ),
            metric_windows=(20,),
            aggregation_kinds=("active_equal_mean", "corr_weighted_mean"),
            subject_set_ids=("core_crypto",),
            base_url="http://example.com",
        )
        result = run_validation(store, spec=spec, recorded_at="2026-03-29T00:00:00+00:00")
    finally:
        validation_service._load_price_frame_from_signal_noise = original_loader
        store.close()

    store = EvaluationStore(db_path)
    try:
        run = store.get_validation_run(result.run_id)
        assert run is not None
        assert json.loads(run.spec_json)["target_ids"] == [
            "residual_return_1d",
            "residual_return_3d",
        ]
        signal_results = store.list_validation_signal_results(run_id=result.run_id)
        meta_results = store.list_validation_meta_results(run_id=result.run_id)
        decision_results = store.list_validation_decision_results(run_id=result.run_id)

        assert len(signal_results) == 4
        assert len(meta_results) == 4
        assert len(decision_results) == 4
        assert {item.subject_set_id for item in decision_results} == {"core_crypto"}
        assert {item.target_id for item in signal_results} == {
            "residual_return_1d",
            "residual_return_3d",
        }
        assert {item.aggregation_kind for item in meta_results} == {
            "active_equal_mean",
            "corr_weighted_mean",
        }
    finally:
        store.close()


def test_run_validation_clips_unavailable_tail_dates(tmp_path):
    from alpha_os.store import EvaluationStore
    from alpha_os.validation_service import run_validation
    from alpha_os.validation_spec import ValidationDateRange, ValidationSpec
    import alpha_os.validation_service as validation_service

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()
    _register_default_validation_signal_specs(store)
    _register_singleton_subject_set(store)

    frame = {
        "timestamp": [
            "2026-03-20T00:00:00Z",
            "2026-03-21T00:00:00Z",
            "2026-03-22T00:00:00Z",
            "2026-03-23T00:00:00Z",
            "2026-03-24T00:00:00Z",
            "2026-03-25T00:00:00Z",
        ],
        "value": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
    }

    def _fake_loader(*, base_url: str, asset: str, observation_spec):
        import pandas as pd

        return pd.DataFrame(frame)

    original_loader = validation_service._load_price_frame_from_signal_noise
    validation_service._load_price_frame_from_signal_noise = _fake_loader
    try:
        spec = ValidationSpec(
            signal_ids=("momentum_1d",),
            target_ids=("residual_return_3d",),
            date_ranges=(
                ValidationDateRange(
                    label="tail_clipped",
                    start_date="2026-03-20",
                    end_date="2026-03-25",
                ),
            ),
            metric_windows=(20,),
            aggregation_kinds=("active_equal_mean",),
            subject_set_ids=("core_crypto",),
            base_url="http://example.com",
        )
        result = run_validation(store, spec=spec, recorded_at="2026-03-29T00:00:00+00:00")
    finally:
        validation_service._load_price_frame_from_signal_noise = original_loader
        store.close()

    store = EvaluationStore(db_path)
    try:
        signal_results = store.list_validation_signal_results(run_id=result.run_id)
        assert len(signal_results) == 1
        assert signal_results[0].sample_count == 2
    finally:
        store.close()


def test_screen_signal_discovery_rejects_incomplete_universe_policy(tmp_path):
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
        UniversePolicySpec,
    )
    from alpha_os.signal_discovery_screening_service import screen_signal_discovery
    from alpha_os.signal_discovery import SignalDiscoverySpec
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()
    store.upsert_subject_set(
        "macro_pair",
        definition=SubjectSet(
            subject_set_id="macro_pair",
            observation_specs=(
                ObservationSpec(
                    observation_spec_id="btc_close",
                    observable_id="daily_close",
                ),
                ObservationSpec(
                    observation_spec_id="eth_close",
                    observable_id="daily_close",
                ),
            ),
            bindings=(
                SubjectObservationBinding(
                    subject_id="BTC_spot",
                    asset="BTC",
                    observation_spec_id="btc_close",
                ),
                SubjectObservationBinding(
                    subject_id="ETH_spot",
                    asset="ETH",
                    observation_spec_id="eth_close",
                ),
            ),
            universe_policy=UniversePolicySpec(
                base_currency="USD",
                trading_calendar=None,
                benchmark_id=None,
            ),
        ),
    )
    store.upsert_signal_discovery_spec(
        "macro_pair_search",
        definition=SignalDiscoverySpec(
            signal_discovery_id="macro_pair_search",
            subject_set_id="macro_pair",
            signal_spec_ids=("momentum_1d",),
            target_id="residual_return_1d",
        ),
    )

    try:
        with pytest.raises(ValueError, match="subject set universe policy is incomplete"):
            screen_signal_discovery(
                store,
                signal_discovery_id="macro_pair_search",
            )
    finally:
        store.close()


def test_run_validation_persists_decision_results_for_subject_set(tmp_path):
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
        UniversePolicySpec,
    )
    from alpha_os.store import EvaluationStore
    from alpha_os.validation_service import run_validation
    from alpha_os.validation_spec import ValidationDateRange, ValidationSpec
    import alpha_os.validation_service as validation_service

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()
    store.upsert_subject_set(
        "core_crypto",
        definition=SubjectSet(
            subject_set_id="core_crypto",
            observation_specs=(
                ObservationSpec(
                    observation_spec_id="btc_close",
                    observable_id="daily_close",
                ),
                ObservationSpec(
                    observation_spec_id="eth_close",
                    observable_id="daily_close",
                ),
            ),
            bindings=(
                SubjectObservationBinding(
                    subject_id="BTC_spot",
                    asset="BTC",
                    observation_spec_id="btc_close",
                ),
                SubjectObservationBinding(
                    subject_id="ETH_spot",
                    asset="ETH",
                    observation_spec_id="eth_close",
                ),
            ),
            universe_policy=UniversePolicySpec(
                base_currency="USD",
                trading_calendar="24x7",
                benchmark_id="core_crypto",
            ),
        ),
    )

    frame = {
        "timestamp": [
            "2026-03-20T00:00:00Z",
            "2026-03-21T00:00:00Z",
            "2026-03-22T00:00:00Z",
            "2026-03-23T00:00:00Z",
            "2026-03-24T00:00:00Z",
            "2026-03-25T00:00:00Z",
        ],
        "value": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
    }

    def _fake_loader(*, base_url: str, asset: str, observation_spec):
        import pandas as pd

        return pd.DataFrame(frame)

    original_loader = validation_service._load_price_frame_from_signal_noise
    validation_service._load_price_frame_from_signal_noise = _fake_loader
    try:
        spec = ValidationSpec(
            signal_ids=("momentum_1d", "reversal_1d"),
            target_ids=("residual_return_3d",),
            date_ranges=(
                ValidationDateRange(
                    label="mini",
                    start_date="2026-03-22",
                    end_date="2026-03-22",
                ),
            ),
            metric_windows=(20,),
            aggregation_kinds=("active_equal_mean",),
            subject_set_ids=("core_crypto",),
            base_url="http://example.com",
        )
        result = run_validation(store, spec=spec, recorded_at="2026-03-29T00:00:00+00:00")
    finally:
        validation_service._load_price_frame_from_signal_noise = original_loader
        store.close()

    store = EvaluationStore(db_path)
    try:
        decision_results = store.list_validation_decision_results(run_id=result.run_id)
        assert len(decision_results) == 1
        assert decision_results[0].subject_set_id == "core_crypto"
        assert decision_results[0].step_count == 1
        assert decision_results[0].mean_gross_notional_exposure >= 0.0
        assert decision_results[0].mean_net_notional_exposure == pytest.approx(0.0, abs=1.0)
        assert decision_results[0].mean_long_notional_exposure >= 0.0
        assert decision_results[0].mean_short_notional_exposure >= 0.0
        assert decision_results[0].mean_traded_notional >= 0.0
        assert decision_results[0].cost_notional_total >= 0.0
        assert decision_results[0].funding_cost_notional_total <= decision_results[0].cost_notional_total
        assert decision_results[0].borrow_cost_notional_total >= 0.0
        assert decision_results[0].roll_cost_notional_total >= 0.0
    finally:
        store.close()


def test_run_validation_rejects_incomplete_universe_policy_for_multi_subject_set(tmp_path):
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
    )
    from alpha_os.store import EvaluationStore
    from alpha_os.validation_service import run_validation
    from alpha_os.validation_spec import ValidationDateRange, ValidationSpec

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()
    _register_default_validation_signal_specs(store)
    store.upsert_subject_set(
        "core_crypto",
        definition=SubjectSet(
            subject_set_id="core_crypto",
            observation_specs=(
                ObservationSpec(
                    observation_spec_id="btc_close",
                    observable_id="daily_close",
                ),
                ObservationSpec(
                    observation_spec_id="eth_close",
                    observable_id="daily_close",
                ),
            ),
            bindings=(
                SubjectObservationBinding(
                    subject_id="BTC_spot",
                    asset="BTC",
                    observation_spec_id="btc_close",
                ),
                SubjectObservationBinding(
                    subject_id="ETH_spot",
                    asset="ETH",
                    observation_spec_id="eth_close",
                ),
            ),
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        run_validation(
            store,
            spec=ValidationSpec(
                signal_ids=("momentum_1d",),
                target_ids=("residual_return_1d",),
                date_ranges=(
                    ValidationDateRange(
                        label="mini",
                        start_date="2026-03-22",
                        end_date="2026-03-22",
                    ),
                ),
                metric_windows=(20,),
                aggregation_kinds=("active_equal_mean",),
                subject_set_ids=("core_crypto",),
                base_url="http://example.com",
            ),
            recorded_at="2026-03-29T00:00:00+00:00",
        )

    assert (
        "subject set universe policy is incomplete for multi-subject validation/evaluation: "
        "core_crypto missing base_currency, trading_calendar, benchmark_id"
        in str(excinfo.value)
    )
