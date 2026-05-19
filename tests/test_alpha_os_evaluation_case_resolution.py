from __future__ import annotations

from types import SimpleNamespace

import pytest

from alpha_os.evaluation_task import (
    EvaluationTask,
    build_evaluation_task_id,
)
from alpha_os.evaluation_task_query import (
    select_evaluation_tasks,
)
from alpha_os.strategy_variant import (
    StrategyVariantConfig,
    derive_trading_strategy_from_signal_discovery,
    overridden_strategy_variant_config,
)
from alpha_os.evaluation_cost_config import (
    EvaluationRebalanceFrictionPolicySpec,
    ExecutionCostAssumptionsSpec,
    HoldingCostAssumptionsSpec,
)
from alpha_os.evaluation_plan import build_evaluation_plan
from alpha_os.evaluation_spec import (
    EvaluationDateRange,
    EvaluationFold,
    EvaluationSpec,
)
from alpha_os.portfolio_construction_config import (
    PortfolioConstructionSizingSpec,
    PortfolioConstructionSpec,
)
from alpha_os.signal_discovery import SignalDiscoverySpec
from alpha_os.store import EvaluationStore
from alpha_os.strategy_sleeves import (
    StrategySleeveCompositionSpec,
    StrategySleeveSpec,
)
from alpha_os.trading_strategy import TradingStrategySpec


def _make_evaluation_trading_config(
    *,
    sizing_method: str = "signal_weighted",
    sizing_engine: str | None = None,
    rebalance_interval_steps: int = 1,
    long_only: bool = False,
    direction_mode: str | None = None,
    top_k: int | None = None,
    gross_exposure_cap: float | None = None,
    target_vol: float | None = None,
    gross_leverage_cap: float | None = None,
    net_exposure_target: float | None = None,
    asset_class_weight_caps: dict[str, float] | None = None,
    cluster_weight_caps: dict[str, float] | None = None,
    sleeve_composition: StrategySleeveCompositionSpec | None = None,
) -> StrategyVariantConfig:
    return StrategyVariantConfig(
        portfolio_construction=PortfolioConstructionSpec(
            sizing_policy=PortfolioConstructionSizingSpec(
                sizing_method=sizing_method,
                sizing_engine=sizing_engine,
            ),
            rebalance_interval_steps=rebalance_interval_steps,
            long_only=long_only,
            direction_mode=direction_mode,
            gross_exposure_cap=gross_exposure_cap,
            target_vol=target_vol,
            gross_leverage_cap=gross_leverage_cap,
            net_exposure_target=net_exposure_target,
            asset_class_weight_caps={} if asset_class_weight_caps is None else asset_class_weight_caps,
            cluster_weight_caps={} if cluster_weight_caps is None else cluster_weight_caps,
            sleeve_composition=sleeve_composition,
        ),
        rebalance_friction_policy=EvaluationRebalanceFrictionPolicySpec(),
        execution_cost_assumptions=ExecutionCostAssumptionsSpec(),
        top_k=top_k,
        holding_cost_assumptions=HoldingCostAssumptionsSpec(),
    )


def _make_signal_discovery(
    *,
    signal_discovery_id: str = "global_macro_search",
    subject_set_id: str = "global_macro_core",
    target_id: str = "residual_return_5d",
    signal_spec_id: str = "momentum_1d",
):
    definition = SignalDiscoverySpec(
        signal_discovery_id=signal_discovery_id,
        subject_set_id=subject_set_id,
        signal_spec_ids=(signal_spec_id,),
        target_id=target_id,
    )
    return SimpleNamespace(
        signal_discovery_id=signal_discovery_id,
        definition=definition,
    )


def test_evaluation_task_ignores_legacy_trading_config_document_fields():
    case = EvaluationTask.from_document(
        evaluation_task_id="case:legacy",
        document={
            "strategy_id": "strategy:test",
            "evaluation_spec_id": "evaluation_spec:test",
            "signal_discovery_id": "signal:test",
            "base_url": "http://example.com",
            "created_at": "2026-04-24T00:00:00Z",
            "portfolio_construction": PortfolioConstructionSpec(
                sizing_policy=PortfolioConstructionSizingSpec(
                    sizing_method="equal_weight",
                    sizing_engine="history_based",
                )
            ).to_document(),
            "rebalance_friction_policy": EvaluationRebalanceFrictionPolicySpec().to_document(),
            "execution_cost_assumptions": ExecutionCostAssumptionsSpec().to_document(),
            "holding_cost_assumptions": HoldingCostAssumptionsSpec().to_document(),
        },
    )

    assert not hasattr(case, "portfolio_construction")
    assert not hasattr(case, "rebalance_friction_policy")
    assert not hasattr(case, "execution_cost_assumptions")
    assert not hasattr(case, "holding_cost_assumptions")
    assert "portfolio_construction" not in case.to_document()
    assert "holding_cost_assumptions" not in case.to_document()


def _register_signal_discovery_case(
    store: EvaluationStore,
    *,
    signal_discovery_id: str,
    evaluation_spec_id: str,
    base_url: str,
    config: StrategyVariantConfig,
    created_at: str,
    subject_set_id: str = "global_macro_core",
    target_id: str = "residual_return_5d",
    signal_spec_id: str = "momentum_1d",
) -> tuple[TradingStrategySpec, EvaluationTask]:
    signal_discovery = _make_signal_discovery(
        signal_discovery_id=signal_discovery_id,
        subject_set_id=subject_set_id,
        target_id=target_id,
        signal_spec_id=signal_spec_id,
    )
    store.upsert_signal_discovery_spec(
        signal_discovery_id,
        definition=signal_discovery.definition,
        recorded_at=created_at,
    )
    strategy = derive_trading_strategy_from_signal_discovery(
        signal_discovery=signal_discovery,
        variant_config=config,
        created_at=created_at,
    )
    store.upsert_trading_strategy(trading_strategy=strategy)
    evaluation_task = EvaluationTask(
        evaluation_task_id=build_evaluation_task_id(
            strategy_id=strategy.strategy_id,
            evaluation_spec_id=evaluation_spec_id,
        ),
        strategy_id=strategy.strategy_id,
        evaluation_spec_id=evaluation_spec_id,
    )
    store.upsert_evaluation_task(task=evaluation_task)
    return strategy, evaluation_task


def _make_evaluation_spec_with_two_folds() -> EvaluationSpec:
    return EvaluationSpec(
        execution_range=EvaluationDateRange(
            label="full",
            start_date="2026-01-01",
            end_date="2026-04-30",
        ),
        evaluation_folds=(
            EvaluationFold(
                label="fold_1",
                execution_range=EvaluationDateRange(
                    label="fold_1_execution",
                    start_date="2026-01-01",
                    end_date="2026-02-28",
                ),
                evaluation_date_ranges=(
                    EvaluationDateRange(
                        label="fold_1_eval",
                        start_date="2026-03-01",
                        end_date="2026-03-31",
                    ),
                ),
            ),
            EvaluationFold(
                label="fold_2",
                execution_range=EvaluationDateRange(
                    label="fold_2_execution",
                    start_date="2026-02-01",
                    end_date="2026-03-31",
                ),
                evaluation_date_ranges=(
                    EvaluationDateRange(
                        label="fold_2_eval",
                        start_date="2026-04-01",
                        end_date="2026-04-30",
                    ),
                ),
            ),
        ),
        metric_group_names=("decision_quality",),
        metric_windows=(20,),
    )


def test_overridden_strategy_variant_config_returns_same_config_without_override():
    config = _make_evaluation_trading_config()

    resolved = overridden_strategy_variant_config(
        config,
        sizing_method=None,
        sizing_engine=None,
    )

    assert resolved is config


def test_overridden_strategy_variant_config_preserves_risk_contract():
    sleeve_composition = StrategySleeveCompositionSpec(
        sleeves=(
            StrategySleeveSpec(
                sleeve_id="trend",
                sleeve_kind="trend",
                risk_budget=1.0,
            ),
        )
    )
    config = _make_evaluation_trading_config(
        sizing_method="signal_weighted",
        sizing_engine="rule_based",
        long_only=False,
        top_k=5,
        gross_exposure_cap=1.5,
        target_vol=0.18,
        gross_leverage_cap=1.8,
        net_exposure_target=0.0,
        asset_class_weight_caps={"commodity": 0.4},
        cluster_weight_caps={"rates": 0.35},
        sleeve_composition=sleeve_composition,
    )

    resolved = overridden_strategy_variant_config(
        config,
        sizing_method="hierarchical_risk_parity",
        sizing_engine=None,
    )

    construction = resolved.portfolio_construction
    assert construction.sizing_method == "hierarchical_risk_parity"
    assert construction.sizing_engine == "history_based"
    assert construction.long_only is False
    assert resolved.top_k == 5
    assert construction.gross_exposure_cap == 1.5
    assert construction.target_vol == 0.18
    assert construction.gross_leverage_cap == 1.8
    assert construction.net_exposure_target == 0.0
    assert construction.asset_class_weight_caps == {"commodity": 0.4}
    assert construction.cluster_weight_caps == {"rates": 0.35}
    assert construction.sleeve_composition == sleeve_composition


def test_overridden_strategy_variant_config_can_override_direction_mode():
    config = _make_evaluation_trading_config(long_only=False)

    resolved = overridden_strategy_variant_config(
        config,
        sizing_method=None,
        sizing_engine=None,
        direction_mode="short_only",
    )

    assert resolved.portfolio_construction.direction_mode == "short_only"
    assert resolved.portfolio_construction.long_only is False


def test_overridden_strategy_variant_config_can_override_sizing_engine_only():
    config = _make_evaluation_trading_config(
        sizing_method="signal_weighted",
        sizing_engine="rule_based",
    )

    resolved = overridden_strategy_variant_config(
        config,
        sizing_method=None,
        sizing_engine="optimizer",
    )

    assert resolved.portfolio_construction.sizing_method == "signal_weighted"
    assert resolved.portfolio_construction.sizing_engine == "optimizer"


def test_derived_trading_strategy_uses_top_k_selection_when_top_k_is_set():
    config = _make_evaluation_trading_config(
        sizing_method="equal_weight",
        sizing_engine="history_based",
        rebalance_interval_steps=5,
        long_only=True,
        top_k=3,
        gross_exposure_cap=0.8,
    )

    strategy = derive_trading_strategy_from_signal_discovery(
        signal_discovery=_make_signal_discovery(target_id="residual_return_3d"),
        variant_config=config,
        created_at="2026-04-17T00:00:00Z",
    )

    assert strategy.selection_kind == "top_k"
    assert strategy.portfolio.top_k == 3


def test_derived_trading_strategy_preserves_risk_policy_constraints():
    config = _make_evaluation_trading_config(
        sizing_method="signal_weighted",
        sizing_engine="rule_based",
        long_only=False,
        gross_exposure_cap=1.2,
        target_vol=0.18,
        gross_leverage_cap=1.5,
        net_exposure_target=0.0,
        asset_class_weight_caps={"commodity": 0.35},
        cluster_weight_caps={"rates_us": 0.3},
    )

    strategy = derive_trading_strategy_from_signal_discovery(
        signal_discovery=_make_signal_discovery(target_id="residual_return_3d"),
        variant_config=config,
        created_at="2026-04-17T00:00:00Z",
    )

    construction = strategy.portfolio.portfolio_construction
    assert construction.long_only is False
    assert construction.gross_exposure_cap == 1.2
    assert construction.target_vol == 0.18
    assert construction.gross_leverage_cap == 1.5
    assert construction.net_exposure_target == 0.0
    assert construction.asset_class_weight_caps == {"commodity": 0.35}
    assert construction.cluster_weight_caps == {"rates_us": 0.3}


def test_select_evaluation_tasks_does_not_repair_strategy_from_case_config(tmp_path):
    store = EvaluationStore(tmp_path / "runtime.db")
    try:
        store.ensure_schema()
        signal_discovery = _make_signal_discovery(
            signal_discovery_id="global_macro_search",
        )
        store.upsert_signal_discovery_spec(
            "global_macro_search",
            definition=signal_discovery.definition,
            recorded_at="2026-04-17T00:00:00Z",
        )
        config = _make_evaluation_trading_config(
            sizing_method="signal_weighted",
            sizing_engine="rule_based",
            gross_exposure_cap=1.5,
            target_vol=0.18,
            gross_leverage_cap=1.5,
            net_exposure_target=0.0,
        )
        current_strategy = derive_trading_strategy_from_signal_discovery(
            signal_discovery=signal_discovery,
            variant_config=config,
            created_at="2026-04-17T00:00:00Z",
        )
        stale_document = current_strategy.to_document()
        stale_construction = stale_document["portfolio"]["portfolio_construction"]
        stale_construction.pop("target_vol")
        stale_construction.pop("gross_leverage_cap")
        stale_construction.pop("net_exposure_target")
        stale_strategy = TradingStrategySpec.from_document(stale_document)
        store.upsert_trading_strategy(trading_strategy=stale_strategy)
        store.upsert_evaluation_task(
            task=EvaluationTask(
                evaluation_task_id="case:macro",
                strategy_id=current_strategy.strategy_id,
                evaluation_spec_id="macro_eval",
            )
        )

        resolved_tasks = select_evaluation_tasks(
            store,
            evaluation_spec_id="macro_eval",
            strategy_ids=None,
        )

        refreshed_state = store.get_trading_strategy(current_strategy.strategy_id)
        assert len(resolved_tasks) == 1
        assert refreshed_state is not None
        construction = refreshed_state.trading_strategy.portfolio_construction
        assert construction is not None
        assert construction.target_vol is None
        assert construction.gross_leverage_cap is None
        assert construction.net_exposure_target is None
    finally:
        store.close()


def test_select_evaluation_tasks_dedupes_base_url_override(tmp_path):
    store = EvaluationStore(tmp_path / "runtime.db")
    try:
        store.ensure_schema()
        config = _make_evaluation_trading_config(
            sizing_method="signal_weighted",
            sizing_engine="rule_based",
        )
        strategy, _case = _register_signal_discovery_case(
            store,
            signal_discovery_id="global_macro_search",
            evaluation_spec_id="macro_eval",
            base_url="http://127.0.0.1:8000",
            config=config,
            created_at="2026-04-17T00:00:00Z",
        )
        _register_signal_discovery_case(
            store,
            signal_discovery_id="global_macro_search",
            evaluation_spec_id="macro_eval",
            base_url="http://127.0.0.1:8000",
            config=config,
            created_at="2026-04-18T00:00:00Z",
        )
        resolved_tasks = select_evaluation_tasks(
            store,
            evaluation_spec_id="macro_eval",
            strategy_ids=None,
        )

        assert len(resolved_tasks) == 1
        assert resolved_tasks[0].strategy_id == strategy.strategy_id
    finally:
        store.close()


def test_select_evaluation_tasks_filters_strategy_ids(tmp_path):
    store = EvaluationStore(tmp_path / "runtime.db")
    try:
        store.ensure_schema()
        config = _make_evaluation_trading_config(
            sizing_method="signal_weighted",
            sizing_engine="rule_based",
        )
        strategy_a, _case_a = _register_signal_discovery_case(
            store,
            signal_discovery_id="macro_a",
            evaluation_spec_id="macro_eval",
            base_url="https://signal-noise.example",
            config=config,
            created_at="2026-04-17T00:00:00Z",
            signal_spec_id="momentum_1d",
        )
        strategy_b, _case_b = _register_signal_discovery_case(
            store,
            signal_discovery_id="macro_b",
            evaluation_spec_id="macro_eval",
            base_url="https://signal-noise.example",
            config=config,
            created_at="2026-04-17T00:00:00Z",
            signal_spec_id="carry_5d",
        )

        resolved_tasks = select_evaluation_tasks(
            store,
            evaluation_spec_id="macro_eval",
            strategy_ids=(strategy_b.strategy_id,),
        )

        assert len(resolved_tasks) == 1
        assert resolved_tasks[0].strategy_id == strategy_b.strategy_id
        assert resolved_tasks[0].strategy_id != strategy_a.strategy_id
    finally:
        store.close()


def test_select_evaluation_tasks_is_read_only(tmp_path):
    store = EvaluationStore(tmp_path / "runtime.db")
    try:
        store.ensure_schema()
        config = _make_evaluation_trading_config(
            sizing_method="signal_weighted",
            sizing_engine="rule_based",
            target_vol=0.18,
        )
        _strategy, _case = _register_signal_discovery_case(
            store,
            signal_discovery_id="global_macro_search",
            evaluation_spec_id="macro_eval",
            base_url="http://127.0.0.1:8000",
            config=config,
            created_at="2026-04-17T00:00:00Z",
        )

        resolved_tasks = select_evaluation_tasks(
            store,
            evaluation_spec_id="macro_eval",
            strategy_ids=None,
        )

        assert resolved_tasks == (_case,)
    finally:
        store.close()


def test_select_evaluation_tasks_keeps_existing_strategy_without_refresh(
    tmp_path,
):
    store = EvaluationStore(tmp_path / "runtime.db")
    try:
        store.ensure_schema()
        config = _make_evaluation_trading_config(
            sizing_method="signal_weighted",
            sizing_engine="rule_based",
        )
        strategy, case = _register_signal_discovery_case(
            store,
            signal_discovery_id="global_macro_search",
            evaluation_spec_id="macro_eval",
            base_url="https://signal-noise.example",
            config=config,
            created_at="2026-04-17T00:00:00Z",
        )

        resolved_tasks = select_evaluation_tasks(
            store,
            evaluation_spec_id="macro_eval",
            strategy_ids=None,
        )

        assert resolved_tasks == (case,)
        assert (
            store.get_trading_strategy(strategy.strategy_id).trading_strategy.created_at
            == "2026-04-17T00:00:00Z"
        )
    finally:
        store.close()


def test_select_evaluation_tasks_rejects_protocol_without_cases(tmp_path):
    store = EvaluationStore(tmp_path / "runtime.db")
    try:
        store.ensure_schema()

        with pytest.raises(
            ValueError,
            match="evaluation spec requires at least one evaluation task",
        ):
            select_evaluation_tasks(
                store,
                evaluation_spec_id="macro_eval",
                strategy_ids=None,
            )
    finally:
        store.close()


def test_select_evaluation_tasks_rejects_unknown_strategy_filter(tmp_path):
    store = EvaluationStore(tmp_path / "runtime.db")
    try:
        store.ensure_schema()
        _register_signal_discovery_case(
            store,
            signal_discovery_id="global_macro_search",
            evaluation_spec_id="macro_eval",
            base_url="https://signal-noise.example",
            config=_make_evaluation_trading_config(),
            created_at="2026-04-17T00:00:00Z",
        )

        with pytest.raises(
            ValueError,
            match="evaluation spec does not contain requested strategies",
        ):
            select_evaluation_tasks(
                store,
                evaluation_spec_id="macro_eval",
                strategy_ids=("strategy:missing",),
            )
    finally:
        store.close()


def test_selected_tasks_build_fold_count_plan_entries(tmp_path):
    store = EvaluationStore(tmp_path / "runtime.db")
    try:
        store.ensure_schema()
        config = _make_evaluation_trading_config()
        for base_url, created_at in (
            ("http://127.0.0.1:8000", "2026-04-17T00:00:00Z"),
            ("https://signal-noise.example", "2026-04-18T00:00:00Z"),
        ):
            _register_signal_discovery_case(
                store,
                signal_discovery_id="global_macro_search",
                evaluation_spec_id="macro_eval",
                base_url=base_url,
                config=config,
                created_at=created_at,
            )
        resolved_tasks = select_evaluation_tasks(
            store,
            evaluation_spec_id="macro_eval",
            strategy_ids=None,
        )
        evaluation_spec = _make_evaluation_spec_with_two_folds()

        plan = build_evaluation_plan(
            store,
            evaluation_spec_id="macro_eval",
            evaluation_spec=evaluation_spec,
            evaluation_tasks=resolved_tasks,
            base_url="https://signal-noise.example",
        )

        summary_keys = {
            (
                entry.evaluation_task_id,
                entry.context.strategy_id,
                entry.fold_label,
                tuple(
                    (item.label, item.start_date, item.end_date)
                    for item in entry.evaluation_date_ranges
                ),
            )
            for entry in plan.execution_requests
        }
        assert len(resolved_tasks) == 1
        assert len(plan.execution_requests) == 2
        assert len(summary_keys) == len(plan.execution_requests)
    finally:
        store.close()
