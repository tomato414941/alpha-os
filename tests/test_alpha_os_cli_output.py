from __future__ import annotations

import pytest

from alpha_os.cli_output import (
    print_evaluation_tasks,
    print_evaluation_snapshot,
    print_subject_sets,
)
from alpha_os.portfolio_construction_config import (
    PortfolioConstructionSizingSpec,
    PortfolioConstructionSpec,
)
from alpha_os.store import EvaluationSnapshot
from alpha_os.portfolio_decision import (
    InstrumentSpec,
    ObservationSpec,
    SubjectObservationBinding,
    SubjectSet,
    UniversePolicySpec,
)
def test_print_evaluation_tasks_includes_execution_and_holding_costs(capsys):
    case = ("case:test", "strategy:test")

    print_evaluation_tasks([case])
    captured = capsys.readouterr().out

    assert "case:test" in captured
    assert "strategy=strategy:test" in captured
    assert "gross_exposure_cap=1.0" not in captured
    assert "market_impact_bps=5.0" not in captured
    assert "borrow_fee_bps_per_step=2.5" not in captured


def test_print_evaluation_snapshot_includes_replay_artifacts(capsys):
    snapshot = EvaluationSnapshot(
        evaluation_id="BTC:residual_return_3d:2026-04-17",
        subject_id="BTC_spot",
        asset="BTC",
        target_id="residual_return_3d",
        signal_id="time_series_trend__daily_close__lookback_20@BTC_spot",
        prediction_value=0.12,
        observation_value=-0.03,
        signed_edge=-0.15,
        absolute_error=0.15,
        input_source="signal_noise_backfill",
        input_range_start="2026-04-10",
        input_range_end="2026-04-17",
        funding_cost_bps=1.5,
        borrow_fee_bps=2.5,
        roll_cost_bps=0.75,
        contract_multiplier=5.0,
        observation_spec_id="macro_close",
        observable_id="daily_close",
        adapter_kind="signal_noise_asset_observable",
        created_at="2026-04-17T00:00:00Z",
    )

    print_evaluation_snapshot(snapshot, created=True)
    captured = capsys.readouterr().out

    assert "Evaluation [created] BTC:residual_return_3d:2026-04-17" in captured
    assert "Replay:   funding_bps=1.500000 borrow_bps=2.500000 roll_bps=0.750000 multiplier=5.000000" in captured


def test_evaluation_result_rejects_legacy_profiles_field():
    from alpha_os.evaluation_result import EvaluationResult

    with pytest.raises(ValueError, match="profiles field is no longer supported"):
        EvaluationResult.from_document(
            {
                "strategy_id": "strategy:test",
                "profiles": [],
                "failure_finding_groups": [],
            }
        )


def test_evaluation_result_rejects_legacy_failure_profiles_field():
    from alpha_os.evaluation_result import EvaluationResult

    with pytest.raises(ValueError, match="failure_profiles field is no longer supported"):
        EvaluationResult.from_document(
            {
                "strategy_id": "strategy:test",
                "metric_group_results": [],
                "failure_profiles": [],
            }
        )


def test_evaluation_run_result_roundtrips_metric_groups():
    from alpha_os.evaluation_run_result import EvaluationRunResult
    from alpha_os.evaluation_result import (
        EvaluationResult,
        EvaluationMetricGroupResult,
    )

    run_result = EvaluationRunResult(
        evaluation_run_result_id="run_result:test",
        evaluation_spec_id="protocol:test",
        results={
            "case:test": EvaluationResult(
                strategy_id="strategy:test",
                metric_group_results=(
                    EvaluationMetricGroupResult(
                        metric_group_name="decision_quality",
                        source="native_plan",
                        metrics={"mean_decision_net_return": 0.12},
                    ),
                ),
            )
        },
        created_at="2026-04-18T00:00:00Z",
    )

    restored = EvaluationRunResult.from_document(
        evaluation_run_result_id="run_result:test",
        document=run_result.to_document(),
    )

    assert "results" in restored.to_document()
    assert "summaries" not in restored.to_document()
    result = restored.results["case:test"]
    assert "metric_group_results" in result.to_document()
    assert "profiles" not in result.to_document()
    assert result.metric_group_results[0].metric_group_name == "decision_quality"


def test_print_subject_sets_includes_cross_asset_summary(capsys):
    subject_set = SubjectSet(
        subject_set_id="global_macro_core",
        instruments=(
            InstrumentSpec(
                instrument_id="ES_future",
                instrument_type="future",
                asset="ES",
                asset_class="equity_index",
                region="us",
                cluster="eq_us",
            ),
            InstrumentSpec(
                instrument_id="BTCUSDT_perp",
                instrument_type="perp",
                asset="BTCUSDT",
                venue="binance",
                asset_class="crypto",
                region="global",
                cluster="crypto_major",
            ),
        ),
        observation_specs=(
            ObservationSpec(
                observation_spec_id="macro_close",
                observable_id="daily_close",
                provided_observable_ids=("funding_rate", "basis"),
            ),
        ),
        bindings=(
            SubjectObservationBinding(
                subject_id="ES_future",
                subject_kind="future",
                asset="ES",
                observation_spec_id="macro_close",
                instrument_id="ES_future",
            ),
            SubjectObservationBinding(
                subject_id="BTCUSDT_perp",
                subject_kind="perp",
                asset="BTCUSDT",
                observation_spec_id="macro_close",
                instrument_id="BTCUSDT_perp",
            ),
        ),
        universe_policy=UniversePolicySpec(
            base_currency="USD",
            trading_calendar="24x7",
            benchmark_id="global_macro_core",
        ),
    )

    class State:
        subject_set_id = "global_macro_core"
        definition = subject_set

    print_subject_sets([State()])
    captured = capsys.readouterr().out

    assert "summary=[bindings=2 instruments=2" in captured
    assert "subject_kinds=future,perp" in captured
    assert "instrument_types=future,perp" in captured
    assert "asset_classes=crypto,equity_index" in captured
    assert "regions=global,us" in captured
    assert "clusters=crypto_major,eq_us" in captured
    assert "base_currency=USD" in captured
    assert "trading_calendar=24x7" in captured
    assert "benchmark_id=global_macro_core" in captured
    assert "contract_groups=instrument,observation_spec,binding,universe_policy" in captured


def test_subject_set_store_roundtrip_preserves_universe_policy(tmp_path):
    from alpha_os.store import EvaluationStore

    store = EvaluationStore(tmp_path / "runtime.db")
    store.ensure_schema()
    store.upsert_subject_set(
        "global_macro_core",
        definition=SubjectSet(
            subject_set_id="global_macro_core",
            observation_specs=(
                ObservationSpec(
                    observation_spec_id="macro_close",
                    observable_id="daily_close",
                ),
            ),
            bindings=(
                SubjectObservationBinding(
                    subject_id="BTC_spot",
                    subject_kind="asset",
                    asset="BTC",
                    observation_spec_id="macro_close",
                ),
            ),
            universe_policy=UniversePolicySpec(
                base_currency="USD",
                trading_calendar="24x7",
                benchmark_id="global_macro_core",
            ),
        ),
    )

    restored = store.get_subject_set("global_macro_core")

    assert restored is not None
    assert restored.definition.universe_policy.base_currency == "USD"
    assert restored.definition.universe_policy.trading_calendar == "24x7"
    assert restored.definition.universe_policy.benchmark_id == "global_macro_core"
    store.close()


def test_strategy_and_portfolio_construction_roundtrip_preserve_broader_constraints():
    portfolio_construction = PortfolioConstructionSpec(
        sizing_policy=PortfolioConstructionSizingSpec(
            sizing_method="signal_weighted",
            sizing_engine="rule_based",
        ),
        gross_exposure_cap=1.0,
        target_vol=0.12,
        gross_leverage_cap=1.5,
        net_exposure_target=0.3,
    )
    restored_portfolio_construction = PortfolioConstructionSpec.from_document(
        portfolio_construction.to_document()
    )

    assert restored_portfolio_construction == portfolio_construction

    short_only_construction = PortfolioConstructionSpec(direction_mode="short_only")
    restored_short_only_construction = PortfolioConstructionSpec.from_document(
        short_only_construction.to_document()
    )

    assert restored_short_only_construction.direction_mode == "short_only"
    assert restored_short_only_construction.long_only is False


def test_signed_mean_variance_sizing_spec_defaults_to_signed_optimizer_family():
    from alpha_os.portfolio_construction_config import PortfolioConstructionSizingSpec

    sizing = PortfolioConstructionSizingSpec(sizing_method="signed_mean_variance")
    restored = PortfolioConstructionSizingSpec.from_document(sizing.to_document())

    assert sizing.sizing_engine == "optimizer"
    assert sizing.sizing_family == "signed_optimizer"
    assert restored == sizing


def test_conviction_adjusted_hrp_sizing_spec_defaults_to_risk_budget_family():
    from alpha_os.portfolio_construction_config import PortfolioConstructionSizingSpec

    sizing = PortfolioConstructionSizingSpec(
        sizing_method="conviction_adjusted_hierarchical_risk_parity"
    )
    restored = PortfolioConstructionSizingSpec.from_document(sizing.to_document())

    assert sizing.sizing_engine == "history_based"
    assert sizing.sizing_family == "risk_budget_allocator"
    assert restored == sizing


def test_sizing_spec_rejects_inconsistent_family():
    from alpha_os.portfolio_construction_config import PortfolioConstructionSizingSpec

    with pytest.raises(ValueError, match="sizing_family must match"):
        PortfolioConstructionSizingSpec(
            sizing_method="signed_mean_variance",
            sizing_family="risk_budget_allocator",
        )
