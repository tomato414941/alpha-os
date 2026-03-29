from __future__ import annotations


def test_build_portfolio_decision_input_uses_latest_meta_prediction(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.meta_metrics_service import refresh_target_meta_prediction_metrics
    from alpha_os.portfolio_decision_service import build_portfolio_decision_input
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_hypothesis("reversal_1d")
        store.register_hypothesis("average_gap_3d")

        values = [
            ("2026-03-24", 0.2, 0.0, 0.1),
            ("2026-03-25", 0.3, 0.1, 0.2),
            ("2026-03-26", 0.1, 0.05, 0.05),
        ]
        for date, pred_a, pred_b, obs in values:
            evaluation_id = f"BTC:residual_return_3d:{date}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id="reversal_1d",
                prediction_value=pred_a,
                observation_value=obs,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id="average_gap_3d",
                prediction_value=pred_b,
                observation_value=obs,
            )

        refresh_target_meta_predictions(store)
        refresh_target_meta_prediction_metrics(store)

        decision_input = build_portfolio_decision_input(store)

        assert decision_input is not None
        assert decision_input.portfolio_state.portfolio_id is None
        assert decision_input.predictive_signals[0].source_kind == "meta_prediction"
        assert decision_input.predictive_signals[0].subject_id == "BTC"
        assert decision_input.predictive_signals[0].target_id == "residual_return_3d"
        assert decision_input.predictive_signals[0].confidence is not None
        assert 0.0 <= decision_input.predictive_signals[0].confidence <= 1.0
        assert decision_input.risk_inputs[0].name == "realized_vol_20"
        assert decision_input.cost_inputs == ()
        assert decision_input.uncertainty_inputs[0].name == "small_sample_penalty"
    finally:
        store.close()


def test_build_portfolio_decision_input_merges_explicit_assumptions(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.meta_metrics_service import refresh_target_meta_prediction_metrics
    from alpha_os.portfolio_decision import CostInput, DependenceInput
    from alpha_os.portfolio_decision_service import (
        PortfolioDecisionAssumptions,
        build_portfolio_decision_input,
    )
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_hypothesis("reversal_1d")
        store.register_hypothesis("average_gap_3d")

        values = [
            ("2026-03-24", 0.2, 0.0, 0.1),
            ("2026-03-25", 0.3, 0.1, 0.2),
            ("2026-03-26", 0.1, 0.05, 0.05),
        ]
        for date, pred_a, pred_b, obs in values:
            evaluation_id = f"BTC:residual_return_3d:{date}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id="reversal_1d",
                prediction_value=pred_a,
                observation_value=obs,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id="average_gap_3d",
                prediction_value=pred_b,
                observation_value=obs,
            )

        refresh_target_meta_predictions(store)
        refresh_target_meta_prediction_metrics(store)

        decision_input = build_portfolio_decision_input(
            store,
            portfolio_id="paper_core",
            subject_id="BTC_spot",
            assumptions=PortfolioDecisionAssumptions(
                cost_inputs=(
                    CostInput(
                        name="turnover_penalty",
                        subject_id=None,
                        value=0.01,
                        basis="per_turnover",
                        unit="weight",
                    ),
                ),
                dependence_inputs=(
                    DependenceInput(
                        name="hidden_bet_overlap",
                        left_subject_id="BTC",
                        right_subject_id="ETH",
                        value=0.4,
                        basis="overlap",
                    ),
                ),
            ),
        )

        assert decision_input is not None
        assert decision_input.portfolio_id == "paper_core"
        assert decision_input.predictive_signals[0].subject_id == "BTC_spot"
        assert decision_input.risk_inputs[0].subject_id == "BTC_spot"
        assert decision_input.cost_inputs[0].name == "turnover_penalty"
        assert decision_input.dependence_inputs[0].right_subject_id == "ETH"
    finally:
        store.close()


def test_build_portfolio_decision_output_returns_policy_result(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.meta_metrics_service import refresh_target_meta_prediction_metrics
    from alpha_os.portfolio_decision import (
        CostInput,
        PortfolioPositionState,
        PortfolioState,
    )
    from alpha_os.portfolio_decision_service import (
        PortfolioDecisionAssumptions,
        build_portfolio_decision_output,
    )
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_hypothesis("reversal_1d")
        store.register_hypothesis("average_gap_3d")

        values = [
            ("2026-03-24", 0.4, 0.0, 0.2),
            ("2026-03-25", 0.3, 0.1, 0.1),
            ("2026-03-26", 0.2, 0.0, 0.05),
        ]
        for date, pred_a, pred_b, obs in values:
            evaluation_id = f"BTC:residual_return_3d:{date}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id="reversal_1d",
                prediction_value=pred_a,
                observation_value=obs,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id="average_gap_3d",
                prediction_value=pred_b,
                observation_value=obs,
            )

        refresh_target_meta_predictions(store)
        refresh_target_meta_prediction_metrics(store)

        decision_output = build_portfolio_decision_output(
            store,
            portfolio_id="paper_core",
            subject_id="BTC_spot",
            portfolio_state=PortfolioState(
                portfolio_id="paper_core",
                positions=(PortfolioPositionState(subject_id="BTC_spot", weight=0.05),)
            ),
            assumptions=PortfolioDecisionAssumptions(
                cost_inputs=(
                    CostInput(
                        name="expected_slippage",
                        subject_id="BTC",
                        value=1000.0,
                        basis="per_notional",
                        unit="bps",
                    ),
                ),
            ),
        )

        assert decision_output is not None
        assert decision_output.portfolio_id == "paper_core"
        assert len(decision_output.targets) == 1
        assert decision_output.targets[0].subject_id == "BTC_spot"
        assert 0.0 <= decision_output.targets[0].target_weight <= 0.05
        assert decision_output.targets[0].position_delta < 0.0
        assert decision_output.targets[0].entry_allowed is True
    finally:
        store.close()


def test_build_portfolio_decision_input_returns_none_without_meta_prediction(tmp_path):
    from alpha_os.portfolio_decision_service import build_portfolio_decision_input
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        assert build_portfolio_decision_input(store) is None
    finally:
        store.close()


def test_persist_portfolio_decision_output_writes_runtime_artifact(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.meta_metrics_service import refresh_target_meta_prediction_metrics
    from alpha_os.portfolio_decision import (
        CostInput,
        PortfolioPositionState,
        PortfolioState,
    )
    from alpha_os.portfolio_decision_service import (
        PortfolioDecisionAssumptions,
        build_portfolio_decision_output,
        persist_portfolio_decision_output,
    )
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_hypothesis("reversal_1d")
        store.register_hypothesis("average_gap_3d")

        values = [
            ("2026-03-24", 0.4, 0.0, 0.2),
            ("2026-03-25", 0.3, 0.1, 0.1),
            ("2026-03-26", 0.2, 0.0, 0.05),
        ]
        for date, pred_a, pred_b, obs in values:
            evaluation_id = f"BTC:residual_return_3d:{date}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id="reversal_1d",
                prediction_value=pred_a,
                observation_value=obs,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id="average_gap_3d",
                prediction_value=pred_b,
                observation_value=obs,
            )

        refresh_target_meta_predictions(store)
        refresh_target_meta_prediction_metrics(store)

        assumptions = PortfolioDecisionAssumptions(
            cost_inputs=(
                CostInput(
                    name="expected_slippage",
                    subject_id="BTC",
                    value=1000.0,
                    basis="per_notional",
                    unit="bps",
                ),
            ),
        )
        decision_output = build_portfolio_decision_output(
            store,
            portfolio_id="paper_core",
            portfolio_state=PortfolioState(
                portfolio_id="paper_core",
                positions=(PortfolioPositionState(subject_id="BTC", weight=0.05),),
            ),
            assumptions=assumptions,
        )
        assert decision_output is not None
        assert len(decision_output.targets) == 1

        persist_portfolio_decision_output(
            store,
            decision_output=decision_output,
            target_id="residual_return_3d",
            aggregation_kind="corr_weighted_mean",
            assumptions=assumptions,
        )

        items = store.list_portfolio_decisions(portfolio_id="paper_core", limit=10)
        assert len(items) == 1
        assert items[0].portfolio_id == "paper_core"
        assert items[0].subject_id == "BTC"
        assert items[0].target_id == "residual_return_3d"
        assert items[0].aggregation_kind == "corr_weighted_mean"
        assert items[0].details is not None
        assert items[0].details["assumptions"]["cost_inputs"][0]["name"] == "expected_slippage"
    finally:
        store.close()
