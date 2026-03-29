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
        assert decision_input.predictive_signals[0].source_kind == "meta_prediction"
        assert decision_input.predictive_signals[0].subject_id == "BTC"
        assert decision_input.predictive_signals[0].target_id == "residual_return_3d"
        assert decision_input.predictive_signals[0].confidence is not None
        assert 0.0 <= decision_input.predictive_signals[0].confidence <= 1.0
        assert decision_input.risk_inputs[0].name == "realized_vol_20"
        assert decision_input.cost_inputs[0].name == "turnover_penalty"
        assert decision_input.uncertainty_inputs[0].name == "small_sample_penalty"
    finally:
        store.close()


def test_build_portfolio_decision_output_returns_policy_result(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.meta_metrics_service import refresh_target_meta_prediction_metrics
    from alpha_os.portfolio_decision import PortfolioPositionState, PortfolioState
    from alpha_os.portfolio_decision_service import build_portfolio_decision_output
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
            portfolio_state=PortfolioState(
                positions=(PortfolioPositionState(subject_id="BTC", weight=0.05),)
            ),
        )

        assert decision_output is not None
        assert len(decision_output.targets) == 1
        assert decision_output.targets[0].subject_id == "BTC"
        assert 0.0 < decision_output.targets[0].target_weight < 0.05
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
