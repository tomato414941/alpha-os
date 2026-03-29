from __future__ import annotations


def test_refresh_target_meta_predictions_persists_equal_and_corr_weighted_means(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_model_service import refresh_target_meta_predictions
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_hypothesis("hyp_a")
        store.register_hypothesis("hyp_b")

        apply_evaluation(
            store,
            evaluation_id="BTC:residual_return_3d:2026-03-27",
            hypothesis_id="hyp_a",
            prediction_value=0.2,
            observation_value=0.1,
        )
        apply_evaluation(
            store,
            evaluation_id="BTC:residual_return_3d:2026-03-27",
            hypothesis_id="hyp_b",
            prediction_value=0.0,
            observation_value=0.1,
        )

        refresh_target_meta_predictions(store)
        items = store.list_meta_predictions(limit=10)
        by_kind = {item.aggregation_kind: item for item in items}

        assert set(by_kind) == {"active_equal_mean", "corr_weighted_mean"}
        assert by_kind["active_equal_mean"].value == 0.1
        assert by_kind["active_equal_mean"].contributor_count == 2
        assert by_kind["corr_weighted_mean"].contributor_count == 2
        assert by_kind["corr_weighted_mean"].details is not None
    finally:
        store.close()


def test_refresh_target_meta_predictions_falls_back_to_equal_mean_when_corr_weights_are_non_positive(
    tmp_path,
):
    from alpha_os.meta_model_service import refresh_target_meta_predictions
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_hypothesis("hyp_a")
        store.register_hypothesis("hyp_b")
        store.record_prediction(
            evaluation_id="BTC:residual_return_3d:2026-03-27",
            hypothesis_id="hyp_a",
            prediction_value=0.2,
        )
        store.record_prediction(
            evaluation_id="BTC:residual_return_3d:2026-03-27",
            hypothesis_id="hyp_b",
            prediction_value=0.0,
        )
        refresh_target_meta_predictions(store)
        items = {
            item.aggregation_kind: item
            for item in store.list_meta_predictions(limit=10)
        }
        assert items["active_equal_mean"].value == 0.1
        assert items["corr_weighted_mean"].value == 0.1
    finally:
        store.close()
