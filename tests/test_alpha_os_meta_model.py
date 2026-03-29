from __future__ import annotations


def test_refresh_target_meta_predictions_persists_equal_and_corr_weighted_means(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
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
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
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


def test_refresh_target_meta_prediction_metrics_persists_corr(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.meta_metrics_service import refresh_target_meta_prediction_metrics
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_hypothesis("hyp_a")
        store.register_hypothesis("hyp_b")

        values = [
            ("2026-03-27", 0.2, 0.0, 0.1),
            ("2026-03-28", 0.4, 0.1, 0.2),
            ("2026-03-29", 0.1, 0.0, 0.05),
        ]
        for date, pred_a, pred_b, obs in values:
            evaluation_id = f"BTC:residual_return_3d:{date}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id="hyp_a",
                prediction_value=pred_a,
                observation_value=obs,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id="hyp_b",
                prediction_value=pred_b,
                observation_value=obs,
            )

        refresh_target_meta_predictions(store)
        refresh_target_meta_prediction_metrics(store)
        metrics = store.list_meta_prediction_metrics()
        by_kind = {item.aggregation_kind: item for item in metrics}

        assert set(by_kind) == {"active_equal_mean", "corr_weighted_mean"}
        assert by_kind["active_equal_mean"].sample_count == 3
        assert by_kind["corr_weighted_mean"].sample_count == 3
    finally:
        store.close()


def test_refresh_target_meta_predictions_uses_only_lagged_corr_for_weighting(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_hypothesis("hyp_a")
        store.register_hypothesis("hyp_b")

        history = [
            ("2026-03-24", 0.2, -0.2, 0.2),
            ("2026-03-25", 0.3, -0.3, 0.3),
            ("2026-03-26", 0.1, -0.1, 0.1),
        ]
        for date, pred_a, pred_b, obs in history:
            evaluation_id = f"BTC:residual_return_3d:{date}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id="hyp_a",
                prediction_value=pred_a,
                observation_value=obs,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id="hyp_b",
                prediction_value=pred_b,
                observation_value=obs,
            )

        current_evaluation_id = "BTC:residual_return_3d:2026-03-27"
        store.record_prediction(
            evaluation_id=current_evaluation_id,
            hypothesis_id="hyp_a",
            prediction_value=1.0,
        )
        store.record_prediction(
            evaluation_id=current_evaluation_id,
            hypothesis_id="hyp_b",
            prediction_value=0.0,
        )

        refresh_target_meta_predictions(store)
        items = {
            item.aggregation_kind: item
            for item in store.list_meta_predictions(limit=10)
            if item.evaluation_id == current_evaluation_id
        }
        assert items["active_equal_mean"].value == 0.5
        assert items["corr_weighted_mean"].value == 1.0
        assert items["corr_weighted_mean"].details is not None
        contributors = items["corr_weighted_mean"].details["contributors"]
        weights = {item["hypothesis_id"]: item["weight"] for item in contributors}
        assert weights["hyp_a"] == 1.0
        assert weights["hyp_b"] == 0.0
    finally:
        store.close()


def test_compare_meta_aggregations_cli_orders_by_corr(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"

    for hypothesis_id in ("momentum_1d", "reversal_1d", "average_gap_3d"):
        assert (
            main(
                [
                    "register-hypothesis",
                    "--db",
                    str(db_path),
                    "--hypothesis-id",
                    hypothesis_id,
                ]
            )
            == 0
        )
        capsys.readouterr()

    values = [
        ("2026-03-24", "momentum_1d", "0.5", "0.2"),
        ("2026-03-24", "reversal_1d", "-0.5", "0.2"),
        ("2026-03-24", "average_gap_3d", "0.1", "0.2"),
        ("2026-03-25", "momentum_1d", "0.4", "0.1"),
        ("2026-03-25", "reversal_1d", "-0.4", "0.1"),
        ("2026-03-25", "average_gap_3d", "0.0", "0.1"),
        ("2026-03-26", "momentum_1d", "0.3", "0.05"),
        ("2026-03-26", "reversal_1d", "-0.3", "0.05"),
        ("2026-03-26", "average_gap_3d", "0.05", "0.05"),
    ]
    for date, hypothesis_id, prediction, observation in values:
        assert (
            main(
                [
                    "apply-evaluation",
                    "--db",
                    str(db_path),
                    "--date",
                    date,
                    "--hypothesis-id",
                    hypothesis_id,
                    "--prediction",
                    prediction,
                    "--observation",
                    observation,
                ]
            )
            == 0
        )
        capsys.readouterr()

    assert (
        main(
            [
                "compare-meta-aggregations",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "alpha-os meta aggregation comparison" in output
    assert "residual_return_3d" in output
    assert "1. kind=" in output
