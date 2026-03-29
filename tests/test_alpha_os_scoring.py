from __future__ import annotations

import pytest


def test_numerai_corr_rewards_aligned_predictions():
    import pandas as pd

    from alpha_os.scoring import numerai_corr

    predictions = pd.Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"], dtype=float)
    target = pd.Series([0.0, 0.1, 0.2, 0.3], index=["a", "b", "c", "d"], dtype=float)

    corr = numerai_corr(predictions, target)

    assert corr > 0.0


def test_meta_model_contribution_is_near_zero_when_predictions_match_meta_model():
    import pandas as pd

    from alpha_os.scoring import meta_model_contribution

    predictions = pd.Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"], dtype=float)
    meta_model = pd.Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"], dtype=float)
    target = pd.Series([0.0, 0.1, 0.2, 0.3], index=["a", "b", "c", "d"], dtype=float)

    mmc = meta_model_contribution(predictions, meta_model, target)

    assert mmc is not None
    assert abs(mmc) < 1e-9


def test_compute_hypothesis_metrics_returns_corr_and_mmc():
    import pandas as pd

    from alpha_os.scoring import compute_hypothesis_metrics

    predictions = pd.Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"], dtype=float)
    target = pd.Series([0.0, 0.1, 0.2, 0.3], index=["a", "b", "c", "d"], dtype=float)
    meta_model = pd.Series([0.4, 0.3, 0.2, 0.1], index=["a", "b", "c", "d"], dtype=float)

    metrics = compute_hypothesis_metrics(
        predictions=predictions,
        target=target,
        meta_model=meta_model,
        window_size=4,
    )

    assert metrics.sample_count == 4
    assert metrics.mmc_sample_count == 4
    assert metrics.corr != 0.0
    assert metrics.mmc is not None


def test_compute_hypothesis_metrics_uses_nullable_mmc_when_meta_model_is_missing():
    import pandas as pd

    from alpha_os.scoring import compute_hypothesis_metrics

    predictions = pd.Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"], dtype=float)
    target = pd.Series([0.0, 0.1, 0.2, 0.3], index=["a", "b", "c", "d"], dtype=float)

    metrics = compute_hypothesis_metrics(
        predictions=predictions,
        target=target,
        meta_model=None,
        window_size=4,
    )

    assert metrics.sample_count == 4
    assert metrics.mmc_sample_count == 0
    assert metrics.mmc is None


def test_refresh_hypothesis_metrics_uses_only_active_peers(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.metrics_service import refresh_hypothesis_metrics
    from alpha_os.scoring import compute_hypothesis_metrics
    from alpha_os.store import EvaluationStore
    import pandas as pd

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_hypothesis("hyp_a")
        store.register_hypothesis("hyp_b")
        store.register_hypothesis("hyp_c")

        observations = [0.0, 0.1, 0.2, 0.3]
        pred_a = [0.1, 0.2, 0.3, 0.4]
        pred_b = [0.4, 0.3, 0.2, 0.1]
        pred_c = [10.0, 10.0, 10.0, 10.0]

        for idx, (obs, a, b, c) in enumerate(zip(observations, pred_a, pred_b, pred_c), start=1):
            evaluation_id = f"BTC:residual_return_3d:e{idx}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id="hyp_a",
                prediction_value=a,
                observation_value=obs,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id="hyp_b",
                prediction_value=b,
                observation_value=obs,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id="hyp_c",
                prediction_value=c,
                observation_value=obs,
            )

        store.set_hypothesis_status("hyp_c", action="deactivate")
        refresh_hypothesis_metrics(store, hypothesis_id="hyp_a", window_size=4)
        metric = store.get_hypothesis_metric("hyp_a")
        assert metric is not None

        expected = compute_hypothesis_metrics(
            predictions=pd.Series(pred_a, index=[f"BTC:residual_return_3d:e{i}" for i in range(1, 5)], dtype=float),
            target=pd.Series(observations, index=[f"BTC:residual_return_3d:e{i}" for i in range(1, 5)], dtype=float),
            meta_model=pd.Series(pred_b, index=[f"BTC:residual_return_3d:e{i}" for i in range(1, 5)], dtype=float),
            window_size=4,
        )

        assert metric.corr == pytest.approx(expected.corr)
        assert metric.mmc == pytest.approx(expected.mmc)
        assert metric.mmc_baseline_type == "active_peer_mean"
        assert metric.mmc_peer_count == 1
        assert metric.mmc_sample_count == 4
    finally:
        store.close()


def test_refresh_hypothesis_metrics_sets_nullable_mmc_when_no_active_peers(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.metrics_service import refresh_hypothesis_metrics
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_hypothesis("hyp_a")

        observations = [0.0, 0.1, 0.2, 0.3]
        predictions = [0.1, 0.2, 0.3, 0.4]

        for idx, (obs, pred) in enumerate(zip(observations, predictions), start=1):
            evaluation_id = f"BTC:residual_return_3d:e{idx}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                hypothesis_id="hyp_a",
                prediction_value=pred,
                observation_value=obs,
            )

        refresh_hypothesis_metrics(store, hypothesis_id="hyp_a", window_size=4)
        metric = store.get_hypothesis_metric("hyp_a")
        assert metric is not None
        assert metric.mmc is None
        assert metric.mmc_baseline_type == "active_peer_mean"
        assert metric.mmc_peer_count == 0
        assert metric.mmc_sample_count == 0
    finally:
        store.close()
