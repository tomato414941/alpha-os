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
    assert metrics.corr != 0.0
