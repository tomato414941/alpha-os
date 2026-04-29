from __future__ import annotations

import pytest


def test_portfolio_concentration_metrics_use_gross_signed_weights():
    from alpha_os.portfolio_concentration import (
        concentration_snapshot,
        portfolio_effective_n,
        top_n_gross_share,
    )

    weights = {
        "AAA": 0.40,
        "BBB": -0.30,
        "CCC": 0.20,
        "DDD": -0.10,
    }

    assert portfolio_effective_n(weights.values()) == pytest.approx(
        1.0 / (0.4**2 + 0.3**2 + 0.2**2 + 0.1**2)
    )
    assert top_n_gross_share(weights.values(), top_n=3) == pytest.approx(0.90)

    snapshot = concentration_snapshot(
        weights,
        cluster_by_subject={
            "AAA": "rates",
            "BBB": "rates",
            "CCC": "fx",
            "DDD": "commodity",
        },
        min_abs_weight=0.15,
    )

    assert snapshot.active_position_count == 3
    assert snapshot.max_subject_label == "AAA"
    assert snapshot.max_subject_gross_share == pytest.approx(0.40)
    assert snapshot.max_cluster_label == "rates"
    assert snapshot.max_cluster_gross_share == pytest.approx(0.70)


def test_portfolio_concentration_handles_zero_weights():
    from alpha_os.portfolio_concentration import (
        concentration_snapshot,
        portfolio_effective_n,
        top_n_gross_share,
    )

    assert portfolio_effective_n([0.0, 0.0]) == pytest.approx(0.0)
    assert top_n_gross_share([0.0, 0.0], top_n=3) == pytest.approx(0.0)

    snapshot = concentration_snapshot({"AAA": 0.0, "BBB": 0.0})

    assert snapshot.active_position_count == 0
    assert snapshot.effective_n == pytest.approx(0.0)
    assert snapshot.max_subject_label is None
    assert snapshot.max_cluster_label is None
