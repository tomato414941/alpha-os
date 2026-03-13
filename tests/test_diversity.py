from __future__ import annotations

import numpy as np

import pytest

from alpha_os.alpha.diversity import analyze_diversity, infer_feature_families
from alpha_os.alpha.managed_alphas import AlphaRecord


def test_infer_feature_families_uses_runtime_domains():
    families = infer_feature_families(
        {"book_imbalance_btc", "sp500", "btc_ohlcv", "btc_hashrate"}
    )

    assert "macro" in families
    assert "microstructure" in families
    assert "onchain" in families
    assert "price" in families


def test_analyze_diversity_surfaces_redundant_pairs():
    base = np.linspace(-1.0, 1.0, 80)
    data = {
        "book_imbalance_btc": base,
        "sp500": np.sin(np.linspace(0.0, 4.0, 80)),
    }
    records = [
        AlphaRecord(alpha_id="a", expression="book_imbalance_btc"),
        AlphaRecord(alpha_id="b", expression="(neg book_imbalance_btc)"),
        AlphaRecord(alpha_id="c", expression="sp500"),
    ]

    report = analyze_diversity(records, data, n_days=80, lookback=60, top_pairs=2)

    assert report.summary.n_records == 3
    assert report.summary.n_analyzed == 3
    assert report.top_redundant_pairs[0].alpha_id_a == "a"
    assert report.top_redundant_pairs[0].alpha_id_b == "b"
    assert report.top_redundant_pairs[0].abs_signal_correlation == pytest.approx(1.0)
    assert report.rows[0].alpha_id in {"a", "b"}
    assert report.summary.family_counts["microstructure"] == 2
    assert report.summary.family_counts["macro"] == 1
    assert report.summary.n_unique_features == 2
    assert "book_imbalance_btc" in report.summary.feature_usage_counts
    assert report.top_input_pairs[0].feature_a == "book_imbalance_btc"
    assert report.top_input_pairs[0].feature_b == "sp500"
