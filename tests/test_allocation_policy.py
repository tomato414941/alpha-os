from __future__ import annotations

import numpy as np

from alpha_os.hypotheses.allocation_policy import (
    abs_signal_correlation,
    apply_ranked_feature_usage_cap,
    apply_ranked_replacement_policy,
    dedupe_ranked_ids_by_semantic_key,
    dedupe_ranked_ids_by_signal_similarity,
    resolve_ranked_current_ids,
    seed_ranked_selection,
)


def test_dedupe_ranked_ids_by_semantic_key_keeps_first_seen():
    kept, skipped = dedupe_ranked_ids_by_semantic_key(
        ["a", "b", "c"],
        semantic_key_by_id={"a": "k1", "b": "k1", "c": "k2"},
    )

    assert kept == ["a", "c"]
    assert skipped == ["b"]


def test_dedupe_ranked_ids_by_signal_similarity_skips_highly_correlated_ids():
    base = np.linspace(-1.0, 1.0, 80)
    kept, skipped = dedupe_ranked_ids_by_signal_similarity(
        ["a", "b", "c"],
        signal_by_id={
            "a": base,
            "b": base * 0.999,
            "c": np.sin(np.linspace(0.0, 4.0, 80)),
        },
        similarity_max=0.995,
    )

    assert kept == ["a", "c"]
    assert skipped == ["b"]


def test_apply_ranked_feature_usage_cap_preserves_minimum_keep_count():
    kept, skipped = apply_ranked_feature_usage_cap(
        ["a", "b", "c", "d"],
        feature_names_by_id={
            "a": {"nasdaq"},
            "b": {"nasdaq"},
            "c": {"sp500"},
            "d": {"gold"},
        },
        max_occurrences=1,
        min_keep=3,
    )

    assert kept == ["a", "c", "d"]
    assert skipped == ["b"]


def test_ranked_selection_helpers_seed_and_replace_current_ids():
    ranked_ids = ["c0", "i0", "i1", "c1"]
    current_ids = resolve_ranked_current_ids(["i0", "i1", "missing"], ranked_ids)

    assert current_ids == ["i0", "i1"]

    selected_ids = seed_ranked_selection(
        current_ids=current_ids,
        ranked_ids=ranked_ids,
        max_selected=2,
    )

    selected_ids, replacements = apply_ranked_replacement_policy(
        selected_ids=selected_ids,
        current_ids=current_ids,
        remaining_ids=[item_id for item_id in ranked_ids if item_id not in selected_ids],
        rank_key_by_id={
            "c0": (1.1, 1.0, 1.1),
            "i0": (0.9, 1.0, 0.9),
            "i1": (0.8, 1.0, 0.8),
            "c1": (0.7, 1.0, 0.7),
        },
        score_by_id={
            "c0": 1.1,
            "i0": 0.9,
            "i1": 0.8,
            "c1": 0.7,
        },
        max_replacements=1,
        promotion_margin=0.25,
    )

    assert selected_ids == ["i0", "c0"]
    assert replacements == 1


def test_abs_signal_correlation_handles_low_variance_and_short_series():
    assert abs_signal_correlation(np.ones(20), np.ones(20)) == 0.0
    assert abs_signal_correlation(np.arange(5.0), np.arange(5.0)) == 0.0
