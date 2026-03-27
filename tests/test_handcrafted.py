from __future__ import annotations

import sqlite3

from alpha_os_recovery.legacy.managed_alphas import CandidateSeed, ManagedAlphaStore
from alpha_os_recovery.research.handcrafted import get_handcrafted_expressions, list_handcrafted_sets


def test_list_handcrafted_sets_for_btc():
    assert list_handcrafted_sets("BTC") == ["baseline"]


def test_get_handcrafted_expressions_returns_parseable_canonical_strings():
    expressions = get_handcrafted_expressions("BTC", "baseline")

    assert len(expressions) == 10
    assert expressions[0] == "(roc_20 btc_ohlcv)"
    assert expressions[-1] == "(if_gt vpin_btc 0.8 (neg trade_flow_btc) trade_flow_btc)"


def test_queue_candidate_expressions_is_idempotent(tmp_path):
    registry = ManagedAlphaStore(tmp_path / "alpha_registry.db")
    expressions = get_handcrafted_expressions("BTC", "baseline")[:2]

    inserted_first = registry.queue_candidate_expressions(
        expressions,
        source="manual_btc_baseline",
        behavior_json={"source": "handcrafted"},
    )
    inserted_second = registry.queue_candidate_expressions(
        expressions,
        source="manual_btc_baseline",
        behavior_json={"source": "handcrafted"},
    )

    conn = sqlite3.connect(tmp_path / "alpha_registry.db")
    try:
        rows = conn.execute(
            "SELECT candidate_id, expression, status, behavior_json FROM candidates ORDER BY candidate_id"
        ).fetchall()
    finally:
        conn.close()
        registry.close()

    assert inserted_first == 2
    assert inserted_second == 0
    assert len(rows) == 2
    assert rows[0][0].startswith("manual_btc_baseline_")
    assert rows[0][2] == "pending"
    assert '"source": "handcrafted"' in rows[0][3]


def test_queue_candidates_preserves_per_candidate_metadata(tmp_path):
    store = ManagedAlphaStore(tmp_path / "alpha_registry.db")
    inserted = store.queue_candidates(
        [
            CandidateSeed(
                expression="(roc_20 btc_ohlcv)",
                source="alpha_generator_btc",
                fitness=1.25,
                behavior_json={"source": "alpha_generator", "round": 3},
            ),
            CandidateSeed(
                expression="(neg book_imbalance_btc)",
                source="alpha_generator_btc",
                fitness=0.75,
                behavior_json={"source": "alpha_generator", "round": 4},
            ),
        ]
    )

    conn = sqlite3.connect(tmp_path / "alpha_registry.db")
    try:
        rows = conn.execute(
            "SELECT expression, source, fitness, behavior_json FROM candidates ORDER BY expression"
        ).fetchall()
    finally:
        conn.close()
        store.close()

    assert inserted == 2
    assert rows[0][1] == "alpha_generator_btc"
    assert rows[0][2] == 0.75
    assert '"round": 4' in rows[0][3]
    assert rows[1][1] == "alpha_generator_btc"
    assert rows[1][2] == 1.25
    assert '"round": 3' in rows[1][3]
