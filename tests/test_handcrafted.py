from __future__ import annotations

import sqlite3

from alpha_os.alpha.handcrafted import get_handcrafted_expressions, list_handcrafted_sets
from alpha_os.alpha.registry import AlphaRegistry


def test_list_handcrafted_sets_for_btc():
    assert list_handcrafted_sets("BTC") == ["baseline"]


def test_get_handcrafted_expressions_returns_parseable_canonical_strings():
    expressions = get_handcrafted_expressions("BTC", "baseline")

    assert len(expressions) == 10
    assert expressions[0] == "(roc_20 btc_ohlcv)"
    assert expressions[-1] == "(if_gt vpin_btc 0.8 (neg trade_flow_btc) trade_flow_btc)"


def test_queue_candidate_expressions_is_idempotent(tmp_path):
    registry = AlphaRegistry(tmp_path / "alpha_registry.db")
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
