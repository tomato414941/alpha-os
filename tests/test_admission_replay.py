from __future__ import annotations

import sqlite3

from alpha_os.alpha.admission_replay import (
    alpha_id_for_expression,
    load_candidate_records,
    rebuild_registry,
)
from alpha_os.alpha.lifecycle import LifecycleConfig
from alpha_os.alpha.registry import AlphaRecord, AlphaRegistry, AlphaState


def test_alpha_id_for_expression_is_stable():
    alpha_id = alpha_id_for_expression("(rank btc_ohlcv)")

    assert alpha_id == alpha_id_for_expression("(rank btc_ohlcv)")
    assert alpha_id.startswith("v2_")


def test_alpha_id_for_expression_reuses_existing_id():
    existing = {"(rank btc_ohlcv)": "legacy_1234"}

    alpha_id = alpha_id_for_expression(
        "(rank btc_ohlcv)",
        existing_ids=existing,
    )

    assert alpha_id == "legacy_1234"


def test_load_candidate_records_deduplicates_and_preserves_existing_id(tmp_path):
    db_path = tmp_path / "alpha_registry.db"
    registry = AlphaRegistry(db_path)
    registry.register(
        AlphaRecord(
            alpha_id="existing_a1",
            expression="(rank btc_ohlcv)",
            state=AlphaState.ACTIVE,
            oos_sharpe=0.8,
        )
    )
    registry.close()

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO candidates (
            candidate_id, expression, fitness, status, oos_sharpe, pbo,
            dsr_pvalue, behavior_json, created_at, validated_at, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "cand_old",
            "(rank btc_ohlcv)",
            0.1,
            "adopted",
            0.9,
            0.2,
            0.01,
            "{}",
            100.0,
            200.0,
            None,
        ),
    )
    conn.execute(
        """
        INSERT INTO candidates (
            candidate_id, expression, fitness, status, oos_sharpe, pbo,
            dsr_pvalue, behavior_json, created_at, validated_at, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "cand_new",
            "(rank btc_ohlcv)",
            0.2,
            "adopted",
            1.1,
            0.1,
            0.02,
            "{}",
            101.0,
            300.0,
            None,
        ),
    )
    conn.commit()
    conn.close()

    records = load_candidate_records(db_path)

    assert len(records) == 1
    assert records[0].alpha_id == "existing_a1"
    assert records[0].oos_sharpe == 1.1
    assert records[0].metadata["candidate_id"] == "cand_new"


def test_rebuild_registry_rewrites_alphas_and_clears_diversity_cache(tmp_path):
    db_path = tmp_path / "alpha_registry.db"
    registry = AlphaRegistry(db_path)
    registry.register(
        AlphaRecord(
            alpha_id="legacy_old",
            expression="(old expr)",
            state=AlphaState.ACTIVE,
            oos_sharpe=0.3,
        )
    )
    registry._conn.execute(
        """
        INSERT INTO diversity_cache (alpha_id, diversity_score, computed_at, n_alphas_compared)
        VALUES (?, ?, ?, ?)
        """,
        ("legacy_old", 1.0, 1.0, 1),
    )
    registry._conn.execute(
        """
        INSERT INTO candidates (
            candidate_id, expression, fitness, status, oos_sharpe, pbo,
            dsr_pvalue, behavior_json, created_at, validated_at, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "cand_keep",
            "(keep expr)",
            0.4,
            "adopted",
            1.2,
            0.1,
            0.01,
            "{}",
            100.0,
            200.0,
            None,
        ),
    )
    registry._conn.execute(
        """
        INSERT INTO candidates (
            candidate_id, expression, fitness, status, oos_sharpe, pbo,
            dsr_pvalue, behavior_json, created_at, validated_at, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "cand_drop",
            "(drop expr)",
            0.3,
            "adopted",
            0.8,
            0.1,
            0.01,
            "{}",
            101.0,
            201.0,
            None,
        ),
    )
    registry._conn.commit()
    registry.close()

    stats = rebuild_registry(
        db_path,
        LifecycleConfig(candidate_quality_min=1.0, pbo_max=0.5, dsr_pvalue_max=0.05),
        source="candidates",
        backup=False,
    )

    registry = AlphaRegistry(db_path)
    conn = sqlite3.connect(db_path)
    try:
        assert stats.source_rows == 2
        assert stats.active_count == 1
        assert stats.rejected_count == 1
        assert registry.count(AlphaState.ACTIVE) == 1
        assert registry.count(AlphaState.REJECTED) == 1
        assert registry.count() == 2
        assert conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM diversity_cache").fetchone()[0] == 0
    finally:
        conn.close()
        registry.close()


def test_rebuild_registry_dry_run_does_not_modify_existing_alphas(tmp_path):
    db_path = tmp_path / "alpha_registry.db"
    registry = AlphaRegistry(db_path)
    registry.register(
        AlphaRecord(
            alpha_id="existing_a1",
            expression="(rank btc_ohlcv)",
            state=AlphaState.ACTIVE,
            oos_sharpe=0.9,
        )
    )
    registry._conn.execute(
        """
        INSERT INTO candidates (
            candidate_id, expression, fitness, status, oos_sharpe, pbo,
            dsr_pvalue, behavior_json, created_at, validated_at, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "cand_new",
            "(new expr)",
            0.4,
            "adopted",
            1.2,
            0.1,
            0.01,
            "{}",
            100.0,
            200.0,
            None,
        ),
    )
    registry._conn.commit()
    registry.close()

    stats = rebuild_registry(
        db_path,
        LifecycleConfig(candidate_quality_min=1.0, pbo_max=0.5, dsr_pvalue_max=0.05),
        source="candidates",
        dry_run=True,
        backup=False,
    )

    registry = AlphaRegistry(db_path)
    try:
        assert stats.active_count == 1
        assert registry.count() == 1
        assert registry.get("existing_a1") is not None
    finally:
        registry.close()
