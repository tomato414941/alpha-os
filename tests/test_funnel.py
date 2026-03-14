from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from alpha_os.alpha.funnel import load_funnel_summary
from alpha_os.alpha.managed_alphas import ManagedAlphaStore
from alpha_os.evolution.discovery_pool import DiscoveryPool
from alpha_os.dsl.expr import Feature


def test_load_funnel_summary_counts_pipeline_state(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os.alpha.funnel.asset_data_dir", lambda asset: tmp_path)

    pool = DiscoveryPool()
    signal = np.random.randn(100)
    pool.add_if_empty(Feature("f1"), np.array([1.0, 2.0, 3.0]), signal, fitness=1.2)
    pool.add_if_empty(Feature("f2"), np.array([2.0, 2.0, 3.0]), signal, fitness=0.8)
    pool.save_to_db(Path(tmp_path) / "archive.db")

    store = ManagedAlphaStore(tmp_path / "alpha_registry.db")
    store.close()

    conn = sqlite3.connect(tmp_path / "alpha_registry.db")
    try:
        conn.executemany(
            """
            INSERT INTO candidates (
                candidate_id, expression, fitness, status, behavior_json,
                created_at, validated_at, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "alpha_generator_btc_a",
                    "f1",
                    1.2,
                    "pending",
                    '{"source":"alpha_generator"}',
                    1.0,
                    None,
                    None,
                ),
                (
                    "alpha_generator_btc_b",
                    "f2",
                    0.8,
                    "adopted",
                    '{"source":"alpha_generator","promotion":"manual_discovery_pool"}',
                    2.0,
                    3.0,
                    None,
                ),
                (
                    "manual_btc_baseline_a",
                    "f3",
                    0.5,
                    "rejected",
                    '{"source":"handcrafted"}',
                    4.0,
                    5.0,
                    "feature cap 50: dxy",
                ),
            ],
        )
        conn.executemany(
            """
            INSERT INTO alphas (
                alpha_id, expression, state, fitness, oos_sharpe, oos_log_growth,
                pbo, dsr_pvalue, turnover, correlation_avg, created_at, updated_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("a1", "f1", "active", 1.0, 1.0, 0.5, 0.1, 0.1, 0.0, 0.0, 1.0, 1.0, "{}"),
                ("a2", "f2", "dormant", 0.8, 0.8, 0.4, 0.2, 0.2, 0.0, 0.0, 1.0, 1.0, "{}"),
                ("a3", "f3", "rejected", 0.1, 0.1, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, "{}"),
            ],
        )
        conn.execute(
            """
            INSERT INTO deployed_alphas (alpha_id, slot, deployed_at, deployment_score, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("a1", 0, 1.0, 1.0, "{}"),
        )
        conn.commit()
    finally:
        conn.close()

    summary = load_funnel_summary("BTC")

    assert summary.discovery_pool_entries == 2
    assert summary.candidate_total == 3
    assert summary.candidate_pending == 1
    assert summary.candidate_adopted == 1
    assert summary.candidate_rejected == 1
    assert summary.promoted_total == 2
    assert summary.promoted_manual == 1
    assert summary.managed_active == 1
    assert summary.managed_dormant == 1
    assert summary.managed_rejected == 1
    assert summary.deployed_total == 1
    assert summary.reject_reasons == [("feature cap 50: dxy", 1)]
