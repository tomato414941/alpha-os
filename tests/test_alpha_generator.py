from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from alpha_os.legacy.managed_alphas import AlphaRecord, ManagedAlphaStore
from alpha_os.config import Config
from alpha_os.daemon.alpha_generator import (
    enqueue_discovery_pool_candidates,
)
from alpha_os.evolution.discovery_pool import DiscoveryPool
from alpha_os.dsl.expr import Feature


def test_enqueue_discovery_pool_candidates_selects_top_entries(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os.daemon.alpha_generator.asset_data_dir", lambda asset: tmp_path)
    cfg = Config()
    cfg.alpha_generator.promote_per_round = 2
    pool = DiscoveryPool()
    pool.add(Feature("f1"), 0.4, np.array([5.0, 0.3, -0.1, 0.1]))
    pool.add(Feature("f2"), 1.4, np.array([20.0, 0.7, 0.3, -0.2]))
    pool.add(Feature("f3"), 0.9, np.array([35.0, 0.9, 0.6, 0.4]))
    pool.save_to_db(Path(tmp_path) / "discovery_pool.db")

    selected, inserted = enqueue_discovery_pool_candidates("BTC", cfg)

    conn = sqlite3.connect(tmp_path / "alpha_registry.db")
    try:
        rows = conn.execute(
            "SELECT expression, fitness, behavior_json FROM candidates ORDER BY fitness DESC"
        ).fetchall()
    finally:
        conn.close()

    assert selected == 2
    assert inserted == 2
    assert [row[0] for row in rows] == ["f2", "f3"]
    assert '"enqueue": "manual_discovery_pool"' in rows[0][2]


def test_enqueue_discovery_pool_candidates_uses_path_b_saved_fitness(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os.daemon.alpha_generator.asset_data_dir", lambda asset: tmp_path)
    cfg = Config()
    cfg.alpha_generator.promote_per_round = 2
    pool = DiscoveryPool()
    signal = np.random.randn(100)
    pool.store_candidate(Feature("f1"), np.array([5.0, 0.3, -0.1, 0.1]), signal, fitness=0.4)
    pool.store_candidate(Feature("f2"), np.array([20.0, 0.7, 0.3, -0.2]), signal, fitness=1.4)
    pool.store_candidate(Feature("f3"), np.array([35.0, 0.9, 0.6, 0.4]), signal, fitness=0.9)
    pool.save_to_db(Path(tmp_path) / "discovery_pool.db")

    selected, inserted = enqueue_discovery_pool_candidates("BTC", cfg)

    conn = sqlite3.connect(tmp_path / "alpha_registry.db")
    try:
        rows = conn.execute(
            "SELECT expression, fitness FROM candidates ORDER BY fitness DESC"
        ).fetchall()
    finally:
        conn.close()

    assert selected == 2
    assert inserted == 2
    assert [row[0] for row in rows] == ["f2", "f3"]


def test_enqueue_discovery_pool_candidates_skips_semantic_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os.daemon.alpha_generator.asset_data_dir", lambda asset: tmp_path)
    cfg = Config()
    cfg.alpha_generator.promote_per_round = 3

    store = ManagedAlphaStore(tmp_path / "alpha_registry.db")
    try:
        store.register(
            AlphaRecord(
                alpha_id="a1",
                expression="(corr_10 nasdaq (ts_max_5 russell2000))",
                state="active",
            )
        )
    finally:
        store.close()

    pool = DiscoveryPool()
    signal = np.random.randn(100)
    pool.store_candidate(
        Feature("f1"),
        np.array([5.0, 0.3, -0.1, 0.1]),
        signal,
        fitness=1.4,
    )
    pool.store_candidate(
        Feature("f2"),
        np.array([20.0, 0.7, 0.3, -0.2]),
        signal,
        fitness=1.2,
    )
    pool.save_to_db(Path(tmp_path) / "discovery_pool.db")

    monkeypatch.setattr(
        "alpha_os.daemon.alpha_generator.to_string",
        lambda expr: (
            "(corr_10 (ts_max_5 russell2000) nasdaq)"
            if repr(expr) == "f1"
            else "f2"
        ),
    )

    selected, inserted = enqueue_discovery_pool_candidates("BTC", cfg)

    conn = sqlite3.connect(tmp_path / "alpha_registry.db")
    try:
        rows = conn.execute(
            "SELECT expression FROM candidates ORDER BY expression ASC"
        ).fetchall()
    finally:
        conn.close()

    assert selected == 2
    assert inserted == 1
    assert [row[0] for row in rows] == ["f2"]


def test_enqueue_discovery_pool_candidates_recomputes_zero_fitness(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os.daemon.alpha_generator.asset_data_dir", lambda asset: tmp_path)
    monkeypatch.setattr(
        "alpha_os.daemon.alpha_generator.build_feature_list",
        lambda asset: ["btc_ohlcv"],
    )
    monkeypatch.setattr(
        "alpha_os.daemon.alpha_generator._load_generator_data",
        lambda asset, config, features: (
            {"btc_ohlcv": np.array([5.0, 0.3, -0.1, 0.1])},
            np.array([5.0, 0.3, -0.1, 0.1]),
            ["btc_ohlcv"],
        ),
    )
    scores = {"f1": 0.3, "f2": 1.2, "f3": 0.8}
    monkeypatch.setattr(
        "alpha_os.daemon.alpha_generator._score_expression",
        lambda expression, data, prices, config, benchmark_returns=None: scores[expression],
    )

    cfg = Config()
    cfg.alpha_generator.promote_per_round = 2
    pool = DiscoveryPool()
    signal = np.random.randn(100)
    pool.store_candidate(Feature("f1"), np.array([5.0, 0.3, -0.1, 0.1]), signal)
    pool.store_candidate(Feature("f2"), np.array([20.0, 0.7, 0.3, -0.2]), signal)
    pool.store_candidate(Feature("f3"), np.array([35.0, 0.9, 0.6, 0.4]), signal)
    pool.save_to_db(Path(tmp_path) / "discovery_pool.db")

    selected, inserted = enqueue_discovery_pool_candidates("BTC", cfg)

    conn = sqlite3.connect(tmp_path / "alpha_registry.db")
    try:
        rows = conn.execute(
            "SELECT expression, fitness FROM candidates ORDER BY fitness DESC"
        ).fetchall()
    finally:
        conn.close()

    assert selected == 2
    assert inserted == 2
    assert [row[0] for row in rows] == ["f2", "f3"]
