from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from alpha_os.alpha.managed_alphas import AlphaRecord, ManagedAlphaStore
from alpha_os.config import Config
from alpha_os.daemon.alpha_generator import (
    AlphaGeneratorDaemon,
    AdmissionQueueCandidate,
    _current_rss_mb,
    enqueue_discovery_pool_candidates,
)
from alpha_os.evolution.discovery_pool import DiscoveryPool
from alpha_os.dsl.expr import Feature


def test_enqueue_admission_queue_candidates_applies_limit_and_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os.daemon.alpha_generator.asset_data_dir", lambda asset: tmp_path)
    cfg = Config()
    cfg.alpha_generator.promote_per_round = 2
    cfg.alpha_generator.promotion_min_fitness = 0.5

    daemon = AlphaGeneratorDaemon(asset="BTC", config=cfg)
    inserted = daemon._enqueue_admission_queue_candidates(
        [
            AdmissionQueueCandidate("(a)", 1.2, 1.2, behavior=np.array([])),
            AdmissionQueueCandidate("(b)", 0.8, 0.8, behavior=np.array([])),
            AdmissionQueueCandidate("(c)", 0.4, 0.4, behavior=np.array([])),
        ]
    )

    conn = sqlite3.connect(tmp_path / "alpha_registry.db")
    try:
        rows = conn.execute(
            "SELECT expression, fitness, behavior_json FROM candidates ORDER BY fitness DESC"
        ).fetchall()
    finally:
        conn.close()

    assert inserted == 2
    assert [row[0] for row in rows] == ["(a)", "(b)"]
    assert rows[0][1] == 1.2
    assert '"source": "alpha_generator"' in rows[0][2]
    assert '"round": 0' in rows[0][2]


def test_enqueue_discovery_pool_candidates_selects_top_entries(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os.daemon.alpha_generator.asset_data_dir", lambda asset: tmp_path)
    cfg = Config()
    cfg.alpha_generator.promote_per_round = 2
    pool = DiscoveryPool()
    pool.add(Feature("f1"), 0.4, np.array([1.0, 2.0, 3.0]))
    pool.add(Feature("f2"), 1.4, np.array([2.0, 2.0, 3.0]))
    pool.add(Feature("f3"), 0.9, np.array([3.0, 2.0, 3.0]))
    pool.save_to_db(Path(tmp_path) / "archive.db")

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
    pool.store_candidate(Feature("f1"), np.array([1.0, 2.0, 3.0]), signal, fitness=0.4)
    pool.store_candidate(Feature("f2"), np.array([2.0, 2.0, 3.0]), signal, fitness=1.4)
    pool.store_candidate(Feature("f3"), np.array([3.0, 2.0, 3.0]), signal, fitness=0.9)
    pool.save_to_db(Path(tmp_path) / "archive.db")

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
        np.array([1.0, 2.0, 3.0]),
        signal,
        fitness=1.4,
    )
    pool.store_candidate(
        Feature("f2"),
        np.array([2.0, 2.0, 3.0]),
        signal,
        fitness=1.2,
    )
    pool.save_to_db(Path(tmp_path) / "archive.db")

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


def test_check_memory_reduces_budget_when_over_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os.daemon.alpha_generator.asset_data_dir", lambda asset: tmp_path)
    cfg = Config()
    cfg.alpha_generator.pop_size = 80
    cfg.alpha_generator.memory_limit_mb = 700
    daemon = AlphaGeneratorDaemon(asset="BTC", config=cfg)
    daemon._budget = 80

    monkeypatch.setattr("alpha_os.daemon.alpha_generator._current_rss_mb", lambda: 750.0)

    daemon._check_memory()

    assert daemon._budget == 40


def test_check_memory_recovers_budget_when_below_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os.daemon.alpha_generator.asset_data_dir", lambda asset: tmp_path)
    cfg = Config()
    cfg.alpha_generator.pop_size = 80
    cfg.alpha_generator.memory_limit_mb = 700
    daemon = AlphaGeneratorDaemon(asset="BTC", config=cfg)
    daemon._budget = 20

    monkeypatch.setattr("alpha_os.daemon.alpha_generator._current_rss_mb", lambda: 400.0)

    daemon._check_memory()

    assert daemon._budget == 40


def test_current_rss_mb_falls_back_to_ru_maxrss(monkeypatch):
    class FakeUsage:
        ru_maxrss = 2048

    monkeypatch.setattr("pathlib.Path.read_text", lambda self: (_ for _ in ()).throw(OSError("x")))
    monkeypatch.setattr("resource.getrusage", lambda _: FakeUsage())

    assert _current_rss_mb() == 2.0


def test_enqueue_discovery_pool_candidates_recomputes_zero_fitness(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os.daemon.alpha_generator.asset_data_dir", lambda asset: tmp_path)
    monkeypatch.setattr(
        "alpha_os.daemon.alpha_generator.build_feature_list",
        lambda asset: ["btc_ohlcv"],
    )
    monkeypatch.setattr(
        "alpha_os.daemon.alpha_generator._load_generator_data",
        lambda asset, config, features: (
            {"btc_ohlcv": np.array([1.0, 2.0, 3.0])},
            np.array([1.0, 2.0, 3.0]),
            ["btc_ohlcv"],
        ),
    )
    scores = {"f1": 0.3, "f2": 1.2, "f3": 0.8}
    monkeypatch.setattr(
        "alpha_os.daemon.alpha_generator._score_expression",
        lambda expression, data, prices, config: scores[expression],
    )

    cfg = Config()
    cfg.alpha_generator.promote_per_round = 2
    pool = DiscoveryPool()
    signal = np.random.randn(100)
    pool.store_candidate(Feature("f1"), np.array([1.0, 2.0, 3.0]), signal)
    pool.store_candidate(Feature("f2"), np.array([2.0, 2.0, 3.0]), signal)
    pool.store_candidate(Feature("f3"), np.array([3.0, 2.0, 3.0]), signal)
    pool.save_to_db(Path(tmp_path) / "archive.db")

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
