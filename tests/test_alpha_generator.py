from __future__ import annotations

import sqlite3

import numpy as np

from alpha_os.config import Config
from alpha_os.daemon.alpha_generator import AlphaGeneratorDaemon, PromotionCandidate


def test_queue_promoted_candidates_applies_limit_and_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os.daemon.alpha_generator.asset_data_dir", lambda asset: tmp_path)
    cfg = Config()
    cfg.alpha_generator.promote_per_round = 2
    cfg.alpha_generator.promotion_min_fitness = 0.5

    daemon = AlphaGeneratorDaemon(asset="BTC", config=cfg)
    inserted = daemon._queue_promoted_candidates(
        [
            PromotionCandidate("(a)", 1.2, behavior=np.array([])),
            PromotionCandidate("(b)", 0.8, behavior=np.array([])),
            PromotionCandidate("(c)", 0.4, behavior=np.array([])),
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
