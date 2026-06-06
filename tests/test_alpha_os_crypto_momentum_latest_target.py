from __future__ import annotations

import sys
from pathlib import Path


def test_crypto_momentum_baseline_latest_target_snapshot():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from strategies.crypto_momentum_baseline.latest_target import (
        latest_target_snapshot,
    )

    snapshot = latest_target_snapshot()

    assert snapshot.strategy == "crypto_momentum_baseline"
    assert snapshot.mode == "manual_paper"
    assert snapshot.timestamp == "2025-12-31T00:00:00+00:00"
    assert set(snapshot.target.target_weights) <= {"BTCUSDT", "ETHUSDT"}
