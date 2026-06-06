from __future__ import annotations

import sys
from pathlib import Path


def test_crypto_momentum_baseline_latest_manual_paper_decision():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from strategies.crypto_momentum_baseline.decision import (
        latest_manual_paper_decision,
    )

    decision = latest_manual_paper_decision()

    assert decision.strategy == "crypto_momentum_baseline"
    assert decision.mode == "manual_paper"
    assert decision.timestamp == "2025-12-31T00:00:00+00:00"
    assert set(decision.target.target_weights) <= {"BTCUSDT", "ETHUSDT"}
