from __future__ import annotations

import numpy as np

from alpha_os.alpha.managed_alphas import ManagedAlphaStore
from alpha_os.config import Config
from alpha_os.forward.tracker import ForwardTracker
from alpha_os.governance.audit_log import AuditLog
from alpha_os.paper.tracker import PaperPortfolioTracker
from alpha_os.paper.trader import Trader


class _DummyStore:
    def close(self) -> None:
        return None


def test_trader_recomputes_diversity_when_cache_is_degenerate(tmp_path):
    trader = Trader(
        asset="BTC",
        config=Config(),
        portfolio_tracker=PaperPortfolioTracker(tmp_path / "paper.db"),
        registry=ManagedAlphaStore(tmp_path / "reg.db"),
        forward_tracker=ForwardTracker(tmp_path / "fwd.db"),
        audit_log=AuditLog(tmp_path / "audit.jsonl"),
        store=_DummyStore(),
    )

    base = np.linspace(-1.0, 1.0, 80)
    alpha_ids = ["a", "b", "c"]
    alpha_signal_arrays = {
        "a": base,
        "b": base + 0.001,
        "c": np.tile([1.0, -1.0], 40),
    }
    trader._diversity_cache = {aid: 1.0 for aid in alpha_ids}

    scores = trader._resolve_diversity_scores(alpha_ids, alpha_signal_arrays)

    assert scores["c"] > scores["a"]
    assert scores["c"] > scores["b"]
    trader.close()
