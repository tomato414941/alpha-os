from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_example() -> ModuleType:
    path = Path("examples/trading_strategy_execution_intent.py")
    spec = importlib.util.spec_from_file_location(
        "trading_strategy_execution_intent", path
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_trading_strategy_can_return_execution_intent():
    example = _load_example()

    intent = example.decide_trading_intent(
        example.RiskOffStrategy(),
        example.RiskObservation(
            prices={"BTC": 100.0, "ETH": 50.0},
            current_weights={"BTC": 0.5, "ETH": 0.5},
            risk_score=0.9,
        ),
    )

    assert intent == example.TradingIntent(
        target_weights={"BTC": 0.0, "ETH": 0.0},
        execution=example.ExecutionPreference(urgency="high", order_style="market"),
    )
