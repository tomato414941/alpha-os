from __future__ import annotations

import pytest


def test_trading_environment_round_trips_turnover_cost_rate():
    from alpha_os.evaluation_cost_config import TradingEnvironment

    environment = TradingEnvironment.from_document({"turnover_cost_rate": 0.001})

    assert environment.turnover_cost_rate == pytest.approx(0.001)
    assert environment.to_document()["turnover_cost_rate"] == pytest.approx(0.001)
