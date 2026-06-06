from __future__ import annotations

import pytest

from conftest import load_example_module


def test_alpha_model_can_return_alpha_score():
    example = load_example_module("examples/alpha_model_score.py")

    score = example.score_alpha(
        example.MomentumAlphaModel(),
        example.FeatureBatch(
            features_by_symbol={
                "BTC": {"return_7d": 0.04},
                "ETH": {"return_7d": -0.02},
            }
        ),
    )

    assert score == example.AlphaScore(scores={"BTC": 0.04, "ETH": -0.02})


def test_risk_model_can_estimate_exposure():
    example = load_example_module("examples/risk_model_exposure.py")

    estimate = example.estimate_risk(
        example.ExposureRiskModel(),
        example.PortfolioSnapshot(
            notionals={"BTC": 600.0, "ETH": -200.0},
            equity=1_000.0,
        ),
    )

    assert estimate.gross_exposure == pytest.approx(0.8)
    assert estimate.net_exposure == pytest.approx(0.4)


def test_portfolio_allocator_can_allocate_from_scores():
    example = load_example_module("examples/portfolio_allocator_from_scores.py")

    target = example.allocate_portfolio(
        example.LongOnlyScoreAllocator(),
        example.AlphaScore(scores={"BTC": 0.04, "ETH": 0.02, "SOL": -0.01}),
    )

    assert target.target_weights["BTC"] == pytest.approx(2.0 / 3.0)
    assert target.target_weights["ETH"] == pytest.approx(1.0 / 3.0)
    assert "SOL" not in target.target_weights


def test_execution_slicer_can_create_child_orders():
    example = load_example_module("examples/execution_order_slicer.py")

    child_orders = example.slice_order(
        example.EqualQuantityOrderSlicer(),
        example.ParentOrder(symbol="BTC", quantity=0.9, slices=3),
    )

    assert child_orders == (
        example.ChildOrder(symbol="BTC", quantity=0.3),
        example.ChildOrder(symbol="BTC", quantity=0.3),
        example.ChildOrder(symbol="BTC", quantity=0.3),
    )
