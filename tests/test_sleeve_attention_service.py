from __future__ import annotations

from types import SimpleNamespace


def test_build_sleeve_attention_plan_returns_high_for_unhealthy_sleeve(monkeypatch):
    from alpha_os.hypotheses.sleeve_attention_service import build_sleeve_attention_plan

    monkeypatch.setattr(
        "alpha_os.hypotheses.sleeve_attention_service.load_sleeve_control_metrics",
        lambda *, asset: SimpleNamespace(
            asset=asset,
            template_gap_count=2,
            coverage_retention=0.4,
            capital_conversion=0.1,
            breadth_trend=-0.6,
        ),
    )

    plan = build_sleeve_attention_plan(
        asset="ETH",
        config=SimpleNamespace(
            cross_sectional=SimpleNamespace(registry_asset="BTC"),
        ),
    )

    assert plan.asset == "ETH"
    assert plan.level == "high"
    assert plan.maintenance_lookback_days == 30
    assert plan.rebalance_required is True


def test_build_sleeve_attention_plan_returns_light_for_healthy_reference(monkeypatch):
    from alpha_os.hypotheses.sleeve_attention_service import build_sleeve_attention_plan

    monkeypatch.setattr(
        "alpha_os.hypotheses.sleeve_attention_service.load_sleeve_control_metrics",
        lambda *, asset: SimpleNamespace(
            asset=asset,
            template_gap_count=0,
            coverage_retention=1.0,
            capital_conversion=0.8,
            breadth_trend=0.2,
        ),
    )

    plan = build_sleeve_attention_plan(
        asset="BTC",
        config=SimpleNamespace(
            cross_sectional=SimpleNamespace(registry_asset="BTC"),
        ),
    )

    assert plan.asset == "BTC"
    assert plan.level == "light"
    assert plan.maintenance_lookback_days == 14
    assert plan.rebalance_required is True
