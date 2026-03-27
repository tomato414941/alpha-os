from __future__ import annotations

from types import SimpleNamespace


def test_build_template_gap_search_budget_returns_requested_limit_without_templates(monkeypatch):
    from alpha_os_recovery.hypotheses.search_budget_service import build_template_gap_search_budget

    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.search_budget_service.serious_seed_specs",
        lambda asset: [],
    )

    budget = build_template_gap_search_budget(asset="BTC", base_limit=12)

    assert budget.asset == "BTC"
    assert budget.requested_limit == 12
    assert budget.effective_limit == 12
    assert budget.missing_template_count == 0
    assert budget.coverage_retention == 1.0


def test_build_template_gap_search_budget_closes_when_sleeve_is_healthy(monkeypatch):
    from alpha_os_recovery.hypotheses.search_budget_service import build_template_gap_search_budget

    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.search_budget_service.serious_seed_specs",
        lambda asset: [object()],
    )
    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.search_budget_service.load_sleeve_control_metrics",
        lambda *, asset: SimpleNamespace(
            asset=asset,
            template_gap_count=0,
            template_gaps=[],
            serious_template_backed_count=6,
            serious_template_target_count=6,
            coverage_retention=1.0,
            capital_conversion=0.75,
            breadth_trend=0.2,
        ),
    )

    budget = build_template_gap_search_budget(asset="ETH", base_limit=12)

    assert budget.asset == "ETH"
    assert budget.requested_limit == 12
    assert budget.effective_limit == 0
    assert budget.missing_template_count == 0


def test_build_template_gap_search_budget_scales_with_template_gaps(monkeypatch):
    from alpha_os_recovery.hypotheses.search_budget_service import build_template_gap_search_budget

    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.search_budget_service.serious_seed_specs",
        lambda asset: [object()],
    )
    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.search_budget_service.load_sleeve_control_metrics",
        lambda *, asset: SimpleNamespace(
            asset=asset,
            template_gap_count=2,
            template_gaps=[
                "macro_template:1.00",
                "price_template:0.50",
            ],
            serious_template_backed_count=1,
            serious_template_target_count=3,
            coverage_retention=0.5,
            capital_conversion=0.4,
            breadth_trend=-0.1,
        ),
    )

    budget = build_template_gap_search_budget(asset="ETH", base_limit=12)

    assert budget.asset == "ETH"
    assert budget.requested_limit == 12
    assert budget.effective_limit == 8
    assert budget.missing_template_count == 2
    assert budget.closed_template_count == 0
    assert budget.new_template_count == 2


def test_build_template_gap_search_budget_boosts_stalled_sleeves(monkeypatch):
    from alpha_os_recovery.hypotheses.search_budget_service import build_template_gap_search_budget

    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.search_budget_service.serious_seed_specs",
        lambda asset: [object()],
    )
    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.search_budget_service.load_sleeve_control_metrics",
        lambda *, asset: SimpleNamespace(
            asset=asset,
            template_gap_count=2,
            template_gaps=[
                "macro_template:1.00",
                "price_template:0.50",
            ],
            serious_template_backed_count=1,
            serious_template_target_count=3,
            coverage_retention=0.2,
            capital_conversion=0.1,
            breadth_trend=-0.8,
        ),
    )

    budget = build_template_gap_search_budget(
        asset="ETH",
        base_limit=12,
        previous_template_gaps=[
            "macro_template:1.00",
            "price_template:0.50",
        ],
    )

    assert budget.asset == "ETH"
    assert budget.requested_limit == 12
    assert budget.effective_limit == 12
    assert budget.missing_template_count == 2
    assert budget.closed_template_count == 0
    assert budget.new_template_count == 0
    assert budget.coverage_retention == 0.2
    assert budget.capital_conversion == 0.1
