from __future__ import annotations

from types import SimpleNamespace


def test_build_template_gap_search_budget_returns_requested_limit_without_templates(monkeypatch):
    from alpha_os.hypotheses.search_budget_service import build_template_gap_search_budget

    monkeypatch.setattr(
        "alpha_os.hypotheses.search_budget_service.serious_seed_specs",
        lambda asset: [],
    )

    budget = build_template_gap_search_budget(asset="BTC", base_limit=12)

    assert budget.asset == "BTC"
    assert budget.requested_limit == 12
    assert budget.effective_limit == 12
    assert budget.missing_template_count == 0


def test_build_template_gap_search_budget_closes_when_no_gap(monkeypatch):
    from alpha_os.hypotheses.search_budget_service import build_template_gap_search_budget

    monkeypatch.setattr(
        "alpha_os.hypotheses.search_budget_service.serious_seed_specs",
        lambda asset: [object()],
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.search_budget_service.HypothesisStore",
        lambda *_args, **_kwargs: SimpleNamespace(
            list_observation_active=lambda **_kwargs: [],
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.search_budget_service.serious_template_gap_scores",
        lambda asset, records: {"macro_template": 0.0},
    )

    budget = build_template_gap_search_budget(asset="ETH", base_limit=12)

    assert budget.asset == "ETH"
    assert budget.requested_limit == 12
    assert budget.effective_limit == 0
    assert budget.missing_template_count == 0


def test_build_template_gap_search_budget_scales_with_missing_templates(monkeypatch):
    from alpha_os.hypotheses.search_budget_service import build_template_gap_search_budget

    monkeypatch.setattr(
        "alpha_os.hypotheses.search_budget_service.serious_seed_specs",
        lambda asset: [object()],
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.search_budget_service.HypothesisStore",
        lambda *_args, **_kwargs: SimpleNamespace(
            list_observation_active=lambda **_kwargs: [],
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.search_budget_service.serious_template_gap_scores",
        lambda asset, records: {
            "macro_template": 1.0,
            "price_template": 0.5,
            "derivatives_template": 0.0,
        },
    )

    budget = build_template_gap_search_budget(asset="ETH", base_limit=12)

    assert budget.asset == "ETH"
    assert budget.requested_limit == 12
    assert budget.effective_limit == 4
    assert budget.missing_template_count == 2
