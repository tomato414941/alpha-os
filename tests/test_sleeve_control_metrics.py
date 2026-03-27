from __future__ import annotations

from types import SimpleNamespace


def test_load_sleeve_control_metrics_uses_current_summary_and_snapshot_history(monkeypatch):
    from alpha_os_recovery.hypotheses.sleeve_control_metrics import load_sleeve_control_metrics

    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.sleeve_control_metrics.HypothesisStore",
        lambda *_args, **_kwargs: SimpleNamespace(
            list_observation_active=lambda **_kwargs: [object()],
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.sleeve_control_metrics.build_asset_sleeve_summary",
        lambda _records: SimpleNamespace(
            serious_template_gaps=[
                "macro_template:1.00",
                "price_template:0.50",
            ],
            serious_template_backed_count=3,
            serious_template_target_count=6,
            capital_backed=9,
            actionable_live=18,
        ),
    )
    monkeypatch.setattr(
        "alpha_os_recovery.hypotheses.sleeve_control_metrics.load_recent_sleeve_compare_history",
        lambda *_args, **_kwargs: {
            "ETH": [
                {
                    "serious_template_backed": 6,
                    "breadth": 8.0,
                },
                {
                    "serious_template_backed": 4,
                    "breadth": 7.5,
                },
            ]
        },
    )

    metrics = load_sleeve_control_metrics(asset="eth")

    assert metrics.asset == "ETH"
    assert metrics.template_gap_count == 2
    assert metrics.template_gaps == [
        "macro_template:1.00",
        "price_template:0.50",
    ]
    assert metrics.coverage_retention == 0.5
    assert metrics.capital_conversion == 0.5
    assert metrics.breadth_trend == 0.5
