from __future__ import annotations

from types import SimpleNamespace

from alpha_os.hypotheses.trade_transition_diagnostics import (
    build_trade_transition_summary,
    capture_batch_transition_snapshot,
)


def _record(
    hypothesis_id: str,
    *,
    expression: str,
    stake: float,
    metadata: dict,
):
    return SimpleNamespace(
        hypothesis_id=hypothesis_id,
        expression=expression,
        source="random_dsl",
        stake=stake,
        metadata=metadata,
    )


def test_build_trade_transition_summary_reports_entry_and_exit_reasons():
    pre = capture_batch_transition_snapshot(
        [
            _record(
                "exit_me",
                expression="btc_difficulty",
                stake=0.04,
                metadata={
                    "research_quality_source": "batch_research_score",
                    "lifecycle_capital_eligible": True,
                    "lifecycle_research_retained": True,
                    "lifecycle_blended_quality": 0.3,
                },
            ),
            _record(
                "stay_out",
                expression="oi_btc_1h",
                stake=0.0,
                metadata={
                    "research_quality_source": "batch_research_score",
                    "lifecycle_research_retained": True,
                    "lifecycle_blended_quality": 0.2,
                },
            ),
        ],
        families=("onchain", "derivatives"),
    )
    post = capture_batch_transition_snapshot(
        [
            _record(
                "exit_me",
                expression="btc_difficulty",
                stake=0.0,
                metadata={
                    "research_quality_source": "batch_research_score",
                    "lifecycle_research_retained": True,
                    "lifecycle_blended_quality": 0.3,
                    "lifecycle_live_promotion_blocker": "weak_live_quality",
                },
            ),
            _record(
                "enter_me",
                expression="oi_btc_1h",
                stake=0.05,
                metadata={
                    "research_quality_source": "batch_research_score",
                    "lifecycle_capital_eligible": True,
                    "lifecycle_research_retained": True,
                    "lifecycle_capital_reason": "research_backed",
                    "lifecycle_blended_quality": 0.4,
                },
            ),
        ],
        families=("onchain", "derivatives"),
    )

    summary = build_trade_transition_summary(pre, post, top=5)

    assert summary["pre_backed"] == 1
    assert summary["post_backed"] == 1
    assert summary["entered"] == 1
    assert summary["exited"] == 1
    assert summary["entry_reasons"]["research_backed"] == 1
    assert summary["exit_reasons"]["live_q"] == 1
    assert any("enter_me" in entry for entry in summary["top_entries"])
    assert any("exit_me" in entry for entry in summary["top_exits"])
