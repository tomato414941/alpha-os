from __future__ import annotations


def test_build_strategy_adaptation_state_creates_reputations_from_survivors():
    from alpha_os.evaluation_report import (
        EvaluationTaskResult,
        EvaluationMetricGroupResult,
    )
    from alpha_os.strategy_adaptation import build_strategy_adaptation_state
    from alpha_os.screening import (
        ScreeningCandidateResult,
        ScreeningPolicy,
        ScreeningResult,
    )

    screening_result = ScreeningResult(
        screening_result_id="discovery:screen",
        signal_discovery_id="discovery",
        policy=ScreeningPolicy(),
        candidates=(
            ScreeningCandidateResult(
                signal_id="trend@AAPL",
                specification_signal_id="trend",
                family_id="trend_family",
                subject_id="AAPL_equity",
                target_id="residual_return_3d",
                kind="momentum",
                lookback=40,
                score=0.2,
                corr=-0.2,
                stability_score=0.9,
                sample_count=20,
                keep=True,
                family_rank=1,
                reasons=(),
            ),
            ScreeningCandidateResult(
                signal_id="reversion@ABBV",
                specification_signal_id="reversion",
                family_id="reversion_family",
                subject_id="ABBV_equity",
                target_id="residual_return_3d",
                kind="reversal",
                lookback=20,
                score=0.3,
                corr=0.3,
                stability_score=1.2,
                sample_count=25,
                keep=True,
                family_rank=1,
                reasons=(),
            ),
        ),
        created_at="2026-04-03T00:00:00+00:00",
    )
    task_result = EvaluationTaskResult(
        evaluation_task_id="case:discovery",
        strategy_id="strategy:discovery",
        signal_discovery_id="discovery",
        metric_group_results=(
            EvaluationMetricGroupResult(
                metric_group_name="signed_belief_quality",
                source="native_plan",
                metrics={"mean_survivor_corr": 0.05},
            ),
        ),
        failure_finding_groups=(),
        artifact_refs={"screening_result_ids": ("discovery:screen",)},
    )

    state = build_strategy_adaptation_state(
        evaluation_report_id="report-1",
        task_result=task_result,
        screening_result=screening_result,
        metrics_by_signal_id=None,
        previous_state=None,
        smoothing=0.5,
        created_at="2026-04-03T00:00:00+00:00",
    )

    assert state.strategy_id == "strategy:discovery"
    assert state.signal_discovery_id == "discovery"
    assert state.source_evaluation_report_id == "report-1"
    assert len(state.signal_reputations) == 2
    assert len(state.family_reputations) == 2
    trend = state.signal_reputations[0]
    assert trend.signal_id == "trend@AAPL"
    assert trend.orientation == -1
    assert round(trend.edge_score, 6) == 0.2
    assert round(trend.confidence, 6) == 0.894427


def test_build_strategy_adaptation_state_smooths_against_previous_state():
    from alpha_os.evaluation_report import EvaluationTaskResult
    from alpha_os.strategy_adaptation import (
        FamilyReputation,
        StrategyAdaptationState,
        SignalReputation,
        build_strategy_adaptation_state,
    )
    from alpha_os.screening import (
        ScreeningCandidateResult,
        ScreeningPolicy,
        ScreeningResult,
    )

    previous_state = StrategyAdaptationState(
        strategy_id="strategy:discovery",
        signal_discovery_id="discovery",
        source_evaluation_report_id="report-0",
        source_screening_result_id="discovery:screen:0",
        signal_reputations=(
            SignalReputation(
                signal_id="trend@AAPL",
                family_id="trend_family",
                subject_id="AAPL_equity",
                target_id="residual_return_3d",
                orientation=1,
                edge_score=0.4,
                mmc=0.1,
                contribution_score=0.25,
                confidence=0.8,
                stability_score=1.5,
                sample_count=10,
                update_count=1,
                updated_at="2026-04-02T00:00:00+00:00",
            ),
        ),
        family_reputations=(
            FamilyReputation(
                family_id="trend_family",
                mean_edge_score=0.4,
                mean_confidence=0.8,
                mean_stability_score=1.5,
                subject_coverage=1,
                member_count=1,
                update_count=1,
                updated_at="2026-04-02T00:00:00+00:00",
            ),
        ),
        created_at="2026-04-02T00:00:00+00:00",
    )
    screening_result = ScreeningResult(
        screening_result_id="discovery:screen:1",
        signal_discovery_id="discovery",
        policy=ScreeningPolicy(),
        candidates=(
            ScreeningCandidateResult(
                signal_id="trend@AAPL",
                specification_signal_id="trend",
                family_id="trend_family",
                subject_id="AAPL_equity",
                target_id="residual_return_3d",
                kind="momentum",
                lookback=40,
                score=0.2,
                corr=0.2,
                stability_score=1.0,
                sample_count=20,
                keep=True,
                family_rank=1,
                reasons=(),
            ),
        ),
        created_at="2026-04-03T00:00:00+00:00",
    )
    task_result = EvaluationTaskResult(
        evaluation_task_id="case:discovery",
        strategy_id="strategy:discovery",
        signal_discovery_id="discovery",
        metric_group_results=(),
        failure_finding_groups=(),
        artifact_refs={"screening_result_ids": ("discovery:screen:1",)},
    )

    state = build_strategy_adaptation_state(
        evaluation_report_id="report-1",
        task_result=task_result,
        screening_result=screening_result,
        metrics_by_signal_id=None,
        previous_state=previous_state,
        smoothing=0.5,
        created_at="2026-04-03T00:00:00+00:00",
    )

    assert state.strategy_id == "strategy:discovery"
    reputation = state.signal_reputations[0]
    family = state.family_reputations[0]
    assert round(reputation.edge_score, 6) == 0.3
    assert round(reputation.confidence, 6) == 0.847214
    assert round(family.mean_edge_score, 6) == 0.35
    assert family.update_count == 2


def test_build_strategy_adaptation_state_uses_mmc_to_build_contribution_score():
    from types import SimpleNamespace

    from alpha_os.evaluation_report import EvaluationTaskResult
    from alpha_os.strategy_adaptation import build_strategy_adaptation_state
    from alpha_os.screening import ScreeningCandidateResult, ScreeningPolicy, ScreeningResult

    screening_result = ScreeningResult(
        screening_result_id="discovery:screen",
        signal_discovery_id="discovery",
        policy=ScreeningPolicy(),
        candidates=(
            ScreeningCandidateResult(
                signal_id="trend@AAPL",
                specification_signal_id="trend",
                family_id="trend_family",
                subject_id="AAPL_equity",
                target_id="residual_return_3d",
                kind="momentum",
                lookback=40,
                score=0.2,
                corr=0.2,
                stability_score=0.9,
                sample_count=20,
                keep=True,
                family_rank=1,
                reasons=(),
            ),
        ),
        created_at="2026-04-03T00:00:00+00:00",
    )
    task_result = EvaluationTaskResult(
        evaluation_task_id="case:discovery",
        strategy_id="strategy:discovery",
        signal_discovery_id="discovery",
        metric_group_results=(),
        failure_finding_groups=(),
        artifact_refs={"screening_result_ids": ("discovery:screen",)},
    )
    metrics_by_signal_id = {
        "trend@AAPL": SimpleNamespace(mmc=0.1),
    }

    state = build_strategy_adaptation_state(
        evaluation_report_id="report-1",
        task_result=task_result,
        screening_result=screening_result,
        metrics_by_signal_id=metrics_by_signal_id,
        previous_state=None,
        smoothing=0.5,
        created_at="2026-04-03T00:00:00+00:00",
    )

    assert state.strategy_id == "strategy:discovery"
    reputation = state.signal_reputations[0]
    assert round(reputation.mmc or 0.0, 6) == 0.1
    assert round(reputation.contribution_score, 6) == 0.15


def test_strategy_adaptation_state_roundtrip_allows_missing_signal_discovery_id():
    from alpha_os.strategy_adaptation import StrategyAdaptationState

    state = StrategyAdaptationState(
        strategy_id="strategy:constant_hold",
        signal_discovery_id=None,
        source_evaluation_report_id="report-1",
        source_screening_result_id="screen-1",
        signal_reputations=(),
        family_reputations=(),
        created_at="2026-04-03T00:00:00+00:00",
    )

    loaded = StrategyAdaptationState.from_document(
        strategy_id=state.strategy_id,
        document=state.to_document(),
    )

    assert loaded.strategy_id == "strategy:constant_hold"
    assert loaded.signal_discovery_id is None
