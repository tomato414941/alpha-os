from __future__ import annotations

from dataclasses import dataclass

from .portfolio_decision import PortfolioTarget


@dataclass(frozen=True)
class ExecutionPolicySpec:
    no_trade_band: float = 0.0
    turnover_budget: float | None = None

    def __post_init__(self) -> None:
        if self.no_trade_band < 0.0:
            raise ValueError("execution_policy.no_trade_band must be >= 0")
        if self.turnover_budget is not None and self.turnover_budget < 0.0:
            raise ValueError("execution_policy.turnover_budget must be >= 0")


@dataclass(frozen=True)
class SubjectTradeTransition:
    subject_id: str
    current_weight: float
    desired_weight: float
    executed_weight: float
    desired_delta: float
    executed_delta: float
    skipped: bool
    reason: str
    expected_trade_cost: float = 0.0
    rejected_reason: str | None = None


@dataclass(frozen=True)
class TradeTransitionTrace:
    desired_turnover: float
    executed_turnover: float
    turnover_suppression: float
    skipped_trade_count: int
    expected_execution_cost: float
    turnover_budget: float | None
    subjects: tuple[SubjectTradeTransition, ...]


@dataclass(frozen=True)
class TradeTransitionRequest:
    desired_targets: dict[str, PortfolioTarget]
    current_weights: dict[str, float]
    capital_base: float
    execution_policy: ExecutionPolicySpec
    per_turnover_cost: float = 0.0


@dataclass(frozen=True)
class TradeTransitionResult:
    executed_targets: dict[str, PortfolioTarget]
    trace: TradeTransitionTrace


@dataclass(frozen=True)
class _TradeTransitionCandidate:
    subject_id: str
    current_weight: float
    desired_weight: float
    desired_delta: float
    adjusted_delta: float
    risk_scale: float
    expected_trade_cost: float
    rejected_reason: str | None


def apply_execution_policy(request: TradeTransitionRequest) -> TradeTransitionResult:
    policy = request.execution_policy
    all_subjects = sorted(
        set(request.desired_targets) | set(request.current_weights)
    )
    candidates: list[_TradeTransitionCandidate] = []
    for subject_id in all_subjects:
        target = request.desired_targets.get(subject_id)
        current_weight = float(request.current_weights.get(subject_id, 0.0))
        desired_weight = current_weight if target is None else float(target.target_weight)
        desired_delta = desired_weight - current_weight
        adjusted_delta = _execution_delta(
            desired_delta,
            no_trade_band=policy.no_trade_band,
        )
        expected_trade_cost = _expected_trade_cost(
            adjusted_delta,
            capital_base=request.capital_base,
            per_turnover_cost=request.per_turnover_cost,
        )
        rejected_reason = _candidate_rejected_reason(
            desired_delta=desired_delta,
            adjusted_delta=adjusted_delta,
        )
        candidates.append(
            _TradeTransitionCandidate(
                subject_id=subject_id,
                current_weight=current_weight,
                desired_weight=desired_weight,
                desired_delta=desired_delta,
                adjusted_delta=adjusted_delta,
                risk_scale=1.0 if target is None else float(target.risk_scale),
                expected_trade_cost=expected_trade_cost,
                rejected_reason=rejected_reason,
            )
        )

    desired_turnover = sum(
        abs(candidate.desired_delta) for candidate in candidates
    )
    executed_delta_by_subject, budget_rejected_subjects = (
        _executed_deltas_from_candidates(candidates, policy=policy)
    )

    executed_targets: dict[str, PortfolioTarget] = {}
    transitions: list[SubjectTradeTransition] = []
    for candidate in candidates:
        executed_delta = executed_delta_by_subject.get(candidate.subject_id, 0.0)
        executed_weight = candidate.current_weight + executed_delta
        reason = _transition_reason(
            desired_delta=candidate.desired_delta,
            adjusted_delta=candidate.adjusted_delta,
            executed_delta=executed_delta,
            rejected_reason=(
                "turnover_budget"
                if candidate.subject_id in budget_rejected_subjects
                else candidate.rejected_reason
            ),
        )
        skipped = abs(executed_delta) == 0.0 and abs(candidate.desired_delta) > 0.0
        transitions.append(
            SubjectTradeTransition(
                subject_id=candidate.subject_id,
                current_weight=candidate.current_weight,
                desired_weight=candidate.desired_weight,
                executed_weight=float(executed_weight),
                desired_delta=candidate.desired_delta,
                executed_delta=executed_delta,
                skipped=skipped,
                reason=reason,
                expected_trade_cost=candidate.expected_trade_cost,
                rejected_reason=reason if skipped else None,
            )
        )
        executed_targets[candidate.subject_id] = PortfolioTarget(
            subject_id=candidate.subject_id,
            target_weight=float(executed_weight),
            position_delta=float(executed_delta),
            target_notional=float(executed_weight * request.capital_base),
            entry_allowed=(
                abs(candidate.current_weight) > 0.0 or abs(executed_weight) > 0.0
            ),
            risk_scale=candidate.risk_scale,
        )

    executed_turnover = sum(abs(item.executed_delta) for item in transitions)
    trace = TradeTransitionTrace(
        desired_turnover=float(desired_turnover),
        executed_turnover=float(executed_turnover),
        turnover_suppression=float(max(desired_turnover - executed_turnover, 0.0)),
        skipped_trade_count=sum(1 for item in transitions if item.skipped),
        expected_execution_cost=float(
            executed_turnover
            * max(float(request.capital_base), 0.0)
            * max(float(request.per_turnover_cost), 0.0)
        ),
        turnover_budget=policy.turnover_budget,
        subjects=tuple(transitions),
    )
    return TradeTransitionResult(
        executed_targets=executed_targets,
        trace=trace,
    )


def _executed_deltas_from_candidates(
    candidates: list[_TradeTransitionCandidate],
    *,
    policy: ExecutionPolicySpec,
) -> tuple[dict[str, float], set[str]]:
    return _budget_scaled_executed_deltas(candidates, policy=policy)


def _budget_scaled_executed_deltas(
    candidates: list[_TradeTransitionCandidate],
    *,
    policy: ExecutionPolicySpec,
) -> tuple[dict[str, float], set[str]]:
    adjusted_turnover = sum(abs(item.adjusted_delta) for item in candidates)
    budget_scale = 1.0
    if (
        policy.turnover_budget is not None
        and policy.turnover_budget >= 0.0
        and adjusted_turnover > policy.turnover_budget
        and adjusted_turnover > 0.0
    ):
        budget_scale = float(policy.turnover_budget) / float(adjusted_turnover)
    return (
        {
            item.subject_id: float(item.adjusted_delta * budget_scale)
            for item in candidates
        },
        set(),
    )


def _candidate_rejected_reason(
    *,
    desired_delta: float,
    adjusted_delta: float,
) -> str | None:
    if abs(desired_delta) == 0.0:
        return None
    if abs(adjusted_delta) == 0.0:
        return "threshold"
    return None


def _expected_trade_cost(
    adjusted_delta: float,
    *,
    capital_base: float,
    per_turnover_cost: float,
) -> float:
    return float(
        abs(adjusted_delta)
        * max(float(capital_base), 0.0)
        * max(float(per_turnover_cost), 0.0)
    )


def _execution_delta(
    desired_delta: float,
    *,
    no_trade_band: float,
) -> float:
    if abs(desired_delta) <= no_trade_band:
        return 0.0
    return float(desired_delta)


def _transition_reason(
    *,
    desired_delta: float,
    adjusted_delta: float,
    executed_delta: float,
    rejected_reason: str | None,
) -> str:
    if abs(desired_delta) == 0.0:
        return "unchanged"
    if rejected_reason is not None:
        return rejected_reason
    if abs(adjusted_delta) == 0.0:
        return "threshold"
    if abs(executed_delta) == 0.0:
        return "turnover_budget"
    if abs(executed_delta) < abs(adjusted_delta):
        return "turnover_budget"
    return "executed"
