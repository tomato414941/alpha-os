from __future__ import annotations

from dataclasses import dataclass

from .portfolio_decision import PortfolioTarget


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


@dataclass(frozen=True)
class TradeTransitionTrace:
    desired_turnover: float
    executed_turnover: float
    turnover_suppression: float
    skipped_trade_count: int
    expected_execution_cost: float
    subjects: tuple[SubjectTradeTransition, ...]


@dataclass(frozen=True)
class TradeTransitionRequest:
    desired_targets: dict[str, PortfolioTarget]
    current_weights: dict[str, float]
    capital_base: float
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
    risk_scale: float
    expected_trade_cost: float


def apply_trade_transition(request: TradeTransitionRequest) -> TradeTransitionResult:
    all_subjects = sorted(
        set(request.desired_targets) | set(request.current_weights)
    )
    candidates: list[_TradeTransitionCandidate] = []
    for subject_id in all_subjects:
        target = request.desired_targets.get(subject_id)
        current_weight = float(request.current_weights.get(subject_id, 0.0))
        desired_weight = current_weight if target is None else float(target.target_weight)
        desired_delta = desired_weight - current_weight
        expected_trade_cost = _expected_trade_cost(
            desired_delta,
            capital_base=request.capital_base,
            per_turnover_cost=request.per_turnover_cost,
        )
        candidates.append(
            _TradeTransitionCandidate(
                subject_id=subject_id,
                current_weight=current_weight,
                desired_weight=desired_weight,
                desired_delta=desired_delta,
                risk_scale=1.0 if target is None else float(target.risk_scale),
                expected_trade_cost=expected_trade_cost,
            )
        )

    desired_turnover = sum(
        abs(candidate.desired_delta) for candidate in candidates
    )
    executed_delta_by_subject = _executed_deltas_from_candidates(candidates)

    executed_targets: dict[str, PortfolioTarget] = {}
    transitions: list[SubjectTradeTransition] = []
    for candidate in candidates:
        executed_delta = executed_delta_by_subject.get(candidate.subject_id, 0.0)
        executed_weight = candidate.current_weight + executed_delta
        reason = _transition_reason(desired_delta=candidate.desired_delta)
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
        subjects=tuple(transitions),
    )
    return TradeTransitionResult(
        executed_targets=executed_targets,
        trace=trace,
    )


def _executed_deltas_from_candidates(
    candidates: list[_TradeTransitionCandidate],
) -> dict[str, float]:
    return {
        item.subject_id: float(item.desired_delta)
        for item in candidates
    }


def _expected_trade_cost(
    weight_delta: float,
    *,
    capital_base: float,
    per_turnover_cost: float,
) -> float:
    return float(
        abs(weight_delta)
        * max(float(capital_base), 0.0)
        * max(float(per_turnover_cost), 0.0)
    )


def _transition_reason(*, desired_delta: float) -> str:
    if abs(desired_delta) == 0.0:
        return "unchanged"
    return "executed"
