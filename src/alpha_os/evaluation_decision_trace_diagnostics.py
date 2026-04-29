from __future__ import annotations

from dataclasses import dataclass
from math import prod

from .evaluation_lane import EvaluationLane
from .portfolio_direction import PORTFOLIO_DIRECTION_MODES, PortfolioDirectionMode


_EPSILON = 1e-12


@dataclass(frozen=True)
class DirectionDiagnosticRow:
    direction: str
    subject_step_count: int
    hit_count: int
    miss_count: int
    neutral_count: int
    hit_rate: float
    signed_edge: float
    net_pnl_notional: float
    gross_pnl_notional: float
    wrong_way_pnl_notional: float


@dataclass(frozen=True)
class DirectionDiagnostics:
    subject_step_count: int
    active_subject_step_count: int
    hit_rate: float
    signed_edge: float
    rows: tuple[DirectionDiagnosticRow, ...]


@dataclass(frozen=True)
class CostTurnoverDiagnostics:
    gross_return: float
    net_return: float
    return_cost_drag: float
    gross_pnl_notional: float
    net_pnl_notional: float
    cost_notional: float
    execution_cost_notional: float
    funding_cost_notional: float
    borrow_cost_notional: float
    roll_cost_notional: float
    total_turnover: float
    average_turnover: float
    traded_notional: float
    cost_per_traded_notional: float
    cost_to_abs_gross_pnl: float


@dataclass(frozen=True)
class ExposureDiagnostics:
    average_gross_exposure: float
    average_net_exposure: float
    average_long_exposure: float
    average_short_exposure: float
    average_risk_scale: float
    max_subject_concentration_label: str | None
    max_subject_concentration: float
    max_cluster_concentration_label: str | None
    max_cluster_concentration: float


@dataclass(frozen=True)
class ContributionDiagnosticRow:
    label: str
    subject_step_count: int
    net_pnl_notional: float
    gross_pnl_notional: float
    cost_notional: float
    wrong_way_pnl_notional: float
    average_weight: float
    average_signal: float
    average_gross_exposure: float


@dataclass(frozen=True)
class ContributionDiagnostics:
    subject_rows: tuple[ContributionDiagnosticRow, ...]
    cluster_rows: tuple[ContributionDiagnosticRow, ...]
    asset_class_rows: tuple[ContributionDiagnosticRow, ...]
    direction_rows: tuple[ContributionDiagnosticRow, ...]


@dataclass(frozen=True)
class BaselineTraceDiagnostics:
    step_count: int
    subject_step_count: int
    direction: DirectionDiagnostics
    cost_turnover: CostTurnoverDiagnostics
    exposure: ExposureDiagnostics
    contribution: ContributionDiagnostics


@dataclass(frozen=True)
class TailRiskAttributionRow:
    label: str
    net_pnl_notional: float
    gross_pnl_notional: float
    cost_notional: float
    wrong_way_pnl_notional: float
    average_weight: float
    average_signal: float
    average_gross_exposure: float


@dataclass(frozen=True)
class TailRiskDirectionRow:
    direction: str
    net_pnl_notional: float
    gross_pnl_notional: float
    cost_notional: float
    wrong_way_pnl_notional: float


@dataclass(frozen=True)
class TailRiskDiagnostics:
    step_count: int
    subject_step_count: int
    gross_return: float
    net_return: float
    cost_notional: float
    funding_cost_notional: float
    borrow_cost_notional: float
    roll_cost_notional: float
    worst_day: str | None
    worst_day_net_return: float
    max_drawdown: float
    subject_losers: tuple[TailRiskAttributionRow, ...]
    cluster_losers: tuple[TailRiskAttributionRow, ...]
    asset_class_losers: tuple[TailRiskAttributionRow, ...]
    direction_rows: tuple[TailRiskDirectionRow, ...]
    exposure: ExposureDiagnostics


@dataclass(frozen=True)
class DirectionAblationContributionRow:
    label: str
    subject_step_count: int
    net_pnl_notional: float
    gross_pnl_notional: float
    cost_notional: float
    average_weight: float
    average_gross_exposure: float


@dataclass(frozen=True)
class DirectionAblationModeResult:
    mode: PortfolioDirectionMode
    step_count: int
    subject_step_count: int
    gross_return: float
    net_return: float
    return_cost_drag: float
    gross_pnl_notional: float
    net_pnl_notional: float
    cost_notional: float
    total_turnover: float
    average_turnover: float
    average_gross_exposure: float
    average_net_exposure: float
    average_long_exposure: float
    average_short_exposure: float
    subject_rows: tuple[DirectionAblationContributionRow, ...]
    asset_class_rows: tuple[DirectionAblationContributionRow, ...]
    cluster_rows: tuple[DirectionAblationContributionRow, ...]


@dataclass(frozen=True)
class DirectionAblationDiagnostics:
    modes: tuple[DirectionAblationModeResult, ...]


@dataclass(frozen=True)
class DecisionTraceDiagnosticRange:
    range_label: str
    baseline: BaselineTraceDiagnostics
    tail_risk: TailRiskDiagnostics
    direction_ablation: DirectionAblationDiagnostics


@dataclass(frozen=True)
class DecisionTraceDiagnosticsReport:
    evaluation_report_id: str
    variant: str
    evaluation_lane: EvaluationLane
    ranges: tuple[DecisionTraceDiagnosticRange, ...]


@dataclass(frozen=True)
class _SubjectStepWithDate:
    step_as_of: str
    step_index: int
    subject_id: str
    asset_class: str | None
    cluster: str | None
    signal_value: float
    realized_return: float
    target_weight: float
    target_notional: float
    traded_notional: float
    gross_pnl_notional: float
    execution_cost_notional: float
    funding_cost_notional: float
    borrow_cost_notional: float
    roll_cost_notional: float
    cost_notional: float
    net_pnl_notional: float
    risk_scale: float


def _compounded_return(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(prod(1.0 + value for value in values) - 1.0)


def _max_drawdown(equity_values: list[float]) -> float:
    running_peak: float | None = None
    worst_drawdown = 0.0
    for value in equity_values:
        if running_peak is None or value > running_peak:
            running_peak = value
        if running_peak is None or running_peak <= 0.0:
            continue
        worst_drawdown = min(worst_drawdown, (value / running_peak) - 1.0)
    return abs(float(worst_drawdown))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= _EPSILON:
        return 0.0
    return float(numerator / denominator)


def _direction_from_weight(weight: float) -> str:
    if weight > _EPSILON:
        return "long"
    if weight < -_EPSILON:
        return "short"
    return "flat"


def _direction_sign(direction: str) -> int:
    if direction == "long":
        return 1
    if direction == "short":
        return -1
    return 0


def _wrong_way_pnl(row: _SubjectStepWithDate) -> float:
    if abs(row.target_weight) <= _EPSILON:
        return 0.0
    if row.target_weight * row.realized_return >= 0.0:
        return 0.0
    return float(row.gross_pnl_notional)


def _build_subject_rows(steps, subject_steps) -> list[_SubjectStepWithDate]:
    step_by_index = {step.step_index: step for step in steps}
    return [
        _SubjectStepWithDate(
            step_as_of=step_by_index[item.step_index].step_as_of,
            step_index=item.step_index,
            subject_id=item.subject_id,
            asset_class=item.asset_class,
            cluster=item.cluster,
            signal_value=item.signal_value,
            realized_return=item.realized_return,
            target_weight=item.target_weight,
            target_notional=item.target_notional,
            traded_notional=item.traded_notional,
            gross_pnl_notional=item.gross_pnl_notional,
            execution_cost_notional=item.execution_cost_notional,
            funding_cost_notional=item.funding_cost_notional,
            borrow_cost_notional=item.borrow_cost_notional,
            roll_cost_notional=item.roll_cost_notional,
            cost_notional=item.cost_notional,
            net_pnl_notional=item.net_pnl_notional,
            risk_scale=item.risk_scale,
        )
        for item in subject_steps
        if item.step_index in step_by_index
    ]


def _direction_diagnostics(rows: list[_SubjectStepWithDate]) -> DirectionDiagnostics:
    direction_rows: list[DirectionDiagnosticRow] = []
    active_rows = [
        row for row in rows if _direction_from_weight(row.target_weight) != "flat"
    ]
    for direction in ("long", "short", "flat"):
        items = [
            row for row in rows if _direction_from_weight(row.target_weight) == direction
        ]
        sign = _direction_sign(direction)
        hit_count = sum(
            1 for row in items if sign != 0 and sign * row.realized_return > _EPSILON
        )
        miss_count = sum(
            1 for row in items if sign != 0 and sign * row.realized_return < -_EPSILON
        )
        neutral_count = len(items) - hit_count - miss_count
        direction_rows.append(
            DirectionDiagnosticRow(
                direction=direction,
                subject_step_count=len(items),
                hit_count=hit_count,
                miss_count=miss_count,
                neutral_count=neutral_count,
                hit_rate=_safe_ratio(float(hit_count), float(hit_count + miss_count)),
                signed_edge=_mean([row.signal_value * row.realized_return for row in items]),
                net_pnl_notional=sum(row.net_pnl_notional for row in items),
                gross_pnl_notional=sum(row.gross_pnl_notional for row in items),
                wrong_way_pnl_notional=sum(_wrong_way_pnl(row) for row in items),
            )
        )
    total_hit_count = sum(
        1 for row in active_rows if row.target_weight * row.realized_return > _EPSILON
    )
    total_miss_count = sum(
        1 for row in active_rows if row.target_weight * row.realized_return < -_EPSILON
    )
    return DirectionDiagnostics(
        subject_step_count=len(rows),
        active_subject_step_count=len(active_rows),
        hit_rate=_safe_ratio(
            float(total_hit_count),
            float(total_hit_count + total_miss_count),
        ),
        signed_edge=_mean([row.signal_value * row.realized_return for row in rows]),
        rows=tuple(direction_rows),
    )


def _cost_turnover_diagnostics(steps, rows: list[_SubjectStepWithDate]):
    gross_pnl = sum(step.gross_pnl_notional for step in steps)
    cost = sum(step.cost_notional for step in steps)
    traded_notional = sum(step.traded_notional for step in steps)
    return CostTurnoverDiagnostics(
        gross_return=_compounded_return([step.gross_return for step in steps]),
        net_return=_compounded_return([step.net_return for step in steps]),
        return_cost_drag=(
            _compounded_return([step.gross_return for step in steps])
            - _compounded_return([step.net_return for step in steps])
        ),
        gross_pnl_notional=gross_pnl,
        net_pnl_notional=sum(step.net_pnl_notional for step in steps),
        cost_notional=cost,
        execution_cost_notional=sum(row.execution_cost_notional for row in rows),
        funding_cost_notional=sum(step.funding_cost_notional for step in steps),
        borrow_cost_notional=sum(step.borrow_cost_notional for step in steps),
        roll_cost_notional=sum(step.roll_cost_notional for step in steps),
        total_turnover=sum(step.turnover for step in steps),
        average_turnover=_mean([step.turnover for step in steps]),
        traded_notional=traded_notional,
        cost_per_traded_notional=_safe_ratio(cost, traded_notional),
        cost_to_abs_gross_pnl=_safe_ratio(cost, abs(gross_pnl)),
    )


def _max_concentration(
    rows: list[_SubjectStepWithDate],
    *,
    label_for_row,
) -> tuple[str | None, float]:
    rows_by_step: dict[int, list[_SubjectStepWithDate]] = {}
    for row in rows:
        rows_by_step.setdefault(row.step_index, []).append(row)
    best_label: str | None = None
    best_value = 0.0
    for step_rows in rows_by_step.values():
        gross = sum(abs(row.target_notional) for row in step_rows)
        if gross <= _EPSILON:
            continue
        grouped: dict[str, float] = {}
        for row in step_rows:
            label = label_for_row(row)
            grouped[label] = grouped.get(label, 0.0) + abs(row.target_notional)
        for label, value in grouped.items():
            concentration = value / gross
            if concentration > best_value:
                best_label = label
                best_value = concentration
    return best_label, float(best_value)


def _exposure_diagnostics(steps, rows: list[_SubjectStepWithDate]) -> ExposureDiagnostics:
    subject_label, subject_concentration = _max_concentration(
        rows,
        label_for_row=lambda row: row.subject_id,
    )
    cluster_label, cluster_concentration = _max_concentration(
        rows,
        label_for_row=lambda row: row.cluster or "-",
    )
    return ExposureDiagnostics(
        average_gross_exposure=_mean([step.gross_notional_exposure for step in steps]),
        average_net_exposure=_mean([step.net_notional_exposure for step in steps]),
        average_long_exposure=_mean([step.long_notional_exposure for step in steps]),
        average_short_exposure=_mean([step.short_notional_exposure for step in steps]),
        average_risk_scale=_mean([row.risk_scale for row in rows]),
        max_subject_concentration_label=subject_label,
        max_subject_concentration=subject_concentration,
        max_cluster_concentration_label=cluster_label,
        max_cluster_concentration=cluster_concentration,
    )


def _contribution_rows(
    rows: list[_SubjectStepWithDate],
    *,
    labels_by_row: list[str],
    step_count: int,
    top_n: int,
) -> tuple[ContributionDiagnosticRow, ...]:
    grouped: dict[str, list[_SubjectStepWithDate]] = {}
    for label, row in zip(labels_by_row, rows):
        grouped.setdefault(label, []).append(row)
    denominator = max(step_count, 1)
    result = [
        ContributionDiagnosticRow(
            label=label,
            subject_step_count=len(items),
            net_pnl_notional=sum(item.net_pnl_notional for item in items),
            gross_pnl_notional=sum(item.gross_pnl_notional for item in items),
            cost_notional=sum(item.cost_notional for item in items),
            wrong_way_pnl_notional=sum(_wrong_way_pnl(item) for item in items),
            average_weight=_mean([item.target_weight for item in items]),
            average_signal=_mean([item.signal_value for item in items]),
            average_gross_exposure=sum(abs(item.target_notional) for item in items)
            / denominator,
        )
        for label, items in grouped.items()
    ]
    return tuple(sorted(result, key=lambda item: (item.net_pnl_notional, item.label))[:top_n])


def _baseline_diagnostics(steps, rows: list[_SubjectStepWithDate], *, top_n: int):
    return BaselineTraceDiagnostics(
        step_count=len(steps),
        subject_step_count=len(rows),
        direction=_direction_diagnostics(rows),
        cost_turnover=_cost_turnover_diagnostics(steps, rows),
        exposure=_exposure_diagnostics(steps, rows),
        contribution=ContributionDiagnostics(
            subject_rows=_contribution_rows(
                rows,
                labels_by_row=[row.subject_id for row in rows],
                step_count=len(steps),
                top_n=top_n,
            ),
            cluster_rows=_contribution_rows(
                rows,
                labels_by_row=[row.cluster or "-" for row in rows],
                step_count=len(steps),
                top_n=top_n,
            ),
            asset_class_rows=_contribution_rows(
                rows,
                labels_by_row=[row.asset_class or "-" for row in rows],
                step_count=len(steps),
                top_n=top_n,
            ),
            direction_rows=_contribution_rows(
                rows,
                labels_by_row=[_direction_from_weight(row.target_weight) for row in rows],
                step_count=len(steps),
                top_n=top_n,
            ),
        ),
    )


def _tail_risk_rows(
    rows: list[_SubjectStepWithDate],
    *,
    labels_by_row: list[str],
    step_count: int,
    top_n: int,
) -> tuple[TailRiskAttributionRow, ...]:
    grouped: dict[str, list[_SubjectStepWithDate]] = {}
    for label, row in zip(labels_by_row, rows):
        grouped.setdefault(label, []).append(row)
    denominator = max(step_count, 1)
    result = [
        TailRiskAttributionRow(
            label=label,
            net_pnl_notional=sum(item.net_pnl_notional for item in items),
            gross_pnl_notional=sum(item.gross_pnl_notional for item in items),
            cost_notional=sum(item.cost_notional for item in items),
            wrong_way_pnl_notional=sum(_wrong_way_pnl(item) for item in items),
            average_weight=_mean([item.target_weight for item in items]),
            average_signal=_mean([item.signal_value for item in items]),
            average_gross_exposure=sum(abs(item.target_notional) for item in items)
            / denominator,
        )
        for label, items in grouped.items()
    ]
    return tuple(sorted(result, key=lambda item: (item.net_pnl_notional, item.label))[:top_n])


def _tail_risk_direction_rows(
    rows: list[_SubjectStepWithDate],
) -> tuple[TailRiskDirectionRow, ...]:
    result: list[TailRiskDirectionRow] = []
    for direction in ("long", "short", "flat"):
        items = [item for item in rows if _direction_from_weight(item.target_weight) == direction]
        result.append(
            TailRiskDirectionRow(
                direction=direction,
                net_pnl_notional=sum(item.net_pnl_notional for item in items),
                gross_pnl_notional=sum(item.gross_pnl_notional for item in items),
                cost_notional=sum(item.cost_notional for item in items),
                wrong_way_pnl_notional=sum(_wrong_way_pnl(item) for item in items),
            )
        )
    return tuple(result)


def _tail_risk_diagnostics(steps, rows: list[_SubjectStepWithDate], *, top_n: int):
    worst_step = min(steps, key=lambda step: step.net_return, default=None)
    return TailRiskDiagnostics(
        step_count=len(steps),
        subject_step_count=len(rows),
        gross_return=_compounded_return([step.gross_return for step in steps]),
        net_return=_compounded_return([step.net_return for step in steps]),
        cost_notional=sum(step.cost_notional for step in steps),
        funding_cost_notional=sum(step.funding_cost_notional for step in steps),
        borrow_cost_notional=sum(step.borrow_cost_notional for step in steps),
        roll_cost_notional=sum(step.roll_cost_notional for step in steps),
        worst_day=None if worst_step is None else worst_step.step_as_of,
        worst_day_net_return=0.0 if worst_step is None else worst_step.net_return,
        max_drawdown=_max_drawdown([step.net_equity for step in steps]),
        subject_losers=_tail_risk_rows(
            rows,
            labels_by_row=[row.subject_id for row in rows],
            step_count=len(steps),
            top_n=top_n,
        ),
        cluster_losers=_tail_risk_rows(
            rows,
            labels_by_row=[row.cluster or "-" for row in rows],
            step_count=len(steps),
            top_n=top_n,
        ),
        asset_class_losers=_tail_risk_rows(
            rows,
            labels_by_row=[row.asset_class or "-" for row in rows],
            step_count=len(steps),
            top_n=top_n,
        ),
        direction_rows=_tail_risk_direction_rows(rows),
        exposure=_exposure_diagnostics(steps, rows),
    )


def _row_is_active_for_mode(
    row: _SubjectStepWithDate,
    mode: PortfolioDirectionMode,
) -> bool:
    if mode == "long_short":
        return abs(row.target_weight) > _EPSILON
    if mode == "long_only":
        return row.target_weight > _EPSILON
    return row.target_weight < -_EPSILON


def _rows_by_step(rows: list[_SubjectStepWithDate]) -> dict[int, list[_SubjectStepWithDate]]:
    grouped: dict[int, list[_SubjectStepWithDate]] = {}
    for row in rows:
        grouped.setdefault(row.step_index, []).append(row)
    return grouped


def _ablation_contribution_rows(
    rows: list[_SubjectStepWithDate],
    *,
    labels_by_row: list[str],
    step_count: int,
    top_n: int,
) -> tuple[DirectionAblationContributionRow, ...]:
    grouped: dict[str, list[_SubjectStepWithDate]] = {}
    for label, row in zip(labels_by_row, rows):
        grouped.setdefault(label, []).append(row)
    denominator = max(step_count, 1)
    results = [
        DirectionAblationContributionRow(
            label=label,
            subject_step_count=len(items),
            net_pnl_notional=sum(item.net_pnl_notional for item in items),
            gross_pnl_notional=sum(item.gross_pnl_notional for item in items),
            cost_notional=sum(item.cost_notional for item in items),
            average_weight=_mean([item.target_weight for item in items]),
            average_gross_exposure=sum(abs(item.target_notional) for item in items)
            / denominator,
        )
        for label, items in grouped.items()
    ]
    return tuple(
        sorted(results, key=lambda item: (-abs(item.net_pnl_notional), item.label))[:top_n]
    )


def _ablation_mode_result(
    *,
    mode: PortfolioDirectionMode,
    steps,
    rows: list[_SubjectStepWithDate],
    top_n: int,
) -> DirectionAblationModeResult:
    selected_rows = [row for row in rows if _row_is_active_for_mode(row, mode)]
    selected_by_step = _rows_by_step(selected_rows)
    gross_step_returns: list[float] = []
    net_step_returns: list[float] = []
    turnover_by_step: list[float] = []
    gross_exposure_by_step: list[float] = []
    net_exposure_by_step: list[float] = []
    long_exposure_by_step: list[float] = []
    short_exposure_by_step: list[float] = []
    for step in steps:
        step_rows = selected_by_step.get(step.step_index, [])
        gross_step_returns.append(sum(row.gross_pnl_notional for row in step_rows))
        net_step_returns.append(sum(row.net_pnl_notional for row in step_rows))
        turnover_by_step.append(sum(row.traded_notional for row in step_rows))
        weights = [row.target_weight for row in step_rows]
        gross_exposure_by_step.append(sum(abs(weight) for weight in weights))
        net_exposure_by_step.append(sum(weights))
        long_exposure_by_step.append(sum(max(weight, 0.0) for weight in weights))
        short_exposure_by_step.append(abs(sum(min(weight, 0.0) for weight in weights)))
    gross_return = _compounded_return(gross_step_returns)
    net_return = _compounded_return(net_step_returns)
    return DirectionAblationModeResult(
        mode=mode,
        step_count=len(steps),
        subject_step_count=len(selected_rows),
        gross_return=gross_return,
        net_return=net_return,
        return_cost_drag=gross_return - net_return,
        gross_pnl_notional=sum(row.gross_pnl_notional for row in selected_rows),
        net_pnl_notional=sum(row.net_pnl_notional for row in selected_rows),
        cost_notional=sum(row.cost_notional for row in selected_rows),
        total_turnover=sum(turnover_by_step),
        average_turnover=_mean(turnover_by_step),
        average_gross_exposure=_mean(gross_exposure_by_step),
        average_net_exposure=_mean(net_exposure_by_step),
        average_long_exposure=_mean(long_exposure_by_step),
        average_short_exposure=_mean(short_exposure_by_step),
        subject_rows=_ablation_contribution_rows(
            selected_rows,
            labels_by_row=[row.subject_id for row in selected_rows],
            step_count=len(steps),
            top_n=top_n,
        ),
        asset_class_rows=_ablation_contribution_rows(
            selected_rows,
            labels_by_row=[row.asset_class or "-" for row in selected_rows],
            step_count=len(steps),
            top_n=top_n,
        ),
        cluster_rows=_ablation_contribution_rows(
            selected_rows,
            labels_by_row=[row.cluster or "-" for row in selected_rows],
            step_count=len(steps),
            top_n=top_n,
        ),
    )


def _direction_ablation_diagnostics(
    steps,
    rows: list[_SubjectStepWithDate],
    *,
    top_n: int,
) -> DirectionAblationDiagnostics:
    return DirectionAblationDiagnostics(
        modes=tuple(
            _ablation_mode_result(mode=mode, steps=steps, rows=rows, top_n=top_n)
            for mode in PORTFOLIO_DIRECTION_MODES
        )
    )


def _range_labels_from_report(report) -> tuple[str, ...]:
    labels: list[str] = []
    for task_result in report.task_results:
        for label in task_result.artifact_refs.get("evaluation_range_labels", ()):
            if label not in labels:
                labels.append(label)
    return tuple(labels)


def _range_labels_from_trace_steps(store, *, evaluation_report_id: str, variant: str):
    steps = store.list_evaluation_decision_trace_steps(
        evaluation_report_id=evaluation_report_id,
        variant=variant,
        limit=1_000_000,
    )
    labels: list[str] = []
    for step in steps:
        if step.evaluation_range_label not in labels:
            labels.append(step.evaluation_range_label)
    return tuple(labels)


def build_evaluation_decision_trace_diagnostics(
    store,
    *,
    evaluation_report_id: str,
    range_labels: tuple[str, ...] | None = None,
    variant: str = "selected",
    top_n: int = 8,
    row_limit: int = 1_000_000,
) -> DecisionTraceDiagnosticsReport:
    report_state = store.get_evaluation_report(evaluation_report_id)
    if report_state is None:
        raise ValueError(f"evaluation report does not exist: {evaluation_report_id}")
    resolved_range_labels = range_labels
    if resolved_range_labels is None:
        resolved_range_labels = _range_labels_from_report(report_state.report)
    if not resolved_range_labels:
        resolved_range_labels = _range_labels_from_trace_steps(
            store,
            evaluation_report_id=evaluation_report_id,
            variant=variant,
        )
    ranges: list[DecisionTraceDiagnosticRange] = []
    for range_label in resolved_range_labels:
        steps = store.list_evaluation_decision_trace_steps(
            evaluation_report_id=evaluation_report_id,
            evaluation_range_label=range_label,
            variant=variant,
            limit=row_limit,
        )
        subject_steps = store.list_evaluation_decision_trace_subject_steps(
            evaluation_report_id=evaluation_report_id,
            evaluation_range_label=range_label,
            variant=variant,
            limit=row_limit,
        )
        if not steps or not subject_steps:
            raise ValueError(
                "evaluation decision trace is missing; rerun evaluation with "
                f"trace persistence before diagnostics: report={evaluation_report_id} "
                f"range={range_label} variant={variant}"
            )
        rows = _build_subject_rows(steps, subject_steps)
        ranges.append(
            DecisionTraceDiagnosticRange(
                range_label=range_label,
                baseline=_baseline_diagnostics(
                    steps,
                    rows,
                    top_n=max(int(top_n), 1),
                ),
                tail_risk=_tail_risk_diagnostics(
                    steps,
                    rows,
                    top_n=max(int(top_n), 1),
                ),
                direction_ablation=_direction_ablation_diagnostics(
                    steps,
                    rows,
                    top_n=max(int(top_n), 1),
                ),
            )
        )
    return DecisionTraceDiagnosticsReport(
        evaluation_report_id=evaluation_report_id,
        variant=variant,
        evaluation_lane="diagnostic",
        ranges=tuple(ranges),
    )
