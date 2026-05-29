from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .contract_boundaries import (
    PortfolioConstraintBoundary,
    default_portfolio_constraint_boundary,
)
from .evaluation_cost_config import TradingEnvironment
from .portfolio_decision import (
    CostInput,
    DependenceInput,
    HistoricalReturnInput,
    ObservedPortfolioInputs,
    PortfolioDecisionInput,
    PortfolioDecisionAssumptions,
    PortfolioPositionState,
    PortfolioState,
    PortfolioTarget,
    PredictiveSignalInput,
    RiskInput,
    SizingDiagnostics,
    UncertaintyInput,
)
from .portfolio_construction_config import PortfolioConstructionSpec
from .portfolio_construction_pipeline import (
    PortfolioConstructionStageTrace,
    build_portfolio_construction_request,
    construct_portfolio_targets,
)
from .portfolio_sizing_policy import (
    PortfolioSizingPolicy,
    apply_portfolio_sizing_policy,
)


@dataclass(frozen=True)
class SubjectBacktestSeries:
    subject_id: str
    signal_series: pd.Series
    realized_return_series: pd.Series
    target_id: str = "residual_return_3d"
    historical_return_series: pd.Series | None = None
    confidence_series: pd.Series | None = None
    risk_series: pd.Series | None = None
    uncertainty_series: pd.Series | None = None
    funding_cost_bps_series: pd.Series | None = None
    borrow_fee_bps_series: pd.Series | None = None
    roll_cost_bps_series: pd.Series | None = None
    contract_multiplier: float | None = None


@dataclass(frozen=True)
class DependenceBacktestSeries:
    left_subject_id: str
    right_subject_id: str
    series: pd.Series


@dataclass(frozen=True)
class DecisionBacktestInput:
    subject_series: tuple[SubjectBacktestSeries, ...]
    initial_capital_base: float = 1.0
    initial_positions: tuple[PortfolioPositionState, ...] = ()
    initial_holding_period_days: int = 0
    dependence_series: tuple[DependenceBacktestSeries, ...] = ()
    portfolio_construction: PortfolioConstructionSpec = field(
        default_factory=PortfolioConstructionSpec
    )
    asset_class_by_subject: dict[str, str] | None = None
    cluster_by_subject: dict[str, str] | None = None
    trading_environment: TradingEnvironment = field(default_factory=TradingEnvironment)
    historical_return_lookback_steps: int | None = None

    @property
    def subject_ids(self) -> tuple[str, ...]:
        return tuple(item.subject_id for item in self.subject_series)


@dataclass(frozen=True)
class DecisionBacktestSubjectStep:
    subject_id: str
    signal_value: float
    realized_return: float
    target_weight: float
    position_delta: float
    target_notional: float
    traded_notional: float
    risk_scale: float
    entry_allowed: bool
    gross_pnl_notional: float = 0.0
    execution_cost_notional: float = 0.0
    funding_cost_notional: float = 0.0
    borrow_cost_notional: float = 0.0
    roll_cost_notional: float = 0.0
    cost_notional: float = 0.0
    net_pnl_notional: float = 0.0
    net_return_contribution: float = 0.0
    funding_cost_bps: float = 0.0
    borrow_fee_bps: float = 0.0
    roll_cost_bps: float = 0.0
    contract_multiplier: float | None = None
    target_contracts: float | None = None
    traded_contracts: float | None = None


@dataclass(frozen=True)
class SubjectRebalanceTrace:
    subject_id: str
    current_weight: float
    desired_weight: float
    executed_weight: float
    desired_delta: float
    executed_delta: float
    expected_trade_cost: float = 0.0


@dataclass(frozen=True)
class RebalanceTrace:
    desired_turnover: float
    executed_turnover: float
    expected_execution_cost: float
    subjects: tuple[SubjectRebalanceTrace, ...]


@dataclass(frozen=True)
class DecisionBacktestStep:
    date: str
    subject_steps: tuple[DecisionBacktestSubjectStep, ...]
    gross_return: float
    gross_pnl_notional: float
    turnover: float
    traded_notional: float
    cost: float
    cost_notional: float
    net_return: float
    net_pnl_notional: float
    gross_leverage_exposure: float
    net_leverage_exposure: float
    long_leverage_exposure: float
    short_leverage_exposure: float
    gross_notional_exposure: float
    net_notional_exposure: float
    long_notional_exposure: float
    short_notional_exposure: float
    funding_cost_notional: float
    borrow_cost_notional: float
    roll_cost_notional: float
    gross_equity: float
    net_equity: float
    construction_trace: tuple[PortfolioConstructionStageTrace, ...] = ()
    execution_trace: RebalanceTrace | None = None
    sizing_diagnostics: SizingDiagnostics = field(default_factory=SizingDiagnostics)

    @property
    def subject_step_by_subject(self) -> dict[str, DecisionBacktestSubjectStep]:
        return {item.subject_id: item for item in self.subject_steps}


@dataclass(frozen=True)
class DecisionBacktestResult:
    subject_ids: tuple[str, ...]
    steps: tuple[DecisionBacktestStep, ...]
    initial_capital_base: float = 1.0

    @property
    def gross_return_total(self) -> float:
        if not self.steps:
            return 0.0
        if self.initial_capital_base <= 0.0:
            return 0.0
        return float((self.steps[-1].gross_equity / self.initial_capital_base) - 1.0)

    @property
    def net_return_total(self) -> float:
        if not self.steps:
            return 0.0
        if self.initial_capital_base <= 0.0:
            return 0.0
        return float((self.steps[-1].net_equity / self.initial_capital_base) - 1.0)

    @property
    def mean_turnover(self) -> float:
        if not self.steps:
            return 0.0
        return float(sum(step.turnover for step in self.steps) / len(self.steps))

    @property
    def mean_gross_notional_exposure(self) -> float:
        if not self.steps:
            return 0.0
        return float(
            sum(step.gross_notional_exposure for step in self.steps) / len(self.steps)
        )

    @property
    def mean_gross_leverage_exposure(self) -> float:
        if not self.steps:
            return 0.0
        return float(
            sum(step.gross_leverage_exposure for step in self.steps) / len(self.steps)
        )

    @property
    def mean_net_leverage_exposure(self) -> float:
        if not self.steps:
            return 0.0
        return float(
            sum(step.net_leverage_exposure for step in self.steps) / len(self.steps)
        )

    @property
    def mean_long_leverage_exposure(self) -> float:
        if not self.steps:
            return 0.0
        return float(
            sum(step.long_leverage_exposure for step in self.steps) / len(self.steps)
        )

    @property
    def mean_short_leverage_exposure(self) -> float:
        if not self.steps:
            return 0.0
        return float(
            sum(step.short_leverage_exposure for step in self.steps) / len(self.steps)
        )

    @property
    def mean_net_notional_exposure(self) -> float:
        if not self.steps:
            return 0.0
        return float(
            sum(step.net_notional_exposure for step in self.steps) / len(self.steps)
        )

    @property
    def mean_long_notional_exposure(self) -> float:
        if not self.steps:
            return 0.0
        return float(
            sum(step.long_notional_exposure for step in self.steps) / len(self.steps)
        )

    @property
    def mean_short_notional_exposure(self) -> float:
        if not self.steps:
            return 0.0
        return float(
            sum(step.short_notional_exposure for step in self.steps) / len(self.steps)
        )

    @property
    def mean_traded_notional(self) -> float:
        if not self.steps:
            return 0.0
        return float(sum(step.traded_notional for step in self.steps) / len(self.steps))

    @property
    def cost_notional_total(self) -> float:
        if not self.steps:
            return 0.0
        return float(sum(step.cost_notional for step in self.steps))

    @property
    def funding_cost_notional_total(self) -> float:
        if not self.steps:
            return 0.0
        return float(sum(step.funding_cost_notional for step in self.steps))

    @property
    def borrow_cost_notional_total(self) -> float:
        if not self.steps:
            return 0.0
        return float(sum(step.borrow_cost_notional for step in self.steps))

    @property
    def roll_cost_notional_total(self) -> float:
        if not self.steps:
            return 0.0
        return float(sum(step.roll_cost_notional for step in self.steps))

    @property
    def max_drawdown(self) -> float:
        peak = max(float(self.initial_capital_base), 0.0)
        drawdown = 0.0
        for step in self.steps:
            peak = max(peak, step.net_equity)
            if peak > 0.0:
                drawdown = max(drawdown, 1.0 - (step.net_equity / peak))
        return float(drawdown)


@dataclass(frozen=True)
class PortfolioBacktestState:
    current_weights: dict[str, float]
    gross_equity: float
    net_equity: float
    net_peak_equity: float
    current_drawdown: float
    holding_period_days: int
    recent_turnover: float
    rebalance_step: int

    @property
    def capital_base(self) -> float:
        return max(float(self.net_equity), 0.0)


@dataclass(frozen=True)
class BacktestStepAccounting:
    gross_pnl_notional: float
    gross_return: float
    gross_leverage_exposure: float
    net_leverage_exposure: float
    long_leverage_exposure: float
    short_leverage_exposure: float
    gross_notional_exposure: float
    net_notional_exposure: float
    long_notional_exposure: float
    short_notional_exposure: float
    turnover: float
    traded_notional: float
    cost: float
    cost_notional: float
    net_pnl_notional: float
    net_return: float
    funding_cost_notional: float
    borrow_cost_notional: float
    roll_cost_notional: float


def run_decision_backtest(
    backtest_input: DecisionBacktestInput,
    *,
    sizing_policy: PortfolioSizingPolicy | None = None,
) -> DecisionBacktestResult:
    aligned = _aligned_frame(backtest_input)
    subject_ids = _subject_ids(backtest_input)
    state = _initial_backtest_state(backtest_input)
    steps: list[DecisionBacktestStep] = []

    for date, row in aligned.iterrows():
        if _is_rebalance_step(state, backtest_input):
            (
                desired_targets_by_subject,
                construction_trace,
                sizing_diagnostics,
            ) = _build_rebalance_targets(
                backtest_input,
                state=state,
                row=row,
                date=str(date),
                subject_ids=subject_ids,
                sizing_policy=sizing_policy,
            )
            targets_by_subject, execution_trace = _execute_rebalance_targets(
                desired_targets=desired_targets_by_subject,
                current_weights=state.current_weights,
                capital_base=state.capital_base,
                per_turnover_cost=_per_turnover_execution_cost(backtest_input),
            )
        else:
            targets_by_subject = build_hold_targets(
                subject_ids=subject_ids,
                current_weights=state.current_weights,
                capital_base=state.capital_base,
            )
            construction_trace = ()
            execution_trace = None
            sizing_diagnostics = SizingDiagnostics()
        subject_steps = build_decision_backtest_subject_steps(
            backtest_input,
            row=row,
            subject_ids=subject_ids,
            targets_by_subject=targets_by_subject,
            current_weights=state.current_weights,
            capital_base=state.capital_base,
        )
        accounting = build_backtest_step_accounting(
            subject_steps=subject_steps,
            capital_base=state.capital_base,
            backtest_input=backtest_input,
        )
        state = advance_portfolio_state(
            state,
            subject_steps=subject_steps,
            accounting=accounting,
        )
        steps.append(
            DecisionBacktestStep(
                date=str(date),
                subject_steps=tuple(subject_steps),
                gross_return=accounting.gross_return,
                gross_pnl_notional=accounting.gross_pnl_notional,
                turnover=accounting.turnover,
                traded_notional=accounting.traded_notional,
                cost=accounting.cost,
                cost_notional=accounting.cost_notional,
                net_return=accounting.net_return,
                net_pnl_notional=accounting.net_pnl_notional,
                gross_leverage_exposure=accounting.gross_leverage_exposure,
                net_leverage_exposure=accounting.net_leverage_exposure,
                long_leverage_exposure=accounting.long_leverage_exposure,
                short_leverage_exposure=accounting.short_leverage_exposure,
                gross_notional_exposure=accounting.gross_notional_exposure,
                net_notional_exposure=accounting.net_notional_exposure,
                long_notional_exposure=accounting.long_notional_exposure,
                short_notional_exposure=accounting.short_notional_exposure,
                funding_cost_notional=accounting.funding_cost_notional,
                borrow_cost_notional=accounting.borrow_cost_notional,
                roll_cost_notional=accounting.roll_cost_notional,
                gross_equity=state.gross_equity,
                net_equity=state.net_equity,
                construction_trace=construction_trace,
                execution_trace=execution_trace,
                sizing_diagnostics=sizing_diagnostics,
            )
        )

    return DecisionBacktestResult(
        subject_ids=subject_ids,
        steps=tuple(steps),
        initial_capital_base=max(float(backtest_input.initial_capital_base), 0.0),
    )


def _initial_backtest_state(backtest_input: DecisionBacktestInput) -> PortfolioBacktestState:
    initial_equity = max(float(backtest_input.initial_capital_base), 0.0)
    current_weights = {
        position.subject_id: float(position.weight)
        for position in backtest_input.initial_positions
    }
    return PortfolioBacktestState(
        current_weights=current_weights,
        gross_equity=initial_equity,
        net_equity=initial_equity,
        net_peak_equity=initial_equity,
        current_drawdown=0.0,
        holding_period_days=max(backtest_input.initial_holding_period_days, 0),
        recent_turnover=0.0,
        rebalance_step=0,
    )


def _is_rebalance_step(
    state: PortfolioBacktestState,
    backtest_input: DecisionBacktestInput,
) -> bool:
    return state.rebalance_step % max(backtest_input.portfolio_construction.rebalance_interval_steps, 1) == 0


def _build_rebalance_targets(
    backtest_input: DecisionBacktestInput,
    *,
    state: PortfolioBacktestState,
    row: pd.Series,
    date: str,
    subject_ids: tuple[str, ...],
    sizing_policy: PortfolioSizingPolicy | None,
) -> tuple[
    dict[str, PortfolioTarget],
    tuple[PortfolioConstructionStageTrace, ...],
    SizingDiagnostics,
]:
    decision_input = _portfolio_decision_input_for_backtest_row(
        backtest_input,
        state=state,
        row=row,
        date=date,
        subject_ids=subject_ids,
    )
    decision_output = apply_portfolio_sizing_policy(
        decision_input,
        sizing_policy=sizing_policy,
    )
    construction_result = construct_portfolio_targets(
        build_portfolio_construction_request(
            targets=decision_output.targets,
            current_weights=state.current_weights,
            capital_base=state.capital_base,
            gross_exposure_cap=backtest_input.portfolio_construction.gross_exposure_cap,
            gross_leverage_cap=backtest_input.portfolio_construction.gross_leverage_cap,
            net_exposure_target=backtest_input.portfolio_construction.net_exposure_target,
            target_vol=backtest_input.portfolio_construction.target_vol,
            risk_by_subject={
                subject_id: max(
                    float(_optional_value(row, ("risk", subject_id)) or 0.0),
                    0.0,
                )
                for subject_id in subject_ids
            },
            constraint_boundary=default_portfolio_constraint_boundary(),
            direction_mode=backtest_input.portfolio_construction.direction_mode,
            top_k=backtest_input.portfolio_construction.top_k,
            active_weight_budget=backtest_input.portfolio_construction.active_weight_budget,
            asset_class_by_subject=(
                {}
                if backtest_input.asset_class_by_subject is None
                else backtest_input.asset_class_by_subject
            ),
            cluster_by_subject=(
                {}
                if backtest_input.cluster_by_subject is None
                else backtest_input.cluster_by_subject
            ),
            asset_class_weight_caps=(
                {}
                if backtest_input.portfolio_construction.asset_class_weight_caps is None
                else backtest_input.portfolio_construction.asset_class_weight_caps
            ),
            cluster_weight_caps=(
                {}
                if backtest_input.portfolio_construction.cluster_weight_caps is None
                else backtest_input.portfolio_construction.cluster_weight_caps
            ),
        )
    )
    return (
        construction_result.targets,
        construction_result.trace,
        decision_output.sizing_diagnostics,
    )


def _per_turnover_execution_cost(backtest_input: DecisionBacktestInput) -> float:
    return (
        max(backtest_input.trading_environment.turnover_cost_rate, 0.0)
        + max(backtest_input.trading_environment.market_impact_bps, 0.0) / 10000.0
        + max(backtest_input.trading_environment.fee_bps, 0.0) / 10000.0
        + max(backtest_input.trading_environment.bid_ask_spread_bps, 0.0) / 10000.0
    )


def _execute_rebalance_targets(
    *,
    desired_targets: dict[str, PortfolioTarget],
    current_weights: dict[str, float],
    capital_base: float,
    per_turnover_cost: float,
) -> tuple[dict[str, PortfolioTarget], RebalanceTrace]:
    all_subjects = sorted(set(desired_targets) | set(current_weights))
    executed_targets: dict[str, PortfolioTarget] = {}
    subject_traces: list[SubjectRebalanceTrace] = []

    for subject_id in all_subjects:
        target = desired_targets.get(subject_id)
        current_weight = float(current_weights.get(subject_id, 0.0))
        desired_weight = current_weight if target is None else float(target.target_weight)
        delta = desired_weight - current_weight
        executed_weight = current_weight + delta
        expected_trade_cost = (
            abs(delta)
            * max(float(capital_base), 0.0)
            * max(float(per_turnover_cost), 0.0)
        )
        subject_traces.append(
            SubjectRebalanceTrace(
                subject_id=subject_id,
                current_weight=current_weight,
                desired_weight=desired_weight,
                executed_weight=float(executed_weight),
                desired_delta=float(delta),
                executed_delta=float(delta),
                expected_trade_cost=float(expected_trade_cost),
            )
        )
        executed_targets[subject_id] = PortfolioTarget(
            subject_id=subject_id,
            target_weight=float(executed_weight),
            position_delta=float(delta),
            target_notional=float(executed_weight * capital_base),
            entry_allowed=abs(current_weight) > 0.0 or abs(executed_weight) > 0.0,
            risk_scale=1.0 if target is None else float(target.risk_scale),
        )

    executed_turnover = sum(abs(item.executed_delta) for item in subject_traces)
    return executed_targets, RebalanceTrace(
        desired_turnover=sum(abs(item.desired_delta) for item in subject_traces),
        executed_turnover=float(executed_turnover),
        expected_execution_cost=float(
            executed_turnover
            * max(float(capital_base), 0.0)
            * max(float(per_turnover_cost), 0.0)
        ),
        subjects=tuple(subject_traces),
    )


def _portfolio_decision_input_for_backtest_row(
    backtest_input: DecisionBacktestInput,
    *,
    state: PortfolioBacktestState,
    row: pd.Series,
    date: str,
    subject_ids: tuple[str, ...],
) -> PortfolioDecisionInput:
    return PortfolioDecisionInput(
        portfolio_id=None,
        as_of=date,
        portfolio_state=PortfolioState(
            portfolio_id=None,
            as_of=date,
            positions=tuple(
                PortfolioPositionState(
                    subject_id=subject_id,
                    weight=state.current_weights.get(subject_id, 0.0),
                )
                for subject_id in subject_ids
            ),
            capital_base=state.net_equity,
            gross_limit=backtest_input.portfolio_construction.gross_exposure_cap,
            rebalance_step=state.rebalance_step,
            holding_period_days=state.holding_period_days,
            recent_turnover=state.recent_turnover,
            current_drawdown=state.current_drawdown,
        ),
        observed_inputs=ObservedPortfolioInputs(
            predictive_signals=_predictive_signals_for_row(
                backtest_input,
                row=row,
                subject_ids=subject_ids,
            ),
            risk_inputs=_risk_inputs_for_row(
                backtest_input,
                row=row,
                subject_ids=subject_ids,
            ),
            uncertainty_inputs=_uncertainty_inputs_for_row(
                backtest_input,
                row=row,
                subject_ids=subject_ids,
            ),
            dependence_inputs=_dependence_inputs_for_row(
                backtest_input,
                row=row,
            ),
            historical_return_inputs=_historical_return_inputs_for_row(
                backtest_input,
                date=date,
            ),
        ),
        assumptions=PortfolioDecisionAssumptions(
            cost_inputs=_cost_inputs_for_backtest(
                backtest_input,
                subject_ids=subject_ids,
            ),
        ),
        subject_metadata_by_subject=_subject_metadata_by_subject(backtest_input),
    )


def build_hold_targets(
    *,
    subject_ids: tuple[str, ...],
    current_weights: dict[str, float],
    capital_base: float,
) -> dict[str, PortfolioTarget]:
    return {
        subject_id: hold_position_target(
            subject_id=subject_id,
            current_weight=current_weights.get(subject_id, 0.0),
            capital_base=capital_base,
        )
        for subject_id in subject_ids
    }


def _subject_metadata_by_subject(
    backtest_input: DecisionBacktestInput,
) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    asset_class_by_subject = backtest_input.asset_class_by_subject or {}
    cluster_by_subject = backtest_input.cluster_by_subject or {}
    for subject_id in sorted(set(asset_class_by_subject) | set(cluster_by_subject)):
        values = {
            "asset_class": asset_class_by_subject.get(subject_id),
            "cluster": cluster_by_subject.get(subject_id),
        }
        metadata[subject_id] = {
            key: value
            for key, value in values.items()
            if value is not None
        }
    return metadata


def build_decision_backtest_subject_steps(
    backtest_input: DecisionBacktestInput,
    *,
    row: pd.Series,
    subject_ids: tuple[str, ...],
    targets_by_subject: dict[str, PortfolioTarget],
    current_weights: dict[str, float],
    capital_base: float,
) -> list[DecisionBacktestSubjectStep]:
    subject_steps: list[DecisionBacktestSubjectStep] = []
    for subject_id in subject_ids:
        target = targets_by_subject.get(subject_id)
        if target is None:
            current_weight = current_weights.get(subject_id, 0.0)
            target_weight = current_weight
            delta = 0.0
            target_notional = current_weight * capital_base
            risk_scale = 1.0
            entry_allowed = abs(current_weight) > 0.0
        else:
            target_weight = float(target.target_weight)
            delta = float(target.position_delta)
            target_notional = float(
                target.target_notional
                if target.target_notional is not None
                else target_weight * capital_base
            )
            risk_scale = float(target.risk_scale)
            entry_allowed = bool(target.entry_allowed)
        realized_return = _required_value(
            row,
            ("realized_return", subject_id),
        )
        funding_cost_bps = _optional_value(
            row,
            ("funding_cost_bps", subject_id),
            default=0.0,
        )
        borrow_fee_bps = _optional_value(
            row,
            ("borrow_fee_bps", subject_id),
            default=0.0,
        )
        roll_cost_bps = _optional_value(
            row,
            ("roll_cost_bps", subject_id),
            default=0.0,
        )
        contract_multiplier = _contract_multiplier_for_subject(
            backtest_input,
            subject_id=subject_id,
        )
        traded_notional = abs(delta) * capital_base
        gross_pnl_notional = target_notional * realized_return
        execution_cost_notional = (
            max(backtest_input.trading_environment.turnover_cost_rate, 0.0) * traded_notional
            + max(backtest_input.trading_environment.market_impact_bps, 0.0) / 10000.0 * traded_notional
            + max(backtest_input.trading_environment.fee_bps, 0.0) / 10000.0 * traded_notional
            + max(backtest_input.trading_environment.bid_ask_spread_bps, 0.0) / 10000.0 * traded_notional
        )
        funding_cost_notional = (
            max(backtest_input.trading_environment.funding_bps_per_step, 0.0)
            / 10000.0
            * abs(target_notional)
            + float(funding_cost_bps) / 10000.0 * target_notional
        )
        short_notional = abs(min(target_notional, 0.0))
        borrow_cost_notional = (
            max(backtest_input.trading_environment.borrow_fee_bps_per_step, 0.0)
            / 10000.0
            * short_notional
            + max(float(borrow_fee_bps), 0.0) / 10000.0 * short_notional
        )
        roll_cost_notional = (
            max(float(roll_cost_bps), 0.0) / 10000.0 * abs(target_notional)
        )
        cost_notional = (
            execution_cost_notional
            + funding_cost_notional
            + borrow_cost_notional
            + roll_cost_notional
        )
        net_pnl_notional = gross_pnl_notional - cost_notional
        subject_steps.append(
            DecisionBacktestSubjectStep(
                subject_id=subject_id,
                signal_value=_optional_value(
                    row,
                    ("signal", subject_id),
                    default=0.0,
                ),
                realized_return=realized_return,
                target_weight=target_weight,
                position_delta=delta,
                target_notional=target_notional,
                traded_notional=traded_notional,
                risk_scale=risk_scale,
                entry_allowed=entry_allowed,
                gross_pnl_notional=gross_pnl_notional,
                execution_cost_notional=execution_cost_notional,
                funding_cost_notional=funding_cost_notional,
                borrow_cost_notional=borrow_cost_notional,
                roll_cost_notional=roll_cost_notional,
                cost_notional=cost_notional,
                net_pnl_notional=net_pnl_notional,
                net_return_contribution=(
                    float(net_pnl_notional / capital_base)
                    if capital_base > 0.0
                    else 0.0
                ),
                funding_cost_bps=float(funding_cost_bps),
                borrow_fee_bps=float(borrow_fee_bps),
                roll_cost_bps=float(roll_cost_bps),
                contract_multiplier=contract_multiplier,
                target_contracts=(
                    None
                    if contract_multiplier in {None, 0.0}
                    else float(target_notional / contract_multiplier)
                ),
                traded_contracts=(
                    None
                    if contract_multiplier in {None, 0.0}
                    else float(traded_notional / contract_multiplier)
                ),
            )
        )
    return subject_steps


def build_backtest_step_accounting(
    *,
    subject_steps: list[DecisionBacktestSubjectStep] | tuple[DecisionBacktestSubjectStep, ...],
    capital_base: float,
    backtest_input: DecisionBacktestInput,
) -> BacktestStepAccounting:
    turnover = sum(abs(float(step.position_delta)) for step in subject_steps)
    traded_notional = turnover * capital_base
    gross_pnl_notional = sum(
        float(step.target_notional) * float(step.realized_return)
        for step in subject_steps
    )
    gross_leverage_exposure = sum(
        abs(float(step.target_weight)) for step in subject_steps
    )
    net_leverage_exposure = sum(float(step.target_weight) for step in subject_steps)
    long_leverage_exposure = sum(
        max(float(step.target_weight), 0.0) for step in subject_steps
    )
    short_leverage_exposure = sum(
        abs(min(float(step.target_weight), 0.0)) for step in subject_steps
    )
    gross_notional_exposure = sum(
        abs(float(step.target_notional)) for step in subject_steps
    )
    net_notional_exposure = sum(float(step.target_notional) for step in subject_steps)
    long_notional_exposure = sum(
        max(float(step.target_notional), 0.0) for step in subject_steps
    )
    short_notional_exposure = sum(
        abs(min(float(step.target_notional), 0.0)) for step in subject_steps
    )
    subject_funding_cost_notional = sum(
        (float(step.funding_cost_bps) / 10000.0) * float(step.target_notional)
        for step in subject_steps
    )
    subject_borrow_cost_notional = sum(
        max(float(step.borrow_fee_bps), 0.0)
        / 10000.0
        * abs(min(float(step.target_notional), 0.0))
        for step in subject_steps
    )
    roll_cost_notional = sum(
        max(float(step.roll_cost_bps), 0.0)
        / 10000.0
        * abs(float(step.target_notional))
        for step in subject_steps
    )
    funding_cost_notional = (
        max(backtest_input.trading_environment.funding_bps_per_step, 0.0)
        / 10000.0
        * gross_notional_exposure
        + subject_funding_cost_notional
    )
    borrow_cost_notional = (
        max(backtest_input.trading_environment.borrow_fee_bps_per_step, 0.0)
        / 10000.0
        * short_notional_exposure
        + subject_borrow_cost_notional
    )
    cost_notional = (
        max(backtest_input.trading_environment.turnover_cost_rate, 0.0) * traded_notional
        + max(backtest_input.trading_environment.market_impact_bps, 0.0) / 10000.0 * traded_notional
        + max(backtest_input.trading_environment.fee_bps, 0.0) / 10000.0 * traded_notional
        + max(backtest_input.trading_environment.bid_ask_spread_bps, 0.0) / 10000.0 * traded_notional
        + max(backtest_input.trading_environment.funding_bps_per_step, 0.0)
        / 10000.0
        * gross_notional_exposure
        + max(backtest_input.trading_environment.borrow_fee_bps_per_step, 0.0)
        / 10000.0
        * short_notional_exposure
        + subject_funding_cost_notional
        + subject_borrow_cost_notional
        + roll_cost_notional
    )
    gross_return = (
        float(gross_pnl_notional / capital_base) if capital_base > 0.0 else 0.0
    )
    cost = float(cost_notional / capital_base) if capital_base > 0.0 else 0.0
    net_pnl_notional = float(gross_pnl_notional - cost_notional)
    net_return = float(net_pnl_notional / capital_base) if capital_base > 0.0 else 0.0
    return BacktestStepAccounting(
        gross_pnl_notional=float(gross_pnl_notional),
        gross_return=float(gross_return),
        gross_leverage_exposure=float(gross_leverage_exposure),
        net_leverage_exposure=float(net_leverage_exposure),
        long_leverage_exposure=float(long_leverage_exposure),
        short_leverage_exposure=float(short_leverage_exposure),
        gross_notional_exposure=float(gross_notional_exposure),
        net_notional_exposure=float(net_notional_exposure),
        long_notional_exposure=float(long_notional_exposure),
        short_notional_exposure=float(short_notional_exposure),
        turnover=float(turnover),
        traded_notional=float(traded_notional),
        cost=float(cost),
        cost_notional=float(cost_notional),
        net_pnl_notional=float(net_pnl_notional),
        net_return=float(net_return),
        funding_cost_notional=float(funding_cost_notional),
        borrow_cost_notional=float(borrow_cost_notional),
        roll_cost_notional=float(roll_cost_notional),
    )


def advance_portfolio_state(
    state: PortfolioBacktestState,
    *,
    subject_steps: list[DecisionBacktestSubjectStep] | tuple[DecisionBacktestSubjectStep, ...],
    accounting: BacktestStepAccounting,
) -> PortfolioBacktestState:
    gross_equity = state.gross_equity + accounting.gross_pnl_notional
    net_equity = state.net_equity + accounting.net_pnl_notional
    next_position_notional_by_subject = {
        step.subject_id: float(step.target_notional) * (1.0 + float(step.realized_return))
        for step in subject_steps
    }
    current_weights = {
        subject_id: (
            float(position_notional) / net_equity
            if net_equity > 0.0
            else 0.0
        )
        for subject_id, position_notional in next_position_notional_by_subject.items()
    }
    net_peak_equity = max(state.net_peak_equity, net_equity)
    current_drawdown = (
        float(1.0 - (net_equity / net_peak_equity))
        if net_peak_equity > 0.0
        else 0.0
    )
    return PortfolioBacktestState(
        current_weights=current_weights,
        gross_equity=float(gross_equity),
        net_equity=float(net_equity),
        net_peak_equity=float(net_peak_equity),
        current_drawdown=current_drawdown,
        holding_period_days=(
            1
            if accounting.turnover > 0.0
            else state.holding_period_days + 1
        ),
        recent_turnover=float(accounting.turnover),
        rebalance_step=state.rebalance_step + 1,
    )


def _subject_ids(backtest_input: DecisionBacktestInput) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                *(item.subject_id for item in backtest_input.subject_series),
                *(item.subject_id for item in backtest_input.initial_positions),
            }
        )
    )


def _contract_multiplier_for_subject(
    backtest_input: DecisionBacktestInput,
    *,
    subject_id: str,
) -> float | None:
    for item in backtest_input.subject_series:
        if item.subject_id == subject_id:
            if item.contract_multiplier is None:
                return None
            return float(item.contract_multiplier)
    return None


def _aligned_frame(backtest_input: DecisionBacktestInput) -> pd.DataFrame:
    subject_frames: list[pd.DataFrame] = []
    common_index: pd.Index | None = None
    for item in backtest_input.subject_series:
        realized_index = item.realized_return_series.astype(float).sort_index().index
        common_index = (
            realized_index
            if common_index is None
            else common_index.intersection(realized_index)
        )
    if common_index is None or common_index.empty:
        return pd.DataFrame()
    common_index = common_index.sort_values()
    for item in backtest_input.subject_series:
        subject_frame = pd.DataFrame(
            {
                ("signal", item.subject_id): item.signal_series.astype(float).reindex(common_index),
                ("realized_return", item.subject_id): item.realized_return_series.astype(float).reindex(common_index),
            }
        )
        if item.confidence_series is not None:
            subject_frame[("confidence", item.subject_id)] = item.confidence_series.astype(float).reindex(common_index)
        if item.risk_series is not None:
            subject_frame[("risk", item.subject_id)] = item.risk_series.astype(float).reindex(common_index)
        if item.uncertainty_series is not None:
            subject_frame[("uncertainty", item.subject_id)] = item.uncertainty_series.astype(float).reindex(common_index)
        if item.funding_cost_bps_series is not None:
            subject_frame[("funding_cost_bps", item.subject_id)] = item.funding_cost_bps_series.astype(float).reindex(common_index)
        if item.borrow_fee_bps_series is not None:
            subject_frame[("borrow_fee_bps", item.subject_id)] = item.borrow_fee_bps_series.astype(float).reindex(common_index)
        if item.roll_cost_bps_series is not None:
            subject_frame[("roll_cost_bps", item.subject_id)] = item.roll_cost_bps_series.astype(float).reindex(common_index)
        subject_frames.append(subject_frame)
    for item in backtest_input.dependence_series:
        subject_frames.append(
            pd.DataFrame(
                {
                    _dependence_column(
                        item.left_subject_id,
                        item.right_subject_id,
                    ): item.series.astype(float).reindex(common_index)
                }
            )
        )
    frame = pd.concat(subject_frames, axis=1).sort_index()
    required_columns = [
        ("realized_return", item.subject_id)
        for item in backtest_input.subject_series
    ]
    return frame.dropna(subset=required_columns)


def hold_position_target(
    *,
    subject_id: str,
    current_weight: float,
    capital_base: float,
) -> PortfolioTarget:
    return PortfolioTarget(
        subject_id=subject_id,
        target_weight=float(current_weight),
        position_delta=0.0,
        target_notional=float(current_weight * capital_base),
        entry_allowed=abs(current_weight) > 0.0,
        risk_scale=1.0,
    )


def constrained_targets_by_subject(
    targets: tuple[PortfolioTarget, ...],
    *,
    current_weights: dict[str, float],
    capital_base: float,
    gross_exposure_cap: float | None,
    gross_leverage_cap: float | None,
    net_exposure_target: float | None,
    target_vol: float | None = None,
    risk_by_subject: dict[str, float] | None = None,
    constraint_boundary: PortfolioConstraintBoundary | None = None,
    top_k: int | None,
    asset_class_by_subject: dict[str, str],
    cluster_by_subject: dict[str, str],
    asset_class_weight_caps: dict[str, float],
    cluster_weight_caps: dict[str, float],
    active_weight_budget: float | None = None,
    direction_mode: str | None = None,
) -> dict[str, PortfolioTarget]:
    request = build_portfolio_construction_request(
        targets=targets,
        current_weights=current_weights,
        capital_base=capital_base,
        gross_exposure_cap=gross_exposure_cap,
        gross_leverage_cap=gross_leverage_cap,
        net_exposure_target=net_exposure_target,
        target_vol=target_vol,
        risk_by_subject=risk_by_subject,
        constraint_boundary=constraint_boundary or default_portfolio_constraint_boundary(),
        direction_mode=direction_mode,
        top_k=top_k,
        active_weight_budget=active_weight_budget,
        asset_class_by_subject=asset_class_by_subject,
        cluster_by_subject=cluster_by_subject,
        asset_class_weight_caps=asset_class_weight_caps,
        cluster_weight_caps=cluster_weight_caps,
    )
    return construct_portfolio_targets(request).targets


def _predictive_signals_for_row(
    backtest_input: DecisionBacktestInput,
    *,
    row: pd.Series,
    subject_ids: tuple[str, ...],
) -> tuple[PredictiveSignalInput, ...]:
    items: list[PredictiveSignalInput] = []
    series_by_subject = {item.subject_id: item for item in backtest_input.subject_series}
    for subject_id in subject_ids:
        signal_value = _optional_value(row, ("signal", subject_id))
        if signal_value is None:
            continue
        subject_series = series_by_subject.get(subject_id)
        items.append(
            PredictiveSignalInput(
                source_id="backtest_signal",
                source_kind="backtest_signal",
                subject_id=subject_id,
                target_id=(
                    "residual_return_3d"
                    if subject_series is None
                    else subject_series.target_id
                ),
                value=signal_value,
                confidence=_optional_value(row, ("confidence", subject_id)),
            )
        )
    return tuple(items)


def _risk_inputs_for_row(
    backtest_input: DecisionBacktestInput,
    *,
    row: pd.Series,
    subject_ids: tuple[str, ...],
) -> tuple[RiskInput, ...]:
    items: list[RiskInput] = []
    for subject_id in subject_ids:
        risk_value = _optional_value(row, ("risk", subject_id))
        if risk_value is None:
            continue
        items.append(
            RiskInput(
                name="backtest_risk",
                subject_id=subject_id,
                value=max(risk_value, 0.0),
            )
        )
    if backtest_input.portfolio_construction.gross_exposure_cap is not None:
        items.append(
            RiskInput(
                name="gross_exposure_cap",
                subject_id=None,
                value=float(backtest_input.portfolio_construction.gross_exposure_cap),
                unit="weight",
            )
        )
    return tuple(items)


def _cost_inputs_for_backtest(
    backtest_input: DecisionBacktestInput,
    *,
    subject_ids: tuple[str, ...],
) -> tuple[CostInput, ...]:
    items: list[CostInput] = []
    if backtest_input.trading_environment.turnover_cost_rate > 0.0:
        items.append(
            CostInput(
                name="turnover_cost_rate",
                subject_id=None,
                value=float(backtest_input.trading_environment.turnover_cost_rate),
                basis="per_turnover",
                unit="weight",
            )
        )
    if backtest_input.trading_environment.fee_bps > 0.0:
        items.append(
            CostInput(
                name="fee_bps",
                subject_id=None,
                value=float(backtest_input.trading_environment.fee_bps),
                basis="per_notional",
                unit="bps",
            )
        )
    if backtest_input.trading_environment.bid_ask_spread_bps > 0.0:
        items.append(
            CostInput(
                name="bid_ask_spread_bps",
                subject_id=None,
                value=float(backtest_input.trading_environment.bid_ask_spread_bps),
                basis="per_notional",
                unit="bps",
            )
        )
    if backtest_input.trading_environment.funding_bps_per_step > 0.0:
        items.append(
            CostInput(
                name="funding_bps_per_step",
                subject_id=None,
                value=float(backtest_input.trading_environment.funding_bps_per_step),
                basis="per_notional_per_step",
                unit="bps",
            )
        )
    if backtest_input.trading_environment.borrow_fee_bps_per_step > 0.0:
        items.append(
            CostInput(
                name="borrow_fee_bps_per_step",
                subject_id=None,
                value=float(backtest_input.trading_environment.borrow_fee_bps_per_step),
                basis="per_short_notional_per_step",
                unit="bps",
            )
        )
    for subject_id in subject_ids:
        if backtest_input.trading_environment.market_impact_bps > 0.0:
            items.append(
                CostInput(
                    name="market_impact",
                    subject_id=subject_id,
                    value=float(backtest_input.trading_environment.market_impact_bps),
                    basis="per_notional",
                    unit="bps",
                )
            )
    return tuple(items)


def _uncertainty_inputs_for_row(
    backtest_input: DecisionBacktestInput,
    *,
    row: pd.Series,
    subject_ids: tuple[str, ...],
) -> tuple[UncertaintyInput, ...]:
    items: list[UncertaintyInput] = []
    for subject_id in subject_ids:
        uncertainty_value = _optional_value(row, ("uncertainty", subject_id))
        if uncertainty_value is None:
            continue
        items.append(
            UncertaintyInput(
                subject_id=subject_id,
                source_id="backtest_signal",
                estimate_std=max(uncertainty_value, 0.0),
                basis="per_signal",
                proxy_components={"backtest_uncertainty": max(uncertainty_value, 0.0)},
            )
        )
    return tuple(items)


def _dependence_inputs_for_row(
    backtest_input: DecisionBacktestInput,
    *,
    row: pd.Series,
) -> tuple[DependenceInput, ...]:
    items: list[DependenceInput] = []
    for item in backtest_input.dependence_series:
        dependence_value = _optional_value(
            row,
            _dependence_column(item.left_subject_id, item.right_subject_id),
        )
        if dependence_value is None:
            continue
        items.append(
            DependenceInput(
                name="backtest_dependence",
                left_subject_id=item.left_subject_id,
                right_subject_id=item.right_subject_id,
                value=max(dependence_value, 0.0),
                basis="overlap",
            )
        )
    return tuple(items)


def _dependence_column(
    left_subject_id: str,
    right_subject_id: str,
) -> tuple[str, tuple[str, str]]:
    ordered_pair = tuple(sorted((left_subject_id, right_subject_id)))
    return ("dependence", ordered_pair)


def _historical_return_inputs_for_row(
    backtest_input: DecisionBacktestInput,
    *,
    date: str,
) -> tuple[HistoricalReturnInput, ...]:
    items: list[HistoricalReturnInput] = []
    for subject in backtest_input.subject_series:
        history = (
            (
                subject.historical_return_series
                if subject.historical_return_series is not None
                else subject.realized_return_series
            )
            .astype(float)
            .sort_index()
            .loc[lambda series: series.index < date]
        )
        if backtest_input.historical_return_lookback_steps == 0:
            continue
        if backtest_input.historical_return_lookback_steps is not None:
            history = history.tail(backtest_input.historical_return_lookback_steps)
        if history.empty:
            continue
        items.append(
            HistoricalReturnInput(
                subject_id=subject.subject_id,
                returns_by_date={
                    str(index): float(value)
                    for index, value in history.items()
                },
            )
        )
    return tuple(items)


def _optional_value(
    row: pd.Series,
    column: object,
    *,
    default: float | None = None,
) -> float | None:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return float(value)


def _required_value(
    row: pd.Series,
    column: object,
) -> float:
    value = _optional_value(row, column)
    if value is None:
        raise ValueError(f"missing required backtest value for column={column!r}")
    return value
