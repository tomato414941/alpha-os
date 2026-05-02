from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .contract_boundaries import (
    PortfolioConstraintBoundary,
    default_portfolio_constraint_boundary,
)
from .portfolio_construction_config import PortfolioConstructionSpec, PortfolioRiskBudgetSpec
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
from .portfolio_direction import normalize_portfolio_direction_mode
from .portfolio_construction_pipeline import (
    PortfolioConstructionStageTrace,
    build_portfolio_construction_request,
    construct_portfolio_targets,
)
from .portfolio_execution_policy import (
    ExecutionPolicySpec,
    TradeTransitionRequest,
    TradeTransitionTrace,
    apply_execution_policy,
)
from .portfolio_overlay import ActiveOverlaySpec
from .portfolio_sizing_policy import (
    PortfolioSizingPolicy,
    apply_portfolio_sizing_policy,
)
from .strategy_sleeve_composition import (
    SleeveSignalContribution,
    compose_portfolio_decision_input,
)
from .strategy_sleeves import SleeveAttributionSummary, StrategySleeveCompositionSpec
from .targets import find_target_definition


@dataclass(frozen=True)
class SubjectBacktestSeries:
    subject_id: str
    signal_series: pd.Series
    realized_return_series: pd.Series
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
    portfolio_id: str
    subject_set_id: str | None
    target_id: str
    subject_series: tuple[SubjectBacktestSeries, ...]
    portfolio_construction: PortfolioConstructionSpec | None = None
    initial_capital_base: float = 1.0
    initial_positions: tuple[PortfolioPositionState, ...] = ()
    initial_holding_period_days: int = 0
    dependence_series: tuple[DependenceBacktestSeries, ...] = ()
    asset_class_by_subject: dict[str, str] | None = None
    cluster_by_subject: dict[str, str] | None = None
    asset_class_weight_caps: dict[str, float] | None = None
    cluster_weight_caps: dict[str, float] | None = None
    gross_exposure_cap: float | None = None
    target_vol: float | None = None
    gross_leverage_cap: float | None = None
    net_exposure_target: float | None = None
    risk_budget: PortfolioRiskBudgetSpec | None = None
    turnover_friction: float = 0.0
    market_impact_bps: float = 0.0
    fee_bps: float = 0.0
    bid_ask_spread_bps: float = 0.0
    funding_bps_per_step: float = 0.0
    borrow_fee_bps_per_step: float = 0.0
    execution_cost_aversion: float = 1.0
    execution_mode: str = "utility_priority"
    benefit_scale: float = 1.0
    min_trade_utility: float = 0.0
    uncertainty_aversion: float = 1.0
    risk_aversion: float = 0.0
    partial_fill_enabled: bool = True
    no_trade_band: float = 0.0
    turnover_budget: float | None = None
    execution_policy: ExecutionPolicySpec | None = None
    rebalance_interval_steps: int = 1
    long_only: bool = False
    direction_mode: str | None = None
    top_k: int | None = None
    active_overlay: ActiveOverlaySpec | None = field(default_factory=ActiveOverlaySpec)
    historical_return_lookback_steps: int | None = None
    sleeve_composition: StrategySleeveCompositionSpec | None = None
    subject_metadata_by_subject: dict[str, dict[str, str]] | None = None

    @property
    def subject_ids(self) -> tuple[str, ...]:
        return tuple(item.subject_id for item in self.subject_series)

    def __post_init__(self) -> None:
        construction = self.portfolio_construction
        if construction is not None:
            object.__setattr__(
                self,
                "asset_class_weight_caps",
                dict(construction.asset_class_weight_caps),
            )
            object.__setattr__(
                self,
                "cluster_weight_caps",
                dict(construction.cluster_weight_caps),
            )
            object.__setattr__(self, "gross_exposure_cap", construction.gross_exposure_cap)
            object.__setattr__(self, "target_vol", construction.target_vol)
            object.__setattr__(self, "gross_leverage_cap", construction.gross_leverage_cap)
            object.__setattr__(self, "net_exposure_target", construction.net_exposure_target)
            object.__setattr__(self, "risk_budget", construction.risk_budget)
            object.__setattr__(
                self,
                "rebalance_interval_steps",
                construction.rebalance_interval_steps,
            )
            object.__setattr__(self, "long_only", construction.long_only)
            object.__setattr__(self, "direction_mode", construction.direction_mode)
            object.__setattr__(self, "top_k", construction.top_k)
            object.__setattr__(self, "active_overlay", construction.active_overlay)
            object.__setattr__(
                self,
                "sleeve_composition",
                construction.sleeve_composition,
            )
        direction_mode = normalize_portfolio_direction_mode(
            self.direction_mode,
            long_only=self.long_only,
        )
        object.__setattr__(self, "direction_mode", direction_mode)
        object.__setattr__(self, "long_only", direction_mode == "long_only")


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
    execution_trace: TradeTransitionTrace | None = None
    sizing_diagnostics: SizingDiagnostics = field(default_factory=SizingDiagnostics)

    @property
    def subject_step_by_subject(self) -> dict[str, DecisionBacktestSubjectStep]:
        return {item.subject_id: item for item in self.subject_steps}


@dataclass(frozen=True)
class DecisionBacktestResult:
    portfolio_id: str
    subject_set_id: str | None
    target_id: str
    subject_ids: tuple[str, ...]
    steps: tuple[DecisionBacktestStep, ...]
    initial_capital_base: float = 1.0
    sleeve_attribution_summaries: tuple[SleeveAttributionSummary, ...] = ()

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
    sleeve_attribution_builder = _SleeveBacktestAttributionBuilder(
        backtest_input.sleeve_composition
    )

    for date, row in aligned.iterrows():
        if _is_rebalance_step(state, backtest_input):
            (
                desired_targets_by_subject,
                sleeve_composition_result,
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
            execution_result = apply_execution_policy(
                TradeTransitionRequest(
                    desired_targets=desired_targets_by_subject,
                    current_weights=state.current_weights,
                    capital_base=state.capital_base,
                    execution_policy=_execution_policy_for_backtest(backtest_input),
                    recent_turnover=state.recent_turnover,
                    holding_period_days=state.holding_period_days,
                    signal_horizon_by_subject=_signal_horizon_by_subject(
                        backtest_input,
                        subject_ids=subject_ids,
                    ),
                    signal_value_by_subject=_execution_signal_value_by_subject(
                        row,
                        subject_ids=subject_ids,
                    ),
                    confidence_by_subject=_execution_optional_value_by_subject(
                        row,
                        subject_ids=subject_ids,
                        column_name="confidence",
                        default=1.0,
                    ),
                    uncertainty_by_subject=_execution_optional_value_by_subject(
                        row,
                        subject_ids=subject_ids,
                        column_name="uncertainty",
                        default=0.0,
                    ),
                    risk_by_subject=_execution_optional_value_by_subject(
                        row,
                        subject_ids=subject_ids,
                        column_name="risk",
                        default=0.0,
                    ),
                    execution_friction_level=_execution_friction_level(backtest_input),
                    per_turnover_cost=_per_turnover_execution_cost(backtest_input),
                )
            )
            targets_by_subject = execution_result.executed_targets
            sleeve_attribution_builder.capture_rebalance(
                sleeve_composition_result.contributions
                if sleeve_composition_result is not None
                else (),
                targets_by_subject=targets_by_subject,
                capital_base=state.capital_base,
                row=row,
            )
            execution_trace = execution_result.trace
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
        portfolio_id=backtest_input.portfolio_id,
        subject_set_id=backtest_input.subject_set_id,
        target_id=backtest_input.target_id,
        subject_ids=subject_ids,
        steps=tuple(steps),
        initial_capital_base=max(float(backtest_input.initial_capital_base), 0.0),
        sleeve_attribution_summaries=sleeve_attribution_builder.summaries(),
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
    return state.rebalance_step % max(backtest_input.rebalance_interval_steps, 1) == 0


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
    object | None,
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
    decision_input, sleeve_composition_result = compose_portfolio_decision_input(
        decision_input
    )
    decision_output = apply_portfolio_sizing_policy(
        decision_input,
        sizing_policy=sizing_policy,
        apply_rebalance_friction=False,
    )
    construction_result = construct_portfolio_targets(
        build_portfolio_construction_request(
            targets=decision_output.targets,
            current_weights=state.current_weights,
            capital_base=state.capital_base,
            gross_exposure_cap=backtest_input.gross_exposure_cap,
            gross_leverage_cap=backtest_input.gross_leverage_cap,
            net_exposure_target=backtest_input.net_exposure_target,
            target_vol=backtest_input.target_vol,
            risk_budget=backtest_input.risk_budget,
            risk_by_subject={
                subject_id: max(
                    float(_optional_value(row, ("risk", subject_id)) or 0.0),
                    0.0,
                )
                for subject_id in subject_ids
            },
            constraint_boundary=default_portfolio_constraint_boundary(),
            long_only=backtest_input.long_only,
            direction_mode=backtest_input.direction_mode,
            top_k=backtest_input.top_k,
            active_overlay=backtest_input.active_overlay,
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
                if backtest_input.asset_class_weight_caps is None
                else backtest_input.asset_class_weight_caps
            ),
            cluster_weight_caps=(
                {}
                if backtest_input.cluster_weight_caps is None
                else backtest_input.cluster_weight_caps
            ),
        )
    )
    return (
        construction_result.targets,
        sleeve_composition_result,
        construction_result.trace,
        decision_output.sizing_diagnostics,
    )


def _execution_policy_for_backtest(
    backtest_input: DecisionBacktestInput,
) -> ExecutionPolicySpec:
    if backtest_input.execution_policy is not None:
        return backtest_input.execution_policy
    return ExecutionPolicySpec.from_cost_controls(
        no_trade_band=backtest_input.no_trade_band,
        turnover_budget=backtest_input.turnover_budget,
        turnover_friction=backtest_input.turnover_friction,
        market_impact_bps=backtest_input.market_impact_bps,
        fee_bps=backtest_input.fee_bps,
        bid_ask_spread_bps=backtest_input.bid_ask_spread_bps,
        execution_cost_aversion=backtest_input.execution_cost_aversion,
        mode=backtest_input.execution_mode,
        benefit_scale=backtest_input.benefit_scale,
        min_trade_utility=backtest_input.min_trade_utility,
        uncertainty_aversion=backtest_input.uncertainty_aversion,
        risk_aversion=backtest_input.risk_aversion,
        partial_fill_enabled=backtest_input.partial_fill_enabled,
    )


def _per_turnover_execution_cost(backtest_input: DecisionBacktestInput) -> float:
    return (
        max(backtest_input.turnover_friction, 0.0)
        + max(backtest_input.market_impact_bps, 0.0) / 10000.0
        + max(backtest_input.fee_bps, 0.0) / 10000.0
        + max(backtest_input.bid_ask_spread_bps, 0.0) / 10000.0
    )


def _execution_friction_level(backtest_input: DecisionBacktestInput) -> float:
    return (
        max(backtest_input.turnover_friction, 0.0)
        + max(backtest_input.market_impact_bps, 0.0) / 10000.0
    )


def _signal_horizon_by_subject(
    backtest_input: DecisionBacktestInput,
    *,
    subject_ids: tuple[str, ...],
) -> dict[str, int | None]:
    target_definition = find_target_definition(backtest_input.target_id)
    horizon_days = None if target_definition is None else target_definition.horizon_days
    return {subject_id: horizon_days for subject_id in subject_ids}


def _execution_signal_value_by_subject(
    row: pd.Series,
    *,
    subject_ids: tuple[str, ...],
) -> dict[str, float]:
    return {
        subject_id: float(
            _optional_value(row, ("signal", subject_id), default=0.0)
        )
        for subject_id in subject_ids
    }


def _execution_optional_value_by_subject(
    row: pd.Series,
    *,
    subject_ids: tuple[str, ...],
    column_name: str,
    default: float,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for subject_id in subject_ids:
        value = _optional_value(row, (column_name, subject_id), default=default)
        values[subject_id] = float(default if value is None else value)
    return values


def _portfolio_decision_input_for_backtest_row(
    backtest_input: DecisionBacktestInput,
    *,
    state: PortfolioBacktestState,
    row: pd.Series,
    date: str,
    subject_ids: tuple[str, ...],
) -> PortfolioDecisionInput:
    return PortfolioDecisionInput(
        portfolio_id=backtest_input.portfolio_id,
        as_of=date,
        portfolio_state=PortfolioState(
            portfolio_id=backtest_input.portfolio_id,
            as_of=date,
            positions=tuple(
                PortfolioPositionState(
                    subject_id=subject_id,
                    weight=state.current_weights.get(subject_id, 0.0),
                )
                for subject_id in subject_ids
            ),
            capital_base=state.net_equity,
            gross_limit=backtest_input.gross_exposure_cap,
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
        sleeve_composition=backtest_input.sleeve_composition,
        subject_metadata_by_subject=(
            {}
            if backtest_input.subject_metadata_by_subject is None
            else backtest_input.subject_metadata_by_subject
        ),
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
            max(backtest_input.turnover_friction, 0.0) * traded_notional
            + max(backtest_input.market_impact_bps, 0.0) / 10000.0 * traded_notional
            + max(backtest_input.fee_bps, 0.0) / 10000.0 * traded_notional
            + max(backtest_input.bid_ask_spread_bps, 0.0) / 10000.0 * traded_notional
        )
        funding_cost_notional = (
            max(backtest_input.funding_bps_per_step, 0.0)
            / 10000.0
            * abs(target_notional)
            + float(funding_cost_bps) / 10000.0 * target_notional
        )
        short_notional = abs(min(target_notional, 0.0))
        borrow_cost_notional = (
            max(backtest_input.borrow_fee_bps_per_step, 0.0)
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
        max(backtest_input.funding_bps_per_step, 0.0)
        / 10000.0
        * gross_notional_exposure
        + subject_funding_cost_notional
    )
    borrow_cost_notional = (
        max(backtest_input.borrow_fee_bps_per_step, 0.0)
        / 10000.0
        * short_notional_exposure
        + subject_borrow_cost_notional
    )
    cost_notional = (
        max(backtest_input.turnover_friction, 0.0) * traded_notional
        + max(backtest_input.market_impact_bps, 0.0) / 10000.0 * traded_notional
        + max(backtest_input.fee_bps, 0.0) / 10000.0 * traded_notional
        + max(backtest_input.bid_ask_spread_bps, 0.0) / 10000.0 * traded_notional
        + max(backtest_input.funding_bps_per_step, 0.0)
        / 10000.0
        * gross_notional_exposure
        + max(backtest_input.borrow_fee_bps_per_step, 0.0)
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


class _SleeveBacktestAttributionBuilder:
    def __init__(self, composition: StrategySleeveCompositionSpec | None) -> None:
        self._composition = composition
        self._subject_ids: dict[str, set[str]] = {}
        self._signal_sum: dict[str, float] = {}
        self._abs_signal_sum: dict[str, float] = {}
        self._signal_count: dict[str, int] = {}
        self._gross_exposure_sum: dict[str, float] = {}
        self._net_exposure_sum: dict[str, float] = {}
        self._long_exposure_sum: dict[str, float] = {}
        self._short_exposure_sum: dict[str, float] = {}
        self._funding_cost_sum: dict[str, float] = {}
        self._borrow_cost_sum: dict[str, float] = {}
        self._roll_cost_sum: dict[str, float] = {}
        self._rebalance_count = 0

    def capture_rebalance(
        self,
        contributions: tuple[SleeveSignalContribution, ...],
        *,
        targets_by_subject: dict[str, PortfolioTarget],
        capital_base: float,
        row: pd.Series,
    ) -> None:
        if self._composition is None:
            return
        self._rebalance_count += 1
        contribution_weight_by_subject = self._contribution_weight_by_subject(contributions)
        for contribution in contributions:
            sleeve_id = contribution.sleeve_id
            self._subject_ids.setdefault(sleeve_id, set()).add(contribution.subject_id)
            self._signal_sum[sleeve_id] = (
                self._signal_sum.get(sleeve_id, 0.0) + contribution.raw_signal_value
            )
            self._abs_signal_sum[sleeve_id] = (
                self._abs_signal_sum.get(sleeve_id, 0.0)
                + abs(contribution.raw_signal_value)
            )
            self._signal_count[sleeve_id] = self._signal_count.get(sleeve_id, 0) + 1
        for contribution in contributions:
            subject_weights = contribution_weight_by_subject.get(contribution.subject_id, {})
            allocation_weight = subject_weights.get(contribution.sleeve_id, 0.0)
            if allocation_weight <= 0.0:
                continue
            target = targets_by_subject.get(contribution.subject_id)
            if target is None:
                continue
            target_notional = (
                float(target.target_notional)
                if target.target_notional is not None
                else float(target.target_weight) * capital_base
            )
            sleeve_id = contribution.sleeve_id
            self._gross_exposure_sum[sleeve_id] = (
                self._gross_exposure_sum.get(sleeve_id, 0.0)
                + abs(target_notional) * allocation_weight
            )
            self._net_exposure_sum[sleeve_id] = (
                self._net_exposure_sum.get(sleeve_id, 0.0)
                + target_notional * allocation_weight
            )
            self._long_exposure_sum[sleeve_id] = (
                self._long_exposure_sum.get(sleeve_id, 0.0)
                + max(target_notional, 0.0) * allocation_weight
            )
            self._short_exposure_sum[sleeve_id] = (
                self._short_exposure_sum.get(sleeve_id, 0.0)
                + abs(min(target_notional, 0.0)) * allocation_weight
            )
            funding_cost = (
                float(_optional_value(row, ("funding_cost_bps", contribution.subject_id), default=0.0))
                / 10000.0
                * target_notional
            )
            borrow_cost = (
                max(
                    float(_optional_value(row, ("borrow_fee_bps", contribution.subject_id), default=0.0)),
                    0.0,
                )
                / 10000.0
                * abs(min(target_notional, 0.0))
            )
            roll_cost = (
                max(
                    float(_optional_value(row, ("roll_cost_bps", contribution.subject_id), default=0.0)),
                    0.0,
                )
                / 10000.0
                * abs(target_notional)
            )
            self._funding_cost_sum[sleeve_id] = (
                self._funding_cost_sum.get(sleeve_id, 0.0)
                + funding_cost * allocation_weight
            )
            self._borrow_cost_sum[sleeve_id] = (
                self._borrow_cost_sum.get(sleeve_id, 0.0)
                + borrow_cost * allocation_weight
            )
            self._roll_cost_sum[sleeve_id] = (
                self._roll_cost_sum.get(sleeve_id, 0.0)
                + roll_cost * allocation_weight
            )

    def summaries(self) -> tuple[SleeveAttributionSummary, ...]:
        if self._composition is None:
            return ()
        divisor = max(self._rebalance_count, 1)
        summaries: list[SleeveAttributionSummary] = []
        for sleeve in self._composition.enabled_sleeves:
            signal_count = max(self._signal_count.get(sleeve.sleeve_id, 0), 1)
            funding_cost = self._funding_cost_sum.get(sleeve.sleeve_id, 0.0)
            borrow_cost = self._borrow_cost_sum.get(sleeve.sleeve_id, 0.0)
            roll_cost = self._roll_cost_sum.get(sleeve.sleeve_id, 0.0)
            summaries.append(
                SleeveAttributionSummary(
                    sleeve_id=sleeve.sleeve_id,
                    sleeve_kind=sleeve.sleeve_kind,
                    risk_budget=sleeve.risk_budget,
                    subject_count=len(self._subject_ids.get(sleeve.sleeve_id, set())),
                    mean_signal=self._signal_sum.get(sleeve.sleeve_id, 0.0) / signal_count,
                    mean_abs_signal=(
                        self._abs_signal_sum.get(sleeve.sleeve_id, 0.0) / signal_count
                    ),
                    mean_gross_notional_exposure=(
                        self._gross_exposure_sum.get(sleeve.sleeve_id, 0.0) / divisor
                    ),
                    mean_net_notional_exposure=(
                        self._net_exposure_sum.get(sleeve.sleeve_id, 0.0) / divisor
                    ),
                    mean_long_notional_exposure=(
                        self._long_exposure_sum.get(sleeve.sleeve_id, 0.0) / divisor
                    ),
                    mean_short_notional_exposure=(
                        self._short_exposure_sum.get(sleeve.sleeve_id, 0.0) / divisor
                    ),
                    total_cost_notional=funding_cost + borrow_cost + roll_cost,
                    total_funding_cost_notional=funding_cost,
                    total_borrow_cost_notional=borrow_cost,
                    total_roll_cost_notional=roll_cost,
                )
            )
        return tuple(summaries)

    @staticmethod
    def _contribution_weight_by_subject(
        contributions: tuple[SleeveSignalContribution, ...],
    ) -> dict[str, dict[str, float]]:
        values_by_subject: dict[str, dict[str, float]] = {}
        for contribution in contributions:
            values_by_subject.setdefault(contribution.subject_id, {})[
                contribution.sleeve_id
            ] = abs(contribution.weighted_signal_value)
        weights_by_subject: dict[str, dict[str, float]] = {}
        for subject_id, values in values_by_subject.items():
            total = sum(values.values())
            if total <= 0.0:
                continue
            weights_by_subject[subject_id] = {
                sleeve_id: value / total
                for sleeve_id, value in values.items()
            }
        return weights_by_subject


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
    risk_budget: PortfolioRiskBudgetSpec | None = None,
    constraint_boundary: PortfolioConstraintBoundary | None = None,
    long_only: bool,
    top_k: int | None,
    asset_class_by_subject: dict[str, str],
    cluster_by_subject: dict[str, str],
    asset_class_weight_caps: dict[str, float],
    cluster_weight_caps: dict[str, float],
    active_overlay: ActiveOverlaySpec | None = None,
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
        risk_budget=risk_budget,
        constraint_boundary=constraint_boundary or default_portfolio_constraint_boundary(),
        long_only=long_only,
        direction_mode=direction_mode,
        top_k=top_k,
        active_overlay=active_overlay,
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
    for subject_id in subject_ids:
        signal_value = _optional_value(row, ("signal", subject_id))
        if signal_value is None:
            continue
        items.append(
            PredictiveSignalInput(
                source_id="backtest_signal",
                source_kind="backtest_signal",
                subject_id=subject_id,
                target_id=backtest_input.target_id,
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
    if backtest_input.gross_exposure_cap is not None:
        items.append(
            RiskInput(
                name="gross_exposure_cap",
                subject_id=None,
                value=float(backtest_input.gross_exposure_cap),
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
    if backtest_input.turnover_friction > 0.0:
        items.append(
            CostInput(
                name="turnover_friction",
                subject_id=None,
                value=float(backtest_input.turnover_friction),
                basis="per_turnover",
                unit="weight",
            )
        )
    if backtest_input.fee_bps > 0.0:
        items.append(
            CostInput(
                name="fee_bps",
                subject_id=None,
                value=float(backtest_input.fee_bps),
                basis="per_notional",
                unit="bps",
            )
        )
    if backtest_input.bid_ask_spread_bps > 0.0:
        items.append(
            CostInput(
                name="bid_ask_spread_bps",
                subject_id=None,
                value=float(backtest_input.bid_ask_spread_bps),
                basis="per_notional",
                unit="bps",
            )
        )
    if backtest_input.funding_bps_per_step > 0.0:
        items.append(
            CostInput(
                name="funding_bps_per_step",
                subject_id=None,
                value=float(backtest_input.funding_bps_per_step),
                basis="per_notional_per_step",
                unit="bps",
            )
        )
    if backtest_input.borrow_fee_bps_per_step > 0.0:
        items.append(
            CostInput(
                name="borrow_fee_bps_per_step",
                subject_id=None,
                value=float(backtest_input.borrow_fee_bps_per_step),
                basis="per_short_notional_per_step",
                unit="bps",
            )
        )
    for subject_id in subject_ids:
        if backtest_input.market_impact_bps > 0.0:
            items.append(
                CostInput(
                    name="market_impact",
                    subject_id=subject_id,
                    value=float(backtest_input.market_impact_bps),
                    basis="per_notional",
                    unit="bps",
                )
            )
        if backtest_input.no_trade_band > 0.0:
            items.append(
                CostInput(
                    name="no_trade_band",
                    subject_id=subject_id,
                    value=float(backtest_input.no_trade_band),
                    basis="per_delta_weight",
                    unit="weight",
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
