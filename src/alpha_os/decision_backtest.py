from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .portfolio_decision import (
    CostInput,
    DependenceInput,
    PortfolioDecisionInput,
    PortfolioPositionState,
    PortfolioState,
    PredictiveSignalInput,
    RiskInput,
    UncertaintyInput,
)
from .portfolio_decision_policy import (
    RuleBasedPortfolioPolicy,
    apply_rule_based_policy,
)


@dataclass(frozen=True)
class DecisionBacktestInput:
    portfolio_id: str
    subject_id: str
    target_id: str
    signal_series: pd.Series
    realized_return_series: pd.Series
    asset: str | None = None
    initial_weight: float = 0.0
    confidence_series: pd.Series | None = None
    risk_series: pd.Series | None = None
    uncertainty_series: pd.Series | None = None
    dependence_series: pd.Series | None = None
    gross_exposure_cap: float | None = None
    turnover_penalty: float = 0.0
    expected_slippage_bps: float = 0.0
    no_trade_band: float = 0.0


@dataclass(frozen=True)
class DecisionBacktestStep:
    date: str
    signal_value: float
    target_weight: float
    position_delta: float
    realized_return: float
    gross_return: float
    turnover: float
    cost: float
    net_return: float
    gross_equity: float
    net_equity: float
    risk_scale: float
    entry_allowed: bool


@dataclass(frozen=True)
class DecisionBacktestResult:
    portfolio_id: str
    subject_id: str
    target_id: str
    steps: tuple[DecisionBacktestStep, ...]

    @property
    def gross_return_total(self) -> float:
        if not self.steps:
            return 0.0
        return float(self.steps[-1].gross_equity - 1.0)

    @property
    def net_return_total(self) -> float:
        if not self.steps:
            return 0.0
        return float(self.steps[-1].net_equity - 1.0)

    @property
    def mean_turnover(self) -> float:
        if not self.steps:
            return 0.0
        return float(sum(step.turnover for step in self.steps) / len(self.steps))

    @property
    def max_drawdown(self) -> float:
        peak = 1.0
        drawdown = 0.0
        for step in self.steps:
            peak = max(peak, step.net_equity)
            if peak > 0.0:
                drawdown = max(drawdown, 1.0 - (step.net_equity / peak))
        return float(drawdown)


def run_decision_backtest(
    backtest_input: DecisionBacktestInput,
    *,
    policy: RuleBasedPortfolioPolicy | None = None,
) -> DecisionBacktestResult:
    aligned = _aligned_frame(backtest_input)
    current_weight = float(backtest_input.initial_weight)
    gross_equity = 1.0
    net_equity = 1.0
    steps: list[DecisionBacktestStep] = []

    for date, row in aligned.iterrows():
        decision_input = PortfolioDecisionInput(
            portfolio_id=backtest_input.portfolio_id,
            asset=backtest_input.asset,
            as_of=str(date),
            portfolio_state=PortfolioState(
                portfolio_id=backtest_input.portfolio_id,
                asset=backtest_input.asset,
                as_of=str(date),
                positions=(
                    PortfolioPositionState(
                        subject_id=backtest_input.subject_id,
                        weight=current_weight,
                    ),
                ),
            ),
            predictive_signals=(
                PredictiveSignalInput(
                    source_id="backtest_signal",
                    source_kind="backtest_signal",
                    subject_id=backtest_input.subject_id,
                    target_id=backtest_input.target_id,
                    value=float(row["signal"]),
                    confidence=_optional_value(row, "confidence"),
                ),
            ),
            risk_inputs=_risk_inputs_for_row(
                backtest_input,
                row=row,
            ),
            cost_inputs=_cost_inputs_for_backtest(backtest_input),
            uncertainty_inputs=_uncertainty_inputs_for_row(
                backtest_input,
                row=row,
            ),
            dependence_inputs=_dependence_inputs_for_row(
                backtest_input,
                row=row,
            ),
        )
        decision_output = apply_rule_based_policy(decision_input, policy=policy)
        target = decision_output.targets[0]
        turnover = abs(target.position_delta)
        cost = (
            max(backtest_input.turnover_penalty, 0.0) * turnover
            + max(backtest_input.expected_slippage_bps, 0.0) / 10000.0 * turnover
        )
        realized_return = float(row["realized_return"])
        gross_return = float(target.target_weight * realized_return)
        net_return = float(gross_return - cost)
        gross_equity *= 1.0 + gross_return
        net_equity *= 1.0 + net_return
        steps.append(
            DecisionBacktestStep(
                date=str(date),
                signal_value=float(row["signal"]),
                target_weight=float(target.target_weight),
                position_delta=float(target.position_delta),
                realized_return=realized_return,
                gross_return=gross_return,
                turnover=float(turnover),
                cost=float(cost),
                net_return=net_return,
                gross_equity=float(gross_equity),
                net_equity=float(net_equity),
                risk_scale=float(target.risk_scale),
                entry_allowed=target.entry_allowed,
            )
        )
        current_weight = float(target.target_weight)

    return DecisionBacktestResult(
        portfolio_id=backtest_input.portfolio_id,
        subject_id=backtest_input.subject_id,
        target_id=backtest_input.target_id,
        steps=tuple(steps),
    )


def _aligned_frame(backtest_input: DecisionBacktestInput) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "signal": backtest_input.signal_series.astype(float),
            "realized_return": backtest_input.realized_return_series.astype(float),
        }
    )
    if backtest_input.confidence_series is not None:
        frame["confidence"] = backtest_input.confidence_series.astype(float)
    if backtest_input.risk_series is not None:
        frame["risk"] = backtest_input.risk_series.astype(float)
    if backtest_input.uncertainty_series is not None:
        frame["uncertainty"] = backtest_input.uncertainty_series.astype(float)
    if backtest_input.dependence_series is not None:
        frame["dependence"] = backtest_input.dependence_series.astype(float)
    return frame.sort_index().dropna(subset=["signal", "realized_return"])


def _risk_inputs_for_row(
    backtest_input: DecisionBacktestInput,
    *,
    row: pd.Series,
) -> tuple[RiskInput, ...]:
    items: list[RiskInput] = []
    if "risk" in row:
        items.append(
            RiskInput(
                name="backtest_risk",
                subject_id=backtest_input.subject_id,
                value=max(float(row["risk"]), 0.0),
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
) -> tuple[CostInput, ...]:
    items: list[CostInput] = []
    if backtest_input.turnover_penalty > 0.0:
        items.append(
            CostInput(
                name="turnover_penalty",
                subject_id=None,
                value=float(backtest_input.turnover_penalty),
                basis="per_turnover",
                unit="weight",
            )
        )
    if backtest_input.expected_slippage_bps > 0.0:
        items.append(
            CostInput(
                name="expected_slippage",
                subject_id=backtest_input.subject_id,
                value=float(backtest_input.expected_slippage_bps),
                basis="per_notional",
                unit="bps",
            )
        )
    if backtest_input.no_trade_band > 0.0:
        items.append(
            CostInput(
                name="no_trade_band",
                subject_id=backtest_input.subject_id,
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
) -> tuple[UncertaintyInput, ...]:
    if "uncertainty" not in row:
        return ()
    return (
        UncertaintyInput(
            name="backtest_uncertainty",
            subject_id=backtest_input.subject_id,
            value=max(float(row["uncertainty"]), 0.0),
            source_id="backtest_signal",
            basis="per_signal",
        ),
    )


def _dependence_inputs_for_row(
    backtest_input: DecisionBacktestInput,
    *,
    row: pd.Series,
) -> tuple[DependenceInput, ...]:
    if "dependence" not in row:
        return ()
    return (
        DependenceInput(
            name="backtest_dependence",
            left_subject_id=backtest_input.subject_id,
            right_subject_id="__portfolio__",
            value=max(float(row["dependence"]), 0.0),
            basis="overlap",
        ),
    )


def _optional_value(row: pd.Series, column: str) -> float | None:
    if column not in row:
        return None
    return float(row[column])
