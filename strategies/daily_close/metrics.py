from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class BacktestSummary:
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    mean_daily_turnover: float


def summarize_backtest(
    *,
    rewards: tuple[float, ...],
    equities: tuple[float, ...],
    transaction_costs: tuple[float, ...],
    transaction_cost_rate: float,
) -> BacktestSummary:
    if not rewards or not equities:
        return BacktestSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mean_return = sum(rewards) / len(rewards)
    variance = sum((value - mean_return) ** 2 for value in rewards) / len(rewards)
    return BacktestSummary(
        total_return=equities[-1] - 1.0,
        annualized_return=(equities[-1] ** (365.0 / len(rewards))) - 1.0,
        annualized_volatility=sqrt(variance) * sqrt(365.0),
        sharpe=mean_return / sqrt(variance) * sqrt(365.0) if variance > 0.0 else 0.0,
        max_drawdown=_max_drawdown(equities),
        mean_daily_turnover=(
            sum(cost / transaction_cost_rate for cost in transaction_costs)
            / len(rewards)
            if transaction_cost_rate > 0.0
            else 0.0
        ),
    )


def _max_drawdown(equity_curve: tuple[float, ...]) -> float:
    peak = equity_curve[0]
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0.0:
            max_drawdown = min(max_drawdown, (equity / peak) - 1.0)
    return max_drawdown
