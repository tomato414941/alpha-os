from __future__ import annotations

from dataclasses import dataclass

from strategies.crypto.momentum import TargetWeights


@dataclass(frozen=True)
class AccountingResult:
    reward: float
    gross_reward: float
    transaction_cost: float
    gross_contribution_by_symbol: dict[str, float]
    equity: float


class PortfolioAccounting:
    def __init__(
        self,
        *,
        initial_equity: float = 1.0,
        transaction_cost_rate: float = 0.001,
    ) -> None:
        self._transaction_cost_rate = transaction_cost_rate
        self._equity = initial_equity
        self._current_weights: dict[str, float] = {}

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def current_weights(self) -> dict[str, float]:
        return dict(self._current_weights)

    def apply(
        self,
        target: TargetWeights,
        *,
        returns_by_symbol: dict[str, float],
    ) -> AccountingResult:
        gross_reward = sum(
            target.target_weights.get(symbol, 0.0) * symbol_return
            for symbol, symbol_return in returns_by_symbol.items()
        )
        gross_contribution_by_symbol = {
            symbol: target.target_weights.get(symbol, 0.0) * symbol_return
            for symbol, symbol_return in returns_by_symbol.items()
        }
        transaction_cost = (
            _turnover(self._current_weights, target.target_weights)
            * self._transaction_cost_rate
        )
        reward = gross_reward - transaction_cost
        self._equity *= 1.0 + reward
        self._current_weights = dict(target.target_weights)
        return AccountingResult(
            reward=reward,
            gross_reward=gross_reward,
            transaction_cost=transaction_cost,
            gross_contribution_by_symbol=gross_contribution_by_symbol,
            equity=self._equity,
        )


def _turnover(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> float:
    symbols = current_weights.keys() | target_weights.keys()
    return sum(
        abs(target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0))
        for symbol in symbols
    )
