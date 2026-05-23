from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .data_repositories import FeaturePlaneRepository
from .evaluation_cost_config import TradingEnvironment
from .evaluation_spec import EvaluationDateRange
from .strategy_backtest import run_strategy_backtest_from_store
from .portfolio_construction_config import PortfolioConstructionSpec
from .evaluation_result import EvaluationResult
from .trading_strategy import TradingStrategySpec
from .universe_contract import validate_subject_set_universe_contract


class EvaluationExecutionReadPort(Protocol):
    def get_trading_strategy(self, strategy_id: str):
        ...

    def get_subject_set(self, subject_set_id: str):
        ...


@dataclass(frozen=True)
class EvaluationExecutionContext:
    store: EvaluationExecutionReadPort
    feature_plane_repository: FeaturePlaneRepository | None = None


def _portfolio_construction_for_strategy(
    trading_strategy: TradingStrategySpec,
) -> PortfolioConstructionSpec:
    return trading_strategy.portfolio.portfolio_construction


def _trading_environment_for_strategy(
    trading_strategy: TradingStrategySpec,
) -> TradingEnvironment:
    return trading_strategy.portfolio.trading_environment


def _trading_strategy_for_id(
    store: EvaluationExecutionReadPort,
    strategy_id: str,
) -> TradingStrategySpec:
    strategy_state = store.get_trading_strategy(strategy_id)
    if strategy_state is None:
        raise ValueError(f"evaluation strategy does not exist: {strategy_id}")
    return strategy_state.trading_strategy


def _subject_set_id_for_strategy(trading_strategy: TradingStrategySpec) -> str:
    subject_set_id = trading_strategy.subject_set_id
    if not isinstance(subject_set_id, str) or not subject_set_id:
        raise ValueError(
            "evaluation strategy is missing subject_set: "
            f"{trading_strategy.strategy_id}"
        )
    return subject_set_id


def _target_id_for_strategy(trading_strategy: TradingStrategySpec) -> str:
    target_id = trading_strategy.target_id
    if not isinstance(target_id, str) or not target_id:
        raise ValueError(
            "evaluation strategy is missing prediction target: "
            f"{trading_strategy.strategy_id}"
        )
    return target_id


def run_strategy_evaluation(
    *,
    strategy_id: str,
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
    metric_group_names: tuple[str, ...],
    base_url: str,
    context: EvaluationExecutionContext,
) -> EvaluationResult:
    store = context.store
    trading_strategy = _trading_strategy_for_id(store, strategy_id)
    portfolio_construction = _portfolio_construction_for_strategy(trading_strategy)
    portfolio = trading_strategy.portfolio
    trading_environment = _trading_environment_for_strategy(trading_strategy)
    subject_set_id = _subject_set_id_for_strategy(trading_strategy)
    target_id = _target_id_for_strategy(trading_strategy)
    subject_set_state = store.get_subject_set(subject_set_id)
    if subject_set_state is not None:
        validate_subject_set_universe_contract(subject_set_state.definition)
    direct_evaluation = run_strategy_backtest_from_store(
        store=store,
        strategy_id=strategy_id,
        subject_set_id=subject_set_id,
        target_id=target_id,
        evaluation_date_ranges=evaluation_date_ranges,
        base_url=base_url,
        portfolio_construction=portfolio_construction,
        no_trade_band=0.0 if portfolio.no_trade_band is None else portfolio.no_trade_band,
        turnover_budget=portfolio.turnover_budget,
        trading_environment=trading_environment,
        feature_plane_repository=context.feature_plane_repository,
    )
    direct_metric_group_results, direct_failure_finding_groups = direct_evaluation
    return EvaluationResult(
        strategy_id=strategy_id,
        metric_group_results=tuple(
            direct_metric_group_results[metric_group_name]
            for metric_group_name in metric_group_names
            if metric_group_name in direct_metric_group_results
        ),
        failure_finding_groups=direct_failure_finding_groups,
    )
