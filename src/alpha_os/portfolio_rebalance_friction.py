from __future__ import annotations

from dataclasses import dataclass

from .portfolio_decision import PortfolioTarget, SizingRequest


@dataclass(frozen=True)
class PortfolioRebalanceFrictionPolicy:
    signal_horizon_shortfall_aversion: float = 1.0
    turnover_cost_aversion: float = 1.0
    recent_turnover_aversion: float = 1.0


def apply_portfolio_rebalance_friction(
    request: SizingRequest,
    targets: list[PortfolioTarget],
    rebalance_friction_policy: PortfolioRebalanceFrictionPolicy | None = None,
) -> list[PortfolioTarget]:
    rebalance_friction_policy = (
        rebalance_friction_policy or PortfolioRebalanceFrictionPolicy()
    )
    adjusted_targets: list[PortfolioTarget] = []
    for index, target in enumerate(targets):
        current_weight = request.current_weights[index]
        target_weight = target.target_weight
        no_trade_band = request.no_trade_bands[index]
        delta = target_weight - current_weight
        if abs(delta) <= no_trade_band:
            target_weight = current_weight
        else:
            transition_threshold = 0.0
            if abs(current_weight) > 0.0:
                transition_threshold = (
                    max(request.turnover_cost_rate, 0.0)
                    * rebalance_friction_policy.turnover_cost_aversion
                )
            delta = _soft_threshold(delta, transition_threshold)
            if abs(current_weight) > 0.0:
                delta *= _holding_period_shrink(
                    holding_period_days=request.holding_period_days,
                    signal_horizon=request.signal_horizons[index],
                    aversion=(
                        rebalance_friction_policy.signal_horizon_shortfall_aversion
                    ),
                )
            delta *= _shrink_from_level(
                request.recent_turnover,
                rebalance_friction_policy.recent_turnover_aversion,
            )
            delta *= _shrink_from_level(
                request.market_impact_levels[index] + max(request.turnover_cost_rate, 0.0),
                rebalance_friction_policy.turnover_cost_aversion,
            )
            target_weight = current_weight + delta
        adjusted_targets.append(
            PortfolioTarget(
                subject_id=target.subject_id,
                target_weight=float(target_weight),
                position_delta=float(target_weight - current_weight),
                target_notional=(
                    None
                    if target.target_notional is None
                    else float(
                        target.target_notional
                        * (0.0 if target.target_weight == 0.0 else target_weight / target.target_weight)
                    )
                ),
                target_quantity=target.target_quantity,
                entry_allowed=not (
                    abs(current_weight) == 0.0 and abs(target_weight) == 0.0
                ),
                risk_scale=target.risk_scale,
            )
        )
    return adjusted_targets


def _holding_period_shrink(
    *,
    holding_period_days: int,
    signal_horizon: int | None,
    aversion: float,
) -> float:
    if aversion <= 0.0 or signal_horizon is None or signal_horizon <= 0:
        return 1.0
    if holding_period_days >= signal_horizon:
        return 1.0
    shortfall_ratio = float(signal_horizon - holding_period_days) / float(signal_horizon)
    return _shrink_from_level(shortfall_ratio, aversion)


def _soft_threshold(value: float, threshold: float) -> float:
    if threshold <= 0.0:
        return float(value)
    if value > threshold:
        return float(value - threshold)
    if value < -threshold:
        return float(value + threshold)
    return 0.0


def _shrink_from_level(level: float, aversion: float) -> float:
    if level <= 0.0 or aversion <= 0.0:
        return 1.0
    return float(1.0 / (1.0 + aversion * level))
