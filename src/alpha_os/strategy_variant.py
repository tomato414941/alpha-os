from __future__ import annotations

from dataclasses import dataclass

from .evaluation_cost_config import (
    EvaluationRebalanceFrictionPolicySpec,
    ExecutionCostAssumptionsSpec,
    HoldingCostAssumptionsSpec,
)
from .portfolio_construction_config import (
    PortfolioConstructionSizingSpec,
    PortfolioConstructionSpec,
)
from .trading_strategy import (
    AdaptationPolicySpec,
    ExecutionPolicySpec,
    HoldingCostPolicySpec,
    RebalanceFrictionPolicySpec,
    StrategyPortfolioSpec,
    TradingStrategyScopeSpec,
    TradingStrategySpec,
    build_trading_strategy_id,
)


@dataclass(frozen=True)
class StrategyVariantConfig:
    portfolio_construction: PortfolioConstructionSpec
    rebalance_friction_policy: EvaluationRebalanceFrictionPolicySpec
    execution_cost_assumptions: ExecutionCostAssumptionsSpec
    top_k: int | None = None
    holding_cost_assumptions: HoldingCostAssumptionsSpec = HoldingCostAssumptionsSpec()

    @property
    def sizing_method(self) -> str:
        return self.portfolio_construction.sizing_method

    @property
    def sizing_engine(self) -> str:
        return self.portfolio_construction.sizing_engine


def strategy_variant_config_from_strategy(
    trading_strategy: TradingStrategySpec,
) -> StrategyVariantConfig:
    portfolio = trading_strategy.portfolio
    construction = portfolio.portfolio_construction
    friction = portfolio.rebalance_friction_policy
    execution = portfolio.execution_policy
    holding = portfolio.holding_cost_policy
    return StrategyVariantConfig(
        portfolio_construction=construction,
        rebalance_friction_policy=EvaluationRebalanceFrictionPolicySpec.from_document(
            {
                key: value
                for key, value in friction.to_document().items()
                if value is not None
            }
        ),
        execution_cost_assumptions=ExecutionCostAssumptionsSpec(
            market_impact_bps=execution.market_impact_bps or 0.0,
            fee_bps=execution.fee_bps or 0.0,
            bid_ask_spread_bps=execution.bid_ask_spread_bps or 0.0,
        ),
        top_k=portfolio.top_k,
        holding_cost_assumptions=HoldingCostAssumptionsSpec(
            funding_bps_per_step=(
                0.0
                if holding.funding_bps_per_step is None
                else holding.funding_bps_per_step
            ),
            borrow_fee_bps_per_step=(
                0.0
                if holding.borrow_fee_bps_per_step is None
                else holding.borrow_fee_bps_per_step
            ),
        ),
    )


def overridden_strategy_variant_config(
    config: StrategyVariantConfig,
    *,
    sizing_method: str | None,
    sizing_engine: str | None,
    direction_mode: str | None = None,
) -> StrategyVariantConfig:
    if sizing_method is None and sizing_engine is None and direction_mode is None:
        return config
    portfolio_construction = config.portfolio_construction
    resolved_sizing_method = (
        portfolio_construction.sizing_method
        if sizing_method is None
        else str(sizing_method)
    )
    if sizing_engine is None and sizing_method is not None:
        resolved_sizing_engine = PortfolioConstructionSizingSpec(
            sizing_method=resolved_sizing_method,
        ).sizing_engine
    else:
        resolved_sizing_engine = (
            portfolio_construction.sizing_engine
            if sizing_engine is None
            else str(sizing_engine)
        )
    return StrategyVariantConfig(
        portfolio_construction=PortfolioConstructionSpec(
            construction_kind=portfolio_construction.construction_kind,
            sizing_policy=PortfolioConstructionSizingSpec(
                sizing_method=resolved_sizing_method,
                sizing_engine=resolved_sizing_engine,
            ),
            rebalance_interval_steps=portfolio_construction.rebalance_interval_steps,
            long_only=portfolio_construction.long_only,
            direction_mode=(
                portfolio_construction.direction_mode
                if direction_mode is None
                else direction_mode
            ),
            active_overlay=portfolio_construction.active_overlay,
            gross_exposure_cap=portfolio_construction.gross_exposure_cap,
            target_vol=portfolio_construction.target_vol,
            gross_leverage_cap=portfolio_construction.gross_leverage_cap,
            net_exposure_target=portfolio_construction.net_exposure_target,
            asset_class_weight_caps=dict(portfolio_construction.asset_class_weight_caps),
            cluster_weight_caps=dict(portfolio_construction.cluster_weight_caps),
            sleeve_composition=portfolio_construction.sleeve_composition,
        ),
        rebalance_friction_policy=config.rebalance_friction_policy,
        execution_cost_assumptions=config.execution_cost_assumptions,
        top_k=config.top_k,
        holding_cost_assumptions=config.holding_cost_assumptions,
    )


def derive_trading_strategy_from_signal_discovery(
    *,
    signal_discovery,
    variant_config: StrategyVariantConfig,
    created_at: str,
) -> TradingStrategySpec:
    definition = signal_discovery.definition
    portfolio_construction = variant_config.portfolio_construction
    rebalance_friction_policy = variant_config.rebalance_friction_policy
    execution_cost_assumptions = variant_config.execution_cost_assumptions
    holding_cost_assumptions = variant_config.holding_cost_assumptions
    sizing_method = portfolio_construction.sizing_method
    family_ids = tuple(
        sorted(
            getattr(item, "resolved_family_id", item.family_id)
            for item in definition.families
        )
    )
    specification_ids = tuple(sorted(definition.signal_spec_ids))
    family_mix_value = (
        ",".join(family_ids)
        if family_ids
        else ("spec:" + ",".join(specification_ids) if specification_ids else "-")
    )
    rebalance_value = f"every_{portfolio_construction.rebalance_interval_steps}_steps"
    top_k_value = None if variant_config.top_k is None else int(variant_config.top_k)
    strategy_id = build_trading_strategy_id(
        signal_discovery_id=signal_discovery.signal_discovery_id,
        subject_set_id=definition.subject_set_id,
        target_id=definition.target_id,
        family_mix=family_mix_value,
        sizing_method=sizing_method,
        sizing_engine=portfolio_construction.sizing_engine,
        rebalance=rebalance_value,
        long_only=portfolio_construction.long_only,
        direction_mode=(
            portfolio_construction.direction_mode
            if portfolio_construction.direction_mode == "short_only"
            else None
        ),
        top_k=top_k_value,
        gross_exposure_cap=(
            None
            if portfolio_construction.gross_exposure_cap is None
            else float(portfolio_construction.gross_exposure_cap)
        ),
        target_vol=(
            None
            if portfolio_construction.target_vol is None
            else float(portfolio_construction.target_vol)
        ),
        gross_leverage_cap=(
            None
            if portfolio_construction.gross_leverage_cap is None
            else float(portfolio_construction.gross_leverage_cap)
        ),
        net_exposure_target=(
            None
            if portfolio_construction.net_exposure_target is None
            else float(portfolio_construction.net_exposure_target)
        ),
        asset_class_weight_caps=dict(portfolio_construction.asset_class_weight_caps),
        cluster_weight_caps=dict(portfolio_construction.cluster_weight_caps),
        market_impact_bps=float(execution_cost_assumptions.market_impact_bps),
        fee_bps=float(execution_cost_assumptions.fee_bps),
        bid_ask_spread_bps=float(execution_cost_assumptions.bid_ask_spread_bps),
        turnover_friction=float(rebalance_friction_policy.turnover_friction),
        no_trade_band=float(rebalance_friction_policy.no_trade_band),
        funding_bps_per_step=float(holding_cost_assumptions.funding_bps_per_step),
        borrow_fee_bps_per_step=float(
            holding_cost_assumptions.borrow_fee_bps_per_step
        ),
        adaptation_enabled=True,
        adaptation_blend=0.2,
        sleeve_composition=portfolio_construction.sleeve_composition,
    )
    return TradingStrategySpec(
        strategy_id=strategy_id,
        label=(
            f"{signal_discovery.signal_discovery_id}:{sizing_method}:"
            f"{rebalance_value}"
        ),
        scope=TradingStrategyScopeSpec(
            subject_set_id=definition.subject_set_id,
            target_id=definition.target_id,
        ),
        signal_discovery_id=signal_discovery.signal_discovery_id,
        position_rule_id="constant_hold",
        family_mix=family_mix_value,
        portfolio=StrategyPortfolioSpec(
            portfolio_construction=portfolio_construction,
            rebalance_friction_policy=RebalanceFrictionPolicySpec(
                turnover_friction=float(rebalance_friction_policy.turnover_friction),
                no_trade_band=float(rebalance_friction_policy.no_trade_band),
                execution_cost_aversion=float(
                    rebalance_friction_policy.execution_cost_aversion
                ),
                execution_mode=rebalance_friction_policy.execution_mode,
                turnover_budget=rebalance_friction_policy.turnover_budget,
                benefit_scale=rebalance_friction_policy.benefit_scale,
                min_trade_utility=rebalance_friction_policy.min_trade_utility,
                uncertainty_aversion=rebalance_friction_policy.uncertainty_aversion,
                risk_aversion=rebalance_friction_policy.risk_aversion,
                partial_fill_enabled=rebalance_friction_policy.partial_fill_enabled,
            ),
            execution_policy=ExecutionPolicySpec(
                market_impact_bps=float(execution_cost_assumptions.market_impact_bps),
                fee_bps=float(execution_cost_assumptions.fee_bps),
                bid_ask_spread_bps=float(execution_cost_assumptions.bid_ask_spread_bps),
            ),
            holding_cost_policy=HoldingCostPolicySpec(
                funding_bps_per_step=float(
                    holding_cost_assumptions.funding_bps_per_step
                ),
                borrow_fee_bps_per_step=float(
                    holding_cost_assumptions.borrow_fee_bps_per_step
                ),
            ),
            selection_kind="all_assets" if top_k_value is None else "top_k",
            top_k=top_k_value,
            rebalance_interval_steps=portfolio_construction.rebalance_interval_steps,
        ),
        created_at=created_at,
        adaptation_policy=AdaptationPolicySpec(
            enabled=True,
            adaptation_blend=0.2,
        ),
    )
