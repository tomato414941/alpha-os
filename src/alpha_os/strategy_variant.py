from __future__ import annotations

from dataclasses import dataclass

from .evaluation_cost_config import TradingEnvironment
from .portfolio_construction_config import (
    PortfolioConstructionSizingSpec,
    PortfolioConstructionSpec,
)
from .trading_strategy import (
    StrategyPortfolioSpec,
    TradingStrategyScopeSpec,
    TradingStrategySpec,
    build_trading_strategy_id,
)


@dataclass(frozen=True)
class StrategyVariantConfig:
    portfolio_construction: PortfolioConstructionSpec
    trading_environment: TradingEnvironment
    top_k: int | None = None

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
    return StrategyVariantConfig(
        portfolio_construction=construction,
        trading_environment=portfolio.trading_environment,
        top_k=portfolio.top_k,
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
        ),
        trading_environment=config.trading_environment,
        top_k=config.top_k,
    )


def derive_trading_strategy_from_signal_discovery(
    *,
    signal_discovery,
    variant_config: StrategyVariantConfig,
    created_at: str,
) -> TradingStrategySpec:
    definition = signal_discovery.definition
    portfolio_construction = variant_config.portfolio_construction
    trading_environment = variant_config.trading_environment
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
            trading_environment=trading_environment,
            selection_kind="all_assets" if top_k_value is None else "top_k",
            top_k=top_k_value,
            rebalance_interval_steps=portfolio_construction.rebalance_interval_steps,
        ),
        created_at=created_at,
    )
