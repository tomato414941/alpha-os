from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeriousTemplate:
    template_id: str
    family: str
    name_template: str
    expression_template: str


@dataclass(frozen=True)
class SeriousTemplateBinding:
    asset: str
    template_id: str
    hypothesis_id: str
    bindings: dict[str, str]
    oos_sharpe: float
    oos_log_growth: float


@dataclass(frozen=True)
class SeriousSeedSpec:
    hypothesis_id: str
    name: str
    expression: str
    family: str
    oos_sharpe: float
    oos_log_growth: float
    template_id: str


_TEMPLATES: dict[str, SeriousTemplate] = {
    "onchain_activity_acceleration": SeriousTemplate(
        template_id="onchain_activity_acceleration",
        family="onchain",
        name_template="{asset_label} Onchain Activity Acceleration V1",
        expression_template="(rank_20 delta_1__{active_addresses})",
    ),
    "onchain_difficulty_regime": SeriousTemplate(
        template_id="onchain_difficulty_regime",
        family="onchain",
        name_template="{asset_label} Onchain Difficulty Regime V1",
        expression_template="(rank_20 (sign {difficulty}))",
    ),
    "onchain_difficulty_momentum": SeriousTemplate(
        template_id="onchain_difficulty_momentum",
        family="onchain",
        name_template="{asset_label} Onchain Difficulty Momentum V1",
        expression_template="(rank_5 (roc_5 {difficulty}))",
    ),
    "derivatives_open_interest_trend": SeriousTemplate(
        template_id="derivatives_open_interest_trend",
        family="derivatives",
        name_template="{asset_label} Derivatives Open Interest Trend V1",
        expression_template="(mean_20 (rank_10 {open_interest}))",
    ),
    "derivatives_funding_crowding": SeriousTemplate(
        template_id="derivatives_funding_crowding",
        family="derivatives",
        name_template="{asset_label} Derivatives Funding Crowding V1",
        expression_template="(sub {funding_rate} (mean_5 {funding_rate}))",
    ),
    "macro_sentiment_acceleration": SeriousTemplate(
        template_id="macro_sentiment_acceleration",
        family="macro",
        name_template="{asset_label} Macro Sentiment Acceleration V1",
        expression_template="(rank_20 delta_1__fear_greed)",
    ),
    "macro_dollar_pressure": SeriousTemplate(
        template_id="macro_dollar_pressure",
        family="macro",
        name_template="{asset_label} Macro Dollar Pressure V1",
        expression_template="(rank_20 roc_5__dxy)",
    ),
    "price_regime_shift": SeriousTemplate(
        template_id="price_regime_shift",
        family="price",
        name_template="{asset_label} Price Regime Shift V1",
        expression_template="(rank_20 zscore_20__{price_signal})",
    ),
    "price_short_term_impulse": SeriousTemplate(
        template_id="price_short_term_impulse",
        family="price",
        name_template="{asset_label} Price Short-Term Impulse V1",
        expression_template="(rank_20 delta_1__{price_signal})",
    ),
}


_BINDINGS: dict[str, list[SeriousTemplateBinding]] = {
    "BTC": [
        SeriousTemplateBinding(
            asset="BTC",
            template_id="onchain_activity_acceleration",
            hypothesis_id="serious_onchain_activity_acceleration_v1",
            bindings={"asset_label": "", "active_addresses": "btc_active_addresses"},
            oos_sharpe=0.72,
            oos_log_growth=0.13,
        ),
        SeriousTemplateBinding(
            asset="BTC",
            template_id="onchain_difficulty_regime",
            hypothesis_id="serious_onchain_difficulty_regime_v1",
            bindings={"asset_label": "", "difficulty": "btc_difficulty"},
            oos_sharpe=0.71,
            oos_log_growth=0.12,
        ),
        SeriousTemplateBinding(
            asset="BTC",
            template_id="onchain_difficulty_momentum",
            hypothesis_id="serious_onchain_difficulty_momentum_v1",
            bindings={"asset_label": "", "difficulty": "btc_difficulty"},
            oos_sharpe=0.68,
            oos_log_growth=0.11,
        ),
        SeriousTemplateBinding(
            asset="BTC",
            template_id="derivatives_open_interest_trend",
            hypothesis_id="serious_derivatives_open_interest_trend_v1",
            bindings={"asset_label": "", "open_interest": "oi_btc_1h"},
            oos_sharpe=0.72,
            oos_log_growth=0.13,
        ),
        SeriousTemplateBinding(
            asset="BTC",
            template_id="derivatives_funding_crowding",
            hypothesis_id="serious_derivatives_funding_crowding_v1",
            bindings={"asset_label": "", "funding_rate": "funding_rate_btc"},
            oos_sharpe=0.66,
            oos_log_growth=0.11,
        ),
        SeriousTemplateBinding(
            asset="BTC",
            template_id="macro_sentiment_acceleration",
            hypothesis_id="serious_macro_sentiment_acceleration_v1",
            bindings={"asset_label": ""},
            oos_sharpe=0.64,
            oos_log_growth=0.10,
        ),
        SeriousTemplateBinding(
            asset="BTC",
            template_id="macro_dollar_pressure",
            hypothesis_id="serious_macro_dollar_pressure_v1",
            bindings={"asset_label": ""},
            oos_sharpe=0.63,
            oos_log_growth=0.10,
        ),
        SeriousTemplateBinding(
            asset="BTC",
            template_id="price_regime_shift",
            hypothesis_id="serious_price_regime_shift_v1",
            bindings={"asset_label": "", "price_signal": "btc_ohlcv"},
            oos_sharpe=0.61,
            oos_log_growth=0.09,
        ),
        SeriousTemplateBinding(
            asset="BTC",
            template_id="price_short_term_impulse",
            hypothesis_id="serious_price_short_term_impulse_v1",
            bindings={"asset_label": "", "price_signal": "btc_ohlcv"},
            oos_sharpe=0.60,
            oos_log_growth=0.09,
        ),
    ],
    "ETH": [
        SeriousTemplateBinding(
            asset="ETH",
            template_id="derivatives_open_interest_trend",
            hypothesis_id="serious_eth_derivatives_open_interest_trend_v1",
            bindings={"asset_label": "ETH", "open_interest": "oi_eth_1h"},
            oos_sharpe=0.68,
            oos_log_growth=0.11,
        ),
        SeriousTemplateBinding(
            asset="ETH",
            template_id="derivatives_funding_crowding",
            hypothesis_id="serious_eth_derivatives_funding_crowding_v1",
            bindings={"asset_label": "ETH", "funding_rate": "funding_rate_eth"},
            oos_sharpe=0.64,
            oos_log_growth=0.10,
        ),
        SeriousTemplateBinding(
            asset="ETH",
            template_id="macro_sentiment_acceleration",
            hypothesis_id="serious_eth_macro_sentiment_acceleration_v1",
            bindings={"asset_label": "ETH"},
            oos_sharpe=0.61,
            oos_log_growth=0.09,
        ),
        SeriousTemplateBinding(
            asset="ETH",
            template_id="macro_dollar_pressure",
            hypothesis_id="serious_eth_macro_dollar_pressure_v1",
            bindings={"asset_label": "ETH"},
            oos_sharpe=0.60,
            oos_log_growth=0.09,
        ),
        SeriousTemplateBinding(
            asset="ETH",
            template_id="price_regime_shift",
            hypothesis_id="serious_eth_price_regime_shift_v1",
            bindings={"asset_label": "ETH", "price_signal": "eth_btc"},
            oos_sharpe=0.59,
            oos_log_growth=0.08,
        ),
        SeriousTemplateBinding(
            asset="ETH",
            template_id="price_short_term_impulse",
            hypothesis_id="serious_eth_price_short_term_impulse_v1",
            bindings={"asset_label": "ETH", "price_signal": "eth_btc"},
            oos_sharpe=0.58,
            oos_log_growth=0.08,
        ),
    ],
}


def serious_seed_specs(asset: str) -> list[SeriousSeedSpec]:
    asset = str(asset).upper()
    specs: list[SeriousSeedSpec] = []
    for binding in _BINDINGS.get(asset, []):
        template = _TEMPLATES[binding.template_id]
        render_args = dict(binding.bindings)
        render_args.setdefault("asset_label", asset)
        specs.append(
            SeriousSeedSpec(
                hypothesis_id=binding.hypothesis_id,
                name=template.name_template.format(**render_args).strip(),
                expression=template.expression_template.format(**render_args),
                family=template.family,
                oos_sharpe=binding.oos_sharpe,
                oos_log_growth=binding.oos_log_growth,
                template_id=template.template_id,
            )
        )
    return specs
