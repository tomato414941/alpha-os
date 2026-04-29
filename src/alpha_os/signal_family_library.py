from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalFamilyBlueprint:
    family_id: str
    family_group: str
    primary_observable_id: str
    secondary_observable_ids: tuple[str, ...] = ()
    conditioning_observable_ids: tuple[str, ...] = ()
    applicable_subject_kinds: tuple[str, ...] = ()
    thesis: str | None = None


_BLUEPRINTS = {
    "trend_family": SignalFamilyBlueprint(
        family_id="trend_family",
        family_group="price",
        primary_observable_id="daily_close",
        applicable_subject_kinds=("asset", "equity", "etf", "index"),
        thesis="Persistent directional moves may continue over medium horizons.",
    ),
    "time_series_trend_family": SignalFamilyBlueprint(
        family_id="time_series_trend_family",
        family_group="trend",
        primary_observable_id="daily_close",
        applicable_subject_kinds=("asset", "equity", "etf", "index", "future", "perp", "crypto"),
        thesis="Directional persistence can generalize beyond cash equity universes.",
    ),
    "term_structure_carry_family": SignalFamilyBlueprint(
        family_id="term_structure_carry_family",
        family_group="carry",
        primary_observable_id="term_structure_slope",
        applicable_subject_kinds=("future", "perp"),
        thesis="Futures curve slope can encode carry and inventory pressure.",
    ),
    "funding_carry_family": SignalFamilyBlueprint(
        family_id="funding_carry_family",
        family_group="carry",
        primary_observable_id="funding_rate",
        applicable_subject_kinds=("perp", "crypto", "asset"),
        thesis="Funding payments can act as a carry signal in perpetual swaps.",
    ),
    "basis_carry_family": SignalFamilyBlueprint(
        family_id="basis_carry_family",
        family_group="carry",
        primary_observable_id="basis",
        applicable_subject_kinds=("future", "perp", "crypto", "asset"),
        thesis="Basis dislocations can indicate carry and crowding opportunity.",
    ),
    "value_anchor_family": SignalFamilyBlueprint(
        family_id="value_anchor_family",
        family_group="fundamental",
        primary_observable_id="valuation_ratio",
        applicable_subject_kinds=("equity", "etf", "asset"),
        thesis="Relative valuation extremes can anchor medium-horizon reversion trades.",
    ),
    "reversal_family": SignalFamilyBlueprint(
        family_id="reversal_family",
        family_group="price",
        primary_observable_id="daily_close",
        applicable_subject_kinds=("asset", "equity", "etf", "index"),
        thesis="Short-term dislocations may mean-revert after overshoots.",
    ),
    "vol_compression_breakout_family": SignalFamilyBlueprint(
        family_id="vol_compression_breakout_family",
        family_group="volatility",
        primary_observable_id="daily_close",
        conditioning_observable_ids=("realized_vol_20d",),
        applicable_subject_kinds=("asset", "equity", "etf", "index"),
        thesis="Low-volatility compression can precede directional breakouts.",
    ),
    "vol_expansion_reversal_family": SignalFamilyBlueprint(
        family_id="vol_expansion_reversal_family",
        family_group="volatility",
        primary_observable_id="daily_close",
        conditioning_observable_ids=("realized_vol_20d",),
        applicable_subject_kinds=("asset", "equity", "etf", "index"),
        thesis="Volatility shocks can create temporary overshoots that mean-revert.",
    ),
    "relative_strength_rank_family": SignalFamilyBlueprint(
        family_id="relative_strength_rank_family",
        family_group="cross_sectional",
        primary_observable_id="cross_sectional_return_rank_20d",
        applicable_subject_kinds=("equity", "etf", "asset"),
        thesis="Relative winners and losers within a universe can carry predictive information.",
    ),
    "peer_mean_reversion_family": SignalFamilyBlueprint(
        family_id="peer_mean_reversion_family",
        family_group="cross_sectional",
        primary_observable_id="cross_sectional_return_rank_20d",
        secondary_observable_ids=("daily_return",),
        applicable_subject_kinds=("equity", "etf", "asset"),
        thesis="Names that underperform peers may revert toward group averages.",
    ),
    "momentum_in_low_vol_regime_family": SignalFamilyBlueprint(
        family_id="momentum_in_low_vol_regime_family",
        family_group="regime_conditioned",
        primary_observable_id="daily_close",
        conditioning_observable_ids=("market_vol_regime_20d",),
        applicable_subject_kinds=("equity", "etf", "asset", "index"),
        thesis="Momentum tends to be more reliable during low-volatility market regimes.",
    ),
    "reversal_after_shock_family": SignalFamilyBlueprint(
        family_id="reversal_after_shock_family",
        family_group="regime_conditioned",
        primary_observable_id="daily_close",
        conditioning_observable_ids=("realized_vol_20d",),
        applicable_subject_kinds=("asset", "equity", "etf", "index"),
        thesis="Large return shocks followed by elevated volatility can set up reversals.",
    ),
    "trend_with_volume_confirmation_family": SignalFamilyBlueprint(
        family_id="trend_with_volume_confirmation_family",
        family_group="interaction",
        primary_observable_id="daily_close",
        secondary_observable_ids=("dollar_volume_20d",),
        applicable_subject_kinds=("asset", "equity", "etf"),
        thesis="Trend signals are stronger when they are confirmed by liquidity expansion.",
    ),
}


def list_signal_family_blueprints() -> list[SignalFamilyBlueprint]:
    return [blueprint for _, blueprint in sorted(_BLUEPRINTS.items())]


def find_signal_family_blueprint(
    family_id: str,
) -> SignalFamilyBlueprint | None:
    return _BLUEPRINTS.get(family_id)
