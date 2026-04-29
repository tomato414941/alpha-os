from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SignalOperatorDefinition:
    operator_id: str
    generated_kind: str
    family_group: str
    primary_observable_families: tuple[str, ...]
    secondary_observable_families: tuple[str, ...] = ()
    conditioning_observable_families: tuple[str, ...] = ()
    requires_secondary: bool = False
    requires_conditioning: bool = False
    requires_subject_set_context: bool = False
    applicable_subject_kinds: tuple[str, ...] = ()
    thesis: str | None = None

    def to_document(self) -> dict[str, Any]:
        document = {
            "operator_id": self.operator_id,
            "generated_kind": self.generated_kind,
            "family_group": self.family_group,
            "primary_observable_families": list(self.primary_observable_families),
            "requires_secondary": self.requires_secondary,
            "requires_conditioning": self.requires_conditioning,
            "requires_subject_set_context": self.requires_subject_set_context,
        }
        if self.secondary_observable_families:
            document["secondary_observable_families"] = list(
                self.secondary_observable_families
            )
        if self.conditioning_observable_families:
            document["conditioning_observable_families"] = list(
                self.conditioning_observable_families
            )
        if self.applicable_subject_kinds:
            document["applicable_subject_kinds"] = list(
                self.applicable_subject_kinds
            )
        if self.thesis is not None:
            document["thesis"] = self.thesis
        return document


_OPERATORS = {
    "trend": SignalOperatorDefinition(
        operator_id="trend",
        generated_kind="momentum",
        family_group="price",
        primary_observable_families=("price", "return"),
        applicable_subject_kinds=("asset", "equity", "etf", "index"),
        thesis="Persistent directional moves may continue.",
    ),
    "time_series_trend": SignalOperatorDefinition(
        operator_id="time_series_trend",
        generated_kind="time_series_trend",
        family_group="trend",
        primary_observable_families=("price", "return"),
        applicable_subject_kinds=(
            "asset",
            "equity",
            "etf",
            "index",
            "future",
            "perp",
            "crypto",
        ),
        thesis="Directional persistence can survive across instrument types.",
    ),
    "term_structure_carry": SignalOperatorDefinition(
        operator_id="term_structure_carry",
        generated_kind="term_structure_carry",
        family_group="carry",
        primary_observable_families=("curve",),
        applicable_subject_kinds=("future", "perp"),
        thesis="Curve slope can proxy for carry in derivative markets.",
    ),
    "funding_carry": SignalOperatorDefinition(
        operator_id="funding_carry",
        generated_kind="funding_carry",
        family_group="carry",
        primary_observable_families=("carry",),
        applicable_subject_kinds=("perp", "crypto", "asset"),
        thesis="Persistent funding imbalances can signal carry opportunities.",
    ),
    "basis_carry": SignalOperatorDefinition(
        operator_id="basis_carry",
        generated_kind="basis_carry",
        family_group="carry",
        primary_observable_families=("basis",),
        applicable_subject_kinds=("future", "perp", "crypto", "asset"),
        thesis="Spot-futures basis can proxy for carry and positioning pressure.",
    ),
    "value_anchor": SignalOperatorDefinition(
        operator_id="value_anchor",
        generated_kind="value_anchor",
        family_group="fundamental",
        primary_observable_families=("value",),
        applicable_subject_kinds=("equity", "etf", "asset"),
        thesis="Assets that look rich relative to their own history can mean-revert.",
    ),
    "mean_reversion": SignalOperatorDefinition(
        operator_id="mean_reversion",
        generated_kind="reversal",
        family_group="price",
        primary_observable_families=("price", "return"),
        applicable_subject_kinds=("asset", "equity", "etf", "index"),
        thesis="Short-term dislocations may mean-revert.",
    ),
    "volatility_breakout": SignalOperatorDefinition(
        operator_id="volatility_breakout",
        generated_kind="vol_compression_breakout",
        family_group="volatility",
        primary_observable_families=("price",),
        conditioning_observable_families=("volatility",),
        requires_conditioning=True,
        applicable_subject_kinds=("asset", "equity", "etf", "index"),
        thesis="Low-volatility compression may precede breakouts.",
    ),
    "volatility_reversal": SignalOperatorDefinition(
        operator_id="volatility_reversal",
        generated_kind="vol_expansion_reversal",
        family_group="volatility",
        primary_observable_families=("price",),
        conditioning_observable_families=("volatility",),
        requires_conditioning=True,
        applicable_subject_kinds=("asset", "equity", "etf", "index"),
        thesis="Volatility expansions can create overshoots that mean-revert.",
    ),
    "low_vol_trend": SignalOperatorDefinition(
        operator_id="low_vol_trend",
        generated_kind="momentum_low_vol",
        family_group="regime_conditioned",
        primary_observable_families=("price",),
        conditioning_observable_families=("regime", "volatility"),
        requires_conditioning=True,
        applicable_subject_kinds=("asset", "equity", "etf", "index"),
        thesis="Momentum is stronger in calm regimes.",
    ),
    "post_shock_reversion": SignalOperatorDefinition(
        operator_id="post_shock_reversion",
        generated_kind="reversal_after_shock",
        family_group="regime_conditioned",
        primary_observable_families=("price", "return"),
        conditioning_observable_families=("volatility",),
        requires_conditioning=True,
        applicable_subject_kinds=("asset", "equity", "etf", "index"),
        thesis="Large shocks in volatile regimes can reverse.",
    ),
    "volume_confirmed_trend": SignalOperatorDefinition(
        operator_id="volume_confirmed_trend",
        generated_kind="trend_volume_confirmation",
        family_group="interaction",
        primary_observable_families=("price",),
        secondary_observable_families=("liquidity", "volume"),
        requires_secondary=True,
        applicable_subject_kinds=("asset", "equity", "etf"),
        thesis="Trend is stronger when liquidity expands.",
    ),
    "relative_strength": SignalOperatorDefinition(
        operator_id="relative_strength",
        generated_kind="relative_strength_rank",
        family_group="cross_sectional",
        primary_observable_families=("cross_sectional",),
        requires_subject_set_context=True,
        applicable_subject_kinds=("asset", "equity", "etf", "index"),
        thesis="Relative winners and losers within a universe can carry information.",
    ),
    "peer_reversion": SignalOperatorDefinition(
        operator_id="peer_reversion",
        generated_kind="peer_mean_reversion",
        family_group="cross_sectional",
        primary_observable_families=("cross_sectional",),
        requires_subject_set_context=True,
        applicable_subject_kinds=("asset", "equity", "etf", "index"),
        thesis="Relative losers can mean-revert toward peer averages.",
    ),
}


def find_signal_operator_definition(
    operator_id: str,
) -> SignalOperatorDefinition | None:
    return _OPERATORS.get(operator_id)


def list_signal_operator_definitions() -> list[SignalOperatorDefinition]:
    return [definition for _, definition in sorted(_OPERATORS.items())]
