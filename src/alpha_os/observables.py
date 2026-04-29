from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObservableDefinition:
    observable_id: str
    family: str
    value_kind: str
    default_resolution: str = "1d"
    params: dict[str, Any] | None = None
    description: str | None = None
    input_observable_ids: tuple[str, ...] = ()
    applicable_subject_kinds: tuple[str, ...] = ()

    def to_document(self) -> dict[str, Any]:
        document = {
            "observable_id": self.observable_id,
            "family": self.family,
            "value_kind": self.value_kind,
            "default_resolution": self.default_resolution,
            "params": {} if self.params is None else dict(self.params),
        }
        if self.description is not None:
            document["description"] = self.description
        if self.input_observable_ids:
            document["input_observable_ids"] = list(self.input_observable_ids)
        if self.applicable_subject_kinds:
            document["applicable_subject_kinds"] = list(self.applicable_subject_kinds)
        return document

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "ObservableDefinition":
        observable_id = document.get("observable_id")
        family = document.get("family")
        value_kind = document.get("value_kind")
        default_resolution = document.get("default_resolution", "1d")
        params = document.get("params", {})
        description = document.get("description")
        input_observable_ids = document.get("input_observable_ids", [])
        applicable_subject_kinds = document.get("applicable_subject_kinds", [])
        if not isinstance(observable_id, str) or not observable_id:
            raise ValueError("observable document is missing observable_id")
        if not isinstance(family, str) or not family:
            raise ValueError("observable document is missing family")
        if not isinstance(value_kind, str) or not value_kind:
            raise ValueError("observable document is missing value_kind")
        if not isinstance(default_resolution, str) or not default_resolution:
            raise ValueError("observable document is missing default_resolution")
        if not isinstance(params, dict):
            raise ValueError("observable document params must be a dict")
        if description is not None and not isinstance(description, str):
            raise ValueError("observable document description must be a string")
        if not isinstance(input_observable_ids, list) or any(
            not isinstance(item, str) or not item
            for item in input_observable_ids
        ):
            raise ValueError(
                "observable document input_observable_ids must be a list of strings"
            )
        if not isinstance(applicable_subject_kinds, list) or any(
            not isinstance(item, str) or not item
            for item in applicable_subject_kinds
        ):
            raise ValueError(
                "observable document applicable_subject_kinds must be a list of strings"
            )
        return cls(
            observable_id=observable_id,
            family=family,
            value_kind=value_kind,
            default_resolution=default_resolution,
            params=dict(params),
            description=description,
            input_observable_ids=tuple(input_observable_ids),
            applicable_subject_kinds=tuple(applicable_subject_kinds),
        )


_OBSERVABLE_DEFINITIONS = {
    "daily_close": ObservableDefinition(
        observable_id="daily_close",
        family="price",
        value_kind="real_value",
        default_resolution="1d",
        params={},
        description="Canonical daily close price for a subject.",
        applicable_subject_kinds=("asset", "equity", "etf", "index"),
    ),
    "daily_return": ObservableDefinition(
        observable_id="daily_return",
        family="return",
        value_kind="real_value",
        default_resolution="1d",
        params={"transform": "pct_change"},
        description="One-day arithmetic return derived from daily_close.",
        input_observable_ids=("daily_close",),
        applicable_subject_kinds=("asset", "equity", "etf", "index"),
    ),
    "realized_vol_20d": ObservableDefinition(
        observable_id="realized_vol_20d",
        family="volatility",
        value_kind="real_value",
        default_resolution="1d",
        params={"lookback": 20, "annualization": 252},
        description="Twenty-day realized volatility derived from daily_return.",
        input_observable_ids=("daily_return",),
        applicable_subject_kinds=("asset", "equity", "etf", "index"),
    ),
    "daily_volume": ObservableDefinition(
        observable_id="daily_volume",
        family="volume",
        value_kind="real_value",
        default_resolution="1d",
        params={},
        description="Canonical daily traded volume for a subject.",
        applicable_subject_kinds=("asset", "equity", "etf"),
    ),
    "dollar_volume_20d": ObservableDefinition(
        observable_id="dollar_volume_20d",
        family="liquidity",
        value_kind="real_value",
        default_resolution="1d",
        params={"lookback": 20},
        description="Twenty-day average dollar volume derived from price and volume.",
        input_observable_ids=("daily_close", "daily_volume"),
        applicable_subject_kinds=("asset", "equity", "etf"),
    ),
    "cross_sectional_return_rank_20d": ObservableDefinition(
        observable_id="cross_sectional_return_rank_20d",
        family="cross_sectional",
        value_kind="rank_value",
        default_resolution="1d",
        params={"lookback": 20, "scope": "subject_set"},
        description="Cross-sectional rank of medium-horizon return within a subject set.",
        input_observable_ids=("daily_return",),
        applicable_subject_kinds=("equity", "etf", "asset", "index"),
    ),
    "market_return_20d": ObservableDefinition(
        observable_id="market_return_20d",
        family="market_state",
        value_kind="real_value",
        default_resolution="1d",
        params={"lookback": 20},
        description="Market-level twenty-day return used for conditional hypotheses.",
        input_observable_ids=("daily_return",),
        applicable_subject_kinds=("equity", "etf", "asset", "index"),
    ),
    "market_vol_regime_20d": ObservableDefinition(
        observable_id="market_vol_regime_20d",
        family="regime",
        value_kind="categorical_value",
        default_resolution="1d",
        params={"lookback": 20, "levels": ("low_vol", "normal_vol", "high_vol")},
        description="Market volatility regime state for conditional hypotheses.",
        input_observable_ids=("realized_vol_20d",),
        applicable_subject_kinds=("equity", "etf", "asset", "index"),
    ),
    "front_price": ObservableDefinition(
        observable_id="front_price",
        family="price",
        value_kind="real_value",
        default_resolution="1d",
        params={},
        description="Front contract price for derivative instruments.",
        applicable_subject_kinds=("future", "perp", "asset"),
    ),
    "next_price": ObservableDefinition(
        observable_id="next_price",
        family="price",
        value_kind="real_value",
        default_resolution="1d",
        params={},
        description="Next contract price used to infer curve shape.",
        applicable_subject_kinds=("future", "perp", "asset"),
    ),
    "term_structure_slope": ObservableDefinition(
        observable_id="term_structure_slope",
        family="curve",
        value_kind="real_value",
        default_resolution="1d",
        params={"method": "next_minus_front_over_front"},
        description="Relative slope between next and front contract prices.",
        input_observable_ids=("front_price", "next_price"),
        applicable_subject_kinds=("future", "perp"),
    ),
    "funding_rate": ObservableDefinition(
        observable_id="funding_rate",
        family="carry",
        value_kind="real_value",
        default_resolution="1d",
        params={},
        description="Periodic funding transfer rate for perpetual futures.",
        applicable_subject_kinds=("perp", "crypto", "asset"),
    ),
    "open_interest": ObservableDefinition(
        observable_id="open_interest",
        family="positioning",
        value_kind="real_value",
        default_resolution="1d",
        params={},
        description="Open interest level for derivatives markets.",
        applicable_subject_kinds=("future", "perp", "asset"),
    ),
    "basis": ObservableDefinition(
        observable_id="basis",
        family="basis",
        value_kind="real_value",
        default_resolution="1d",
        params={},
        description="Spot-futures basis or premium measure.",
        applicable_subject_kinds=("future", "perp", "crypto", "asset"),
    ),
    "borrow_fee": ObservableDefinition(
        observable_id="borrow_fee",
        family="financing",
        value_kind="real_value",
        default_resolution="1d",
        params={},
        description="Short borrow fee or stock loan rate.",
        applicable_subject_kinds=("equity", "etf", "asset"),
    ),
    "valuation_ratio": ObservableDefinition(
        observable_id="valuation_ratio",
        family="value",
        value_kind="real_value",
        default_resolution="1d",
        params={},
        description="Generic valuation ratio such as earnings yield or book-to-price.",
        applicable_subject_kinds=("equity", "etf", "asset"),
    ),
    "earnings_revision": ObservableDefinition(
        observable_id="earnings_revision",
        family="revision",
        value_kind="real_value",
        default_resolution="1d",
        params={},
        description="Net analyst earnings revision signal for equities.",
        applicable_subject_kinds=("equity", "asset"),
    ),
}


def find_observable_definition(observable_id: str) -> ObservableDefinition | None:
    return _OBSERVABLE_DEFINITIONS.get(observable_id)


def list_observable_definitions() -> list[ObservableDefinition]:
    return [definition for _, definition in sorted(_OBSERVABLE_DEFINITIONS.items())]
