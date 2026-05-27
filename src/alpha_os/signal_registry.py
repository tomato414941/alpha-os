from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .observables import find_observable_definition
from .portfolio_decision import ObservationSpec
from .targets import (
    TargetDefinition,
    get_target_definition,
    residual_return_target_definition,
)


DEFAULT_SUBJECT_ID = "BTC"
DEFAULT_HORIZON_DAYS = 3
DEFAULT_TARGET_DEFINITION = residual_return_target_definition(DEFAULT_HORIZON_DAYS)


def default_runtime_asset(subject_id: str | None = None) -> str:
    del subject_id
    return DEFAULT_SUBJECT_ID


@dataclass(frozen=True, init=False)
class SignalSpec:
    signal_id: str
    kind: str
    lookback: int
    required_observable_id: str = "daily_close"
    target: TargetDefinition = DEFAULT_TARGET_DEFINITION

    def __init__(
        self,
        *,
        signal_id: str | None = None,

        kind: str,
        lookback: int,
        required_observable_id: str = "daily_close",
        target: TargetDefinition = DEFAULT_TARGET_DEFINITION,
    ) -> None:
        resolved_signal_id = signal_id
        if resolved_signal_id is None:
            raise ValueError("signal spec requires signal_id")
        object.__setattr__(self, "signal_id", str(resolved_signal_id))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "lookback", lookback)
        object.__setattr__(self, "required_observable_id", required_observable_id)
        object.__setattr__(self, "target", target)
    @property
    def target_id(self) -> str:
        return self.target.target_id

    @property
    def horizon_days(self) -> int | None:
        return self.target.horizon_days

    @property
    def params(self) -> dict[str, int]:
        return {
            "lookback": self.lookback,
        }

    def to_document(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "required_observable_id": self.required_observable_id,
            "target_definition": self.target.to_document(),
            "params": {
                "lookback": self.lookback,
            },
        }

    @classmethod
    def from_document(
        cls,
        *,
        signal_id: str | None = None,

        document: dict[str, Any],
    ) -> "SignalSpec":
        resolved_signal_id = signal_id
        if resolved_signal_id is None:
            raise ValueError("signal spec requires signal_id")
        kind = document.get("kind")
        target_document = document.get("target_definition")
        params = document.get("params")
        if not isinstance(kind, str) or not kind:
            raise ValueError(
                f"signal spec is missing kind: {resolved_signal_id}"
            )
        if not isinstance(target_document, dict):
            raise ValueError(
                "signal spec is missing target_definition: "
                f"{resolved_signal_id}"
            )
        if not isinstance(params, dict):
            raise ValueError(
                f"signal spec is missing params: {resolved_signal_id}"
            )
        lookback = params.get("lookback")
        if not isinstance(lookback, int):
            raise ValueError(
                "signal spec is missing integer lookback: "
                f"{resolved_signal_id}"
            )
        return cls(
            signal_id=resolved_signal_id,
            kind=kind,
            lookback=lookback,
            required_observable_id=str(
                document.get("required_observable_id", "daily_close")
            ),
            target=TargetDefinition.from_document(target_document),
        )


@dataclass(frozen=True, init=False)
class SignalDefinition:
    signal_id: str
    kind: str
    lookback: int
    target: TargetDefinition = DEFAULT_TARGET_DEFINITION
    asset: str = DEFAULT_SUBJECT_ID
    observation_spec: ObservationSpec | None = None

    def __init__(
        self,
        *,
        signal_id: str | None = None,

        kind: str,
        lookback: int,
        target: TargetDefinition = DEFAULT_TARGET_DEFINITION,
        asset: str = DEFAULT_SUBJECT_ID,
        observation_spec: ObservationSpec | None = None,
    ) -> None:
        resolved_signal_id = signal_id
        if resolved_signal_id is None:
            raise ValueError("signal definition requires signal_id")
        object.__setattr__(self, "signal_id", str(resolved_signal_id))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "lookback", lookback)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "observation_spec", observation_spec)
    @property
    def target_id(self) -> str:
        return self.target.target_id

    @property
    def horizon_days(self) -> int | None:
        return self.target.horizon_days

    @property
    def signal_name(self) -> str | None:
        return None

    def to_document(self) -> dict[str, Any]:
        document = signal_document_from_spec(
            specification=SignalSpec(
                signal_id=self.signal_id,
                kind=self.kind,
                lookback=self.lookback,
                required_observable_id=(
                    "daily_close"
                    if self.observation_spec is None
                    else self.observation_spec.observable_id
                ),
                target=self.target,
            ),
            observation_spec=self.observation_spec,
        )
        return document

    @classmethod
    def from_document(
        cls,
        *,
        signal_id: str | None = None,

        document: dict[str, Any],
        asset: str = DEFAULT_SUBJECT_ID,
    ) -> "SignalDefinition":
        resolved_signal_id = signal_id
        if resolved_signal_id is None:
            raise ValueError("signal definition requires signal_id")
        kind = document.get("kind")
        target_document = document.get("target_definition")
        params = document.get("params")
        if not isinstance(kind, str) or not kind:
            raise ValueError(
                f"signal document is missing kind: {resolved_signal_id}"
            )
        if not isinstance(target_document, dict):
            raise ValueError(
                "signal document is missing target_definition: "
                f"{resolved_signal_id}"
            )
        if not isinstance(params, dict):
            raise ValueError(
                f"signal document is missing params: {resolved_signal_id}"
            )
        lookback = params.get("lookback")
        if not isinstance(lookback, int):
            raise ValueError(
                "signal document is missing integer lookback: "
                f"{resolved_signal_id}"
            )
        observation_spec_document = document.get("observation_spec")
        observation_spec = None
        if isinstance(observation_spec_document, dict):
            observation_spec = ObservationSpec(
                observation_spec_id=str(
                    observation_spec_document.get(
                        "observation_spec_id",
                        f"{resolved_signal_id}__observation",
                    )
                ),
                observable_id=str(
                    observation_spec_document.get("observable_id", "daily_close")
                ),
                adapter_kind=str(
                    observation_spec_document.get(
                        "adapter_kind",
                        "signal_noise_asset_observable",
                    )
                ),
                source_id=str(observation_spec_document.get("source_id", "signal_noise")),
                resolution=str(observation_spec_document.get("resolution", "1d")),
            )
        else:
            signal_name = document.get("signal_name")
            if isinstance(signal_name, str) and signal_name:
                observation_spec = asset_observable_observation_spec(
                    observation_spec_id=f"{resolved_signal_id}__legacy",
                )
        return cls(
            signal_id=resolved_signal_id,
            kind=kind,
            lookback=lookback,
            target=TargetDefinition.from_document(target_document),
            asset=asset,
            observation_spec=observation_spec,
        )


_SPECIFICATION_DEFINITIONS = {
    "momentum_1d": SignalSpec(
        signal_id="momentum_1d",
        kind="momentum",
        lookback=1,
        target=residual_return_target_definition(DEFAULT_HORIZON_DAYS),
    ),
    "momentum_3d": SignalSpec(
        signal_id="momentum_3d",
        kind="momentum",
        lookback=3,
        target=residual_return_target_definition(DEFAULT_HORIZON_DAYS),
    ),
    "momentum_5d": SignalSpec(
        signal_id="momentum_5d",
        kind="momentum",
        lookback=5,
        target=residual_return_target_definition(DEFAULT_HORIZON_DAYS),
    ),
    "reversal_1d": SignalSpec(
        signal_id="reversal_1d",
        kind="reversal",
        lookback=1,
        target=residual_return_target_definition(DEFAULT_HORIZON_DAYS),
    ),
    "reversal_3d": SignalSpec(
        signal_id="reversal_3d",
        kind="reversal",
        lookback=3,
        target=residual_return_target_definition(DEFAULT_HORIZON_DAYS),
    ),
    "reversal_5d": SignalSpec(
        signal_id="reversal_5d",
        kind="reversal",
        lookback=5,
        target=residual_return_target_definition(DEFAULT_HORIZON_DAYS),
    ),
    "average_gap_3d": SignalSpec(
        signal_id="average_gap_3d",
        kind="average_gap",
        lookback=3,
        target=residual_return_target_definition(DEFAULT_HORIZON_DAYS),
    ),
    "average_gap_5d": SignalSpec(
        signal_id="average_gap_5d",
        kind="average_gap",
        lookback=5,
        target=residual_return_target_definition(DEFAULT_HORIZON_DAYS),
    ),
    "range_position_5d": SignalSpec(
        signal_id="range_position_5d",
        kind="range_position",
        lookback=5,
        target=residual_return_target_definition(DEFAULT_HORIZON_DAYS),
    ),
    "vol_compression_breakout_20d": SignalSpec(
        signal_id="vol_compression_breakout_20d",
        kind="vol_compression_breakout",
        lookback=20,
        target=residual_return_target_definition(DEFAULT_HORIZON_DAYS),
    ),
    "vol_expansion_reversal_20d": SignalSpec(
        signal_id="vol_expansion_reversal_20d",
        kind="vol_expansion_reversal",
        lookback=20,
        target=residual_return_target_definition(DEFAULT_HORIZON_DAYS),
    ),
    "momentum_low_vol_20d": SignalSpec(
        signal_id="momentum_low_vol_20d",
        kind="momentum_low_vol",
        lookback=20,
        target=residual_return_target_definition(DEFAULT_HORIZON_DAYS),
    ),
    "reversal_after_shock_20d": SignalSpec(
        signal_id="reversal_after_shock_20d",
        kind="reversal_after_shock",
        lookback=20,
        target=residual_return_target_definition(DEFAULT_HORIZON_DAYS),
    ),
    "trend_volume_confirmation_20d": SignalSpec(
        signal_id="trend_volume_confirmation_20d",
        kind="trend_volume_confirmation",
        lookback=20,
        target=residual_return_target_definition(DEFAULT_HORIZON_DAYS),
    ),
    "relative_strength_rank_20d": SignalSpec(
        signal_id="relative_strength_rank_20d",
        kind="relative_strength_rank",
        lookback=20,
        required_observable_id="cross_sectional_return_rank_20d",
        target=residual_return_target_definition(DEFAULT_HORIZON_DAYS),
    ),
    "peer_mean_reversion_20d": SignalSpec(
        signal_id="peer_mean_reversion_20d",
        kind="peer_mean_reversion",
        lookback=20,
        required_observable_id="cross_sectional_return_rank_20d",
        target=residual_return_target_definition(DEFAULT_HORIZON_DAYS),
    ),
}


def signal_document_from_spec(
    *,
    specification: SignalSpec,
    observation_spec: ObservationSpec | None,
) -> dict[str, Any]:
    document = {
        "kind": specification.kind,
        "required_observable_id": specification.required_observable_id,
        "target_definition": specification.target.to_document(),
        "params": {
            "lookback": specification.lookback,
        },
    }
    if observation_spec is not None:
        document["observation_spec"] = {
            "observation_spec_id": observation_spec.observation_spec_id,
            "observable_id": observation_spec.observable_id,
            "adapter_kind": observation_spec.adapter_kind,
            "source_id": observation_spec.source_id,
            "resolution": observation_spec.resolution,
        }
    return document


def asset_observable_observation_spec(
    *,
    observation_spec_id: str,
    observable_id: str = "daily_close",
) -> ObservationSpec:
    return ObservationSpec(
        observation_spec_id=observation_spec_id,
        observable_id=observable_id,
        adapter_kind="signal_noise_asset_observable",
    )


def get_signal_spec(signal_id: str) -> SignalSpec:
    try:
        return _SPECIFICATION_DEFINITIONS[signal_id]
    except KeyError as exc:
        available = ", ".join(sorted(_SPECIFICATION_DEFINITIONS))
        raise ValueError(
            f"unknown signal spec definition: {signal_id} "
            f"(available: {available})"
        ) from exc


def find_signal_spec(
    signal_id: str,
) -> SignalSpec | None:
    return _SPECIFICATION_DEFINITIONS.get(signal_id)


def get_signal_definition(
    signal_id: str,
) -> SignalDefinition:
    specification = get_signal_spec(signal_id)
    return SignalDefinition(
        signal_id=specification.signal_id,
        kind=specification.kind,
        lookback=specification.lookback,
        target=specification.target,
        asset=default_runtime_asset(),
        observation_spec=asset_observable_observation_spec(
            observation_spec_id=f"{specification.signal_id}__default",
            observable_id=specification.required_observable_id,
        ),
    )


def find_signal_definition(
    signal_id: str,
) -> SignalDefinition | None:
    specification = find_signal_spec(signal_id)
    if specification is None:
        return None
    return SignalDefinition(
        signal_id=specification.signal_id,
        kind=specification.kind,
        lookback=specification.lookback,
        target=specification.target,
        asset=default_runtime_asset(),
        observation_spec=asset_observable_observation_spec(
            observation_spec_id=f"{specification.signal_id}__default",
            observable_id=specification.required_observable_id,
        ),
    )


def build_signal_spec(
    *,
    base_signal_id: str,
    signal_id: str,
    target_id: str | None = None,
    required_observable_id: str | None = None,
) -> SignalSpec:
    base_definition = get_signal_spec(base_signal_id)
    target_definition = (
        base_definition.target
        if target_id is None
        else get_target_definition(target_id)
    )
    return SignalSpec(
        signal_id=signal_id,
        kind=base_definition.kind,
        lookback=base_definition.lookback,
        required_observable_id=(
            base_definition.required_observable_id
            if required_observable_id is None
            else str(required_observable_id)
        ),
        target=target_definition,
    )


def build_signal_variant_definition(
    *,
    base_signal_id: str,
    signal_id: str,
    asset: str | None = None,
    target_id: str | None = None,
) -> SignalDefinition:
    specification_definition = build_signal_spec(
        base_signal_id=base_signal_id,
        signal_id=base_signal_id,
        target_id=target_id,
    )
    return SignalDefinition(
        signal_id=signal_id,
        kind=specification_definition.kind,
        lookback=specification_definition.lookback,
        target=specification_definition.target,
        asset=default_runtime_asset() if asset is None else str(asset),
        observation_spec=asset_observable_observation_spec(
            observation_spec_id=f"{signal_id}__variant",
        ),
    )


def _binding_leaf_observable_ids(
    observation_spec: ObservationSpec,
) -> set[str]:
    leaf_ids = {
        observation_spec.observable_id,
        *observation_spec.provided_observable_ids,
    }
    if observation_spec.observable_id == "daily_close":
        leaf_ids.add("daily_volume")
    return leaf_ids


def _required_observable_dependency_roots(
    observable_id: str,
    *,
    seen: set[str] | None = None,
) -> set[str]:
    active = set() if seen is None else set(seen)
    if observable_id in active:
        return {observable_id}
    active.add(observable_id)
    definition = find_observable_definition(observable_id)
    if definition is None or not definition.input_observable_ids:
        return {observable_id}
    roots: set[str] = set()
    for input_observable_id in definition.input_observable_ids:
        roots.update(
            _required_observable_dependency_roots(
                input_observable_id,
                seen=active,
            )
        )
    return roots


def _binding_supports_required_observable(
    *,
    observation_spec: ObservationSpec,
    required_observable_id: str,
) -> bool:
    if observation_spec.observable_id == required_observable_id:
        return True
    roots = _required_observable_dependency_roots(required_observable_id)
    return roots.issubset(_binding_leaf_observable_ids(observation_spec))


def build_subject_bound_signal_definition(
    *,
    specification: SignalSpec,
    subject_id: str,
    asset: str,
    observation_spec: ObservationSpec,
) -> SignalDefinition:
    if not _binding_supports_required_observable(
        observation_spec=observation_spec,
        required_observable_id=specification.required_observable_id,
    ):
        raise ValueError(
            "subject binding observation does not satisfy signal spec: "
            f"specification={specification.signal_id} "
            f"required_observable_id={specification.required_observable_id} "
            f"binding_observable_id={observation_spec.observable_id} "
            f"subject_id={subject_id}"
        )
    return SignalDefinition(
        signal_id=subject_bound_signal_id(
            base_signal_id=specification.signal_id,
            subject_id=subject_id,
        ),
        kind=specification.kind,
        lookback=specification.lookback,
        target=specification.target,
        asset=str(asset),
        observation_spec=observation_spec,
    )


def subject_bound_signal_id(
    *,
    base_signal_id: str,
    subject_id: str,
) -> str:
    return f"{base_signal_id}@{subject_id}"


@dataclass(frozen=True, init=False)
class ExecutableSignal:
    signal_id: str
    subject_id: str
    asset: str
    definition: SignalDefinition

    def __init__(
        self,
        *,
        signal_id: str | None = None,

        subject_id: str,
        asset: str,
        definition: SignalDefinition,
    ) -> None:
        resolved_signal_id = signal_id
        if resolved_signal_id is None:
            raise ValueError("executable signal requires signal_id")
        object.__setattr__(self, "signal_id", str(resolved_signal_id))
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "definition", definition)
    @property
    def target_id(self) -> str:
        return self.definition.target_id

    @property
    def signal_name(self) -> str | None:
        return self.definition.signal_name


def subject_id_for_signal(
    *,
    signal_id: str | None = None,

    asset: str = DEFAULT_SUBJECT_ID,
) -> str:
    resolved_signal_id = signal_id
    if resolved_signal_id is None:
        raise ValueError("subject_id_for_signal requires signal_id")
    if "@" in resolved_signal_id:
        return resolved_signal_id.rsplit("@", 1)[-1]
    return asset


def executable_signal_from_document(
    *,
    signal_id: str | None = None,

    asset: str,
    document: dict[str, Any] | None,
    target_id: str | None = None,
) -> ExecutableSignal:
    resolved_signal_id = signal_id
    if resolved_signal_id is None:
        raise ValueError("executable signal requires signal_id")
    if document is not None:
        definition = SignalDefinition.from_document(
            signal_id=resolved_signal_id,
            document=document,
            asset=asset,
        )
    else:
        definition = find_signal_definition(resolved_signal_id)
        if definition is None:
            resolved_target = (
                residual_return_target_definition(DEFAULT_HORIZON_DAYS)
                if target_id is None
                else get_target_definition(target_id)
            )
            definition = SignalDefinition(
                signal_id=resolved_signal_id,
                kind="opaque",
                lookback=1,
                target=resolved_target,
                asset=asset,
                observation_spec=asset_observable_observation_spec(
                    observation_spec_id=f"{resolved_signal_id}__default",
                    observable_id="daily_close",
                ),
            )
    return ExecutableSignal(
        signal_id=resolved_signal_id,
        subject_id=subject_id_for_signal(
            signal_id=resolved_signal_id,
            asset=asset,
        ),
        asset=asset,
        definition=definition,
    )
