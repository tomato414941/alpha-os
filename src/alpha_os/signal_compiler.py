from __future__ import annotations

from dataclasses import dataclass

from .signal_registry import SignalDefinition


@dataclass(frozen=True)
class CompiledSignalFamily:
    kind: str
    horizon_days: int
    target_id: str
    lookbacks: tuple[int, ...]
    definitions_by_lookback: dict[int, tuple[SignalDefinition, ...]]

    @property
    def definitions(self) -> tuple[SignalDefinition, ...]:
        ordered: list[SignalDefinition] = []
        for lookback in self.lookbacks:
            ordered.extend(self.definitions_by_lookback.get(lookback, ()))
        return tuple(ordered)


def compile_signal_families(
    definitions: list[SignalDefinition],
) -> tuple[CompiledSignalFamily, ...]:
    grouped: dict[
        tuple[str, int, str],
        dict[int, list[SignalDefinition]],
    ] = {}
    for definition in definitions:
        if definition.horizon_days is None:
            raise ValueError(
                f"signal definition must have fixed horizon_days: {definition.signal_id}"
            )
        family_key = (
            definition.kind,
            int(definition.horizon_days),
            definition.target_id,
        )
        lookback_group = grouped.setdefault(family_key, {})
        lookback_group.setdefault(int(definition.lookback), []).append(definition)

    compiled: list[CompiledSignalFamily] = []
    for (kind, horizon_days, target_id), grouped_by_lookback in sorted(grouped.items()):
        lookbacks = tuple(sorted(grouped_by_lookback))
        compiled.append(
            CompiledSignalFamily(
                kind=kind,
                horizon_days=horizon_days,
                target_id=target_id,
                lookbacks=lookbacks,
                definitions_by_lookback={
                    lookback: tuple(grouped_by_lookback[lookback])
                    for lookback in lookbacks
                },
            )
        )
    return tuple(compiled)
