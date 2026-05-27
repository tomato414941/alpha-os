from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
from pathlib import Path

from ..cli_output import (
    print_evaluation_specs,
    print_signal_discovery_specs,
    print_subject_sets,
)
from ..evaluation_application import (
    run_evaluation_use_case,
    run_walk_forward_evaluation_use_case,
)
from ..evaluation_spec import EvaluationSpec
from ..observables import ObservableDefinition
from ..config import (
    DEFAULT_SIGNAL_NOISE_BASE_URL,
    load_runtime_config,
)
from ..signal_registry import (
    SignalSpec,
)
from ..signal_generator import (
    SignalDiscoveryGenerationSpec,
    generate_signal_discovery,
    materialize_signal_specs,
)
from ..signal_discovery import SignalDiscoverySpec
from ..portfolio_decision import (
    InstrumentSpec,
    ObservationSpec,
    SubjectObservationBinding,
    SubjectSet,
    UniversePolicySpec,
)
from ..store import EvaluationStore
from ..trading_strategy import (
    TradingStrategySpec,
)
from ..universe_contract import validate_subject_set_universe_contract


@dataclass(frozen=True)
class _RuntimeManifestCatalogEntry:
    path: Path
    category: str
    instrument_types: tuple[str, ...]
    subject_kinds: tuple[str, ...]
    subject_set_count: int
    signal_discovery_count: int
    evaluation_spec_count: int


DEFAULT_PRIMARY_AGGREGATION_KIND = "corr_weighted_mean"


def _runtime_manifest_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "runtime_manifests"


def _runtime_manifest_category(
    *,
    path: Path,
    instrument_types: tuple[str, ...],
) -> str:
    name = path.stem
    types = set(instrument_types)
    if "diagnostic" in name:
        return "diagnostic"
    if name == "global_macro_futures_daily_trend":
        return "reference"
    if types & {"future", "perp", "fx_forward"}:
        return "cross_asset_example"
    if "etf" in name:
        return "etf_example"
    if "equity" in name:
        return "equity_example"
    return "narrow_example"


def _list_runtime_manifest_catalog_entries() -> list[_RuntimeManifestCatalogEntry]:
    root = _runtime_manifest_dir()
    category_order = {
        "reference": 0,
        "cross_asset_example": 1,
        "diagnostic": 2,
        "equity_example": 3,
        "etf_example": 4,
        "narrow_example": 5,
    }
    entries: list[_RuntimeManifestCatalogEntry] = []
    for path in sorted(root.glob("*.json")):
        manifest = _load_runtime_manifest(path)
        subject_set_documents = manifest.get("subject_sets", [])
        instrument_types = tuple(
            sorted(
                {
                    str(instrument.get("instrument_type"))
                    for document in subject_set_documents
                    if isinstance(document, dict)
                    for instrument in document.get("instruments", [])
                    if isinstance(instrument, dict)
                    and instrument.get("instrument_type") is not None
                }
            )
        )
        subject_kinds = tuple(
            sorted(
                {
                    str(binding.get("subject_kind"))
                    for document in subject_set_documents
                    if isinstance(document, dict)
                    for binding in document.get("bindings", [])
                    if isinstance(binding, dict) and binding.get("subject_kind") is not None
                }
            )
        )
        entries.append(
            _RuntimeManifestCatalogEntry(
                path=path,
                category=_runtime_manifest_category(
                    path=path,
                    instrument_types=instrument_types,
                ),
                instrument_types=instrument_types,
                subject_kinds=subject_kinds,
                subject_set_count=len(subject_set_documents),
                signal_discovery_count=(
                    len(manifest.get("signal_discoveries", []))
                    + len(manifest.get("generated_signal_discoveries", []))
                ),
                evaluation_spec_count=len(manifest.get("evaluation_specs", [])),
            )
        )
    return sorted(
        entries,
        key=lambda item: (category_order.get(item.category, 99), item.path.name),
    )


def cmd_list_runtime_manifests(_args: argparse.Namespace) -> int:
    root = _runtime_manifest_dir()
    entries = _list_runtime_manifest_catalog_entries()
    print("alpha-os runtime manifests")
    print(f"  Root:     {root}")
    print(f"  Count:    {len(entries)}")
    for entry in entries:
        instrument_types = "-" if not entry.instrument_types else ",".join(entry.instrument_types)
        subject_kinds = "-" if not entry.subject_kinds else ",".join(entry.subject_kinds)
        print(
            f"  {entry.path.name} "
            f"category={entry.category} "
            f"instrument_types={instrument_types} "
            f"subject_kinds={subject_kinds} "
            f"subject_sets={entry.subject_set_count} "
            f"signal_discoveries={entry.signal_discovery_count} "
            f"evaluation_specs={entry.evaluation_spec_count}"
        )
    return 0


def build_cli_parser(*, include_runtime_parsers: bool = True) -> argparse.ArgumentParser:
    public_commands = (
        "init",
        "apply-manifest",
        "run-evaluation",
        "run-walk-forward",
        "list-manifests",
    )
    parser = argparse.ArgumentParser(
        prog="alpha-os",
        description="alpha-os evaluation engine",
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{" + ",".join(public_commands) + "}",
    )
    init = sub.add_parser("init", help="Initialize an alpha-os runtime database")
    init.add_argument("--db", type=str, default=None)

    if include_runtime_parsers:
        apply_manifest = sub.add_parser(
            "apply-manifest",
            help=(
                "Apply runtime manifest resources including observables, signal specs, "
                "subject sets, strategy specs, and evaluation specs"
            ),
        )
        apply_manifest.add_argument("--db", type=str, default=None)
        apply_manifest.add_argument("--manifest", type=str, required=True)

        sub.add_parser(
            "list-manifests",
            help="List checked-in runtime manifests with categories",
        )

    run_evaluation = sub.add_parser(
        "run-evaluation",
        help="Execute one evaluation spec",
    )
    run_evaluation.add_argument("--db", type=str, default=None)
    run_evaluation.add_argument("--evaluation-spec-id", type=str, required=True)
    run_evaluation.add_argument(
        "--strategy-id",
        action="append",
        default=None,
    )
    run_evaluation.add_argument(
        "--base-url",
        type=str,
        default=None,
    )
    run_evaluation.add_argument("--details", action="store_true")

    run_walk_forward_evaluation = sub.add_parser(
        "run-walk-forward",
        help="Execute train folds for one evaluation spec and then evaluate frozen fold artifacts",
    )
    run_walk_forward_evaluation.add_argument("--db", type=str, default=None)
    run_walk_forward_evaluation.add_argument("--evaluation-spec-id", type=str, required=True)
    run_walk_forward_evaluation.add_argument(
        "--strategy-id",
        action="append",
        default=None,
    )
    run_walk_forward_evaluation.add_argument(
        "--base-url",
        type=str,
        default=None,
    )
    run_walk_forward_evaluation.add_argument("--details", action="store_true")

    return parser


def _default_evaluation_id(*, subject_id: str, target_id: str, date: str) -> str:
    return f"{subject_id}:{target_id}:{date}"


@contextmanager
def _runtime_store(db_path: str | None) -> Iterator[tuple[object, EvaluationStore]]:
    cfg = load_runtime_config(db_path=db_path)
    store = EvaluationStore(cfg.db_path)
    try:
        yield cfg, store
    finally:
        store.close()


def _rebalance_interval_steps_from_strategy_policy(
    rebalance: str | None,
    rebalance_interval_steps: int | None = None,
) -> int | None:
    if rebalance_interval_steps is not None:
        return int(rebalance_interval_steps)
    if rebalance in {None, "", "-", "none"}:
        return None
    prefix = "every_"
    suffix = "_steps"
    if not rebalance.startswith(prefix) or not rebalance.endswith(suffix):
        raise ValueError(f"unsupported strategy rebalance policy: {rebalance}")
    value = rebalance[len(prefix) : -len(suffix)]
    try:
        steps = int(value)
    except ValueError as exc:
        raise ValueError(f"unsupported strategy rebalance policy: {rebalance}") from exc
    if steps < 1:
        raise ValueError(f"unsupported strategy rebalance policy: {rebalance}")
    return steps


def cmd_init_db(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        store.ensure_schema()
    print(f"Initialized runtime db: {cfg.db_path}")
    print(f"  Subject:  {cfg.default_subject_id}")
    print(f"  Target:   {cfg.target_id}")
    return 0


def _load_runtime_manifest(path: str | Path) -> dict[str, object]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("runtime manifest must be a JSON object")
    return raw


def _build_evaluation_result_key(
    *,
    evaluation_spec_id: str,
    strategy_id: str,
) -> str:
    return f"{evaluation_spec_id}:{strategy_id}"


def _target_key(target: tuple[str, str]) -> str:
    return target[0]


def _target_strategy_id(target: tuple[str, str]) -> str:
    return target[1]


def _select_evaluation_targets(
    read_port,
    *,
    evaluation_spec_id: str,
    strategy_ids: tuple[str, ...] | None,
) -> tuple[tuple[str, str], ...]:
    targets = tuple(
        read_port.list_evaluation_targets(
            evaluation_spec_id=evaluation_spec_id,
            limit=10_000,
        )
    )
    if not targets:
        raise ValueError(
            f"evaluation spec requires at least one evaluation target: {evaluation_spec_id}"
        )
    if strategy_ids:
        allowed_strategy_ids = set(strategy_ids)
        targets = tuple(
            target for target in targets if _target_strategy_id(target) in allowed_strategy_ids
        )
        if not targets:
            raise ValueError(
                f"evaluation spec does not contain requested strategies: {evaluation_spec_id}"
            )
    return tuple(
        sorted(
            targets,
            key=lambda target: (
                _target_strategy_id(target),
                _target_key(target),
            ),
        )
    )


def _runtime_manifest_paths_with_extends(manifest_path: Path) -> tuple[Path, ...]:
    return (*_extended_runtime_manifest_paths(manifest_path), manifest_path)


def _resolve_runtime_manifest_path(value: str | Path) -> Path:
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(_runtime_manifest_dir() / path)
        if path.suffix != ".json":
            candidates.append(_runtime_manifest_dir() / f"{path}.json")
    if path.suffix != ".json":
        candidates.append(path.with_suffix(".json"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ValueError(f"runtime manifest does not exist: {value}")


def _extended_runtime_manifest_paths(manifest_path: Path) -> tuple[Path, ...]:
    manifest = _load_runtime_manifest(manifest_path)
    raw_extends = manifest.get("extends_manifest")
    if raw_extends is None:
        return ()
    if isinstance(raw_extends, str):
        values = [raw_extends]
    elif isinstance(raw_extends, list) and all(isinstance(item, str) for item in raw_extends):
        values = list(raw_extends)
    else:
        raise ValueError("runtime manifest extends_manifest must be a string or list of strings")
    resolved: list[Path] = []
    for value in values:
        relative_candidate = manifest_path.parent / value
        if relative_candidate.exists():
            resolved.append(relative_candidate)
        else:
            resolved.append(_resolve_runtime_manifest_path(value))
    return tuple(resolved)


class _RuntimeManifestReadPort:
    def __init__(self, *, manifest_paths: tuple[Path, ...], created_at: str):
        self.created_at = created_at
        self.observables: dict[str, ObservableDefinition] = {}
        self.signal_specs: dict[str, SignalSpec] = {}
        self.subject_sets: dict[str, SubjectSet] = {}
        self.signal_discoveries: dict[str, SignalDiscoverySpec] = {}
        self.trading_strategies: dict[str, TradingStrategySpec] = {}
        self.evaluation_specs: dict[str, EvaluationSpec] = {}
        for manifest_path in manifest_paths:
            self._apply_manifest_document(_load_runtime_manifest(manifest_path))

    def _apply_manifest_document(self, manifest: dict[str, object]) -> None:
        observable_documents = manifest.get("observables", [])
        specification_documents = manifest.get("signal_specs", [])
        subject_set_documents = manifest.get("subject_sets", [])
        signal_discovery_documents = manifest.get("signal_discoveries", [])
        strategy_spec_documents = manifest.get("strategy_specs", [])
        generated_signal_discovery_documents = manifest.get(
            "generated_signal_discoveries",
            [],
        )
        evaluation_spec_documents = manifest.get("evaluation_specs", [])
        for name, documents in (
            ("observables", observable_documents),
            ("signal_specs", specification_documents),
            ("subject_sets", subject_set_documents),
            ("signal_discoveries", signal_discovery_documents),
            ("strategy_specs", strategy_spec_documents),
            ("generated_signal_discoveries", generated_signal_discovery_documents),
            ("evaluation_specs", evaluation_spec_documents),
        ):
            if not isinstance(documents, list):
                raise ValueError(f"runtime manifest {name} must be a list")
        for item in observable_documents:
            if not isinstance(item, dict):
                raise ValueError("runtime manifest observables must contain objects")
            definition = ObservableDefinition.from_document(item)
            self.observables[definition.observable_id] = definition
        for item in specification_documents:
            self._add_signal_spec_document(item)
        for item in subject_set_documents:
            if not isinstance(item, dict):
                raise ValueError("runtime manifest subject_sets must contain objects")
            subject_set = _subject_set_from_document(item)
            self.subject_sets[subject_set.subject_set_id or ""] = subject_set
        all_signal_discovery_documents = list(signal_discovery_documents)
        for item in generated_signal_discovery_documents:
            if not isinstance(item, dict):
                raise ValueError(
                    "runtime manifest generated_signal_discoveries must contain objects"
                )
            generated_signal_discovery = _generated_signal_discovery_from_document(item)
            for definition in materialize_signal_specs(generated_signal_discovery):
                self.signal_specs[definition.signal_id] = definition
            all_signal_discovery_documents.append(
                {
                    "signal_discovery_id": generated_signal_discovery.signal_discovery_id,
                    **generated_signal_discovery.to_document(),
                }
            )
        for item in all_signal_discovery_documents:
            if not isinstance(item, dict):
                raise ValueError("runtime manifest signal_discoveries must contain objects")
            signal_discovery = _signal_discovery_from_document(item)
            self.signal_discoveries[signal_discovery.signal_discovery_id] = signal_discovery
        for item in strategy_spec_documents:
            if not isinstance(item, dict):
                raise ValueError("runtime manifest strategy_specs must contain objects")
            trading_strategy = _trading_strategy_from_document(item)
            self.trading_strategies[trading_strategy.strategy_id] = trading_strategy
        for item in evaluation_spec_documents:
            if not isinstance(item, dict):
                raise ValueError("runtime manifest evaluation_specs must contain objects")
            evaluation_spec_id, evaluation_spec = _evaluation_spec_from_document(item)
            for strategy_id in evaluation_spec.strategy_ids:
                if strategy_id not in self.trading_strategies:
                    raise ValueError(
                        f"unknown strategy for evaluation spec strategy_ids: {strategy_id}"
                    )
            self.evaluation_specs[evaluation_spec_id] = evaluation_spec

    def _add_signal_spec_document(self, document: object) -> None:
        if not isinstance(document, dict):
            raise ValueError("runtime manifest signal_specs must contain objects")
        signal_id = document.get("signal_id")
        if not isinstance(signal_id, str) or not signal_id:
            raise ValueError("runtime manifest signal spec is missing signal_id")
        definition = SignalSpec.from_document(signal_id=signal_id, document=document)
        self.signal_specs[definition.signal_id] = definition

    def get_evaluation_spec(self, evaluation_spec_id: str):
        evaluation_spec = self.evaluation_specs.get(evaluation_spec_id)
        if evaluation_spec is None:
            return None
        return _ManifestState(
            evaluation_spec_id=evaluation_spec_id,
            definition=evaluation_spec,
        )

    def list_evaluation_targets(
        self,
        *,
        evaluation_spec_id: str | None = None,
        limit: int = 100,
    ):
        if evaluation_spec_id is None:
            targets = [
                (
                    _build_evaluation_result_key(
                        evaluation_spec_id=spec_id,
                        strategy_id=strategy_id,
                    ),
                    strategy_id,
                )
                for spec_id, spec in self.evaluation_specs.items()
                for strategy_id in spec.strategy_ids
            ]
        else:
            spec = self.evaluation_specs.get(evaluation_spec_id)
            targets = (
                []
                if spec is None
                else [
                    (
                        _build_evaluation_result_key(
                            evaluation_spec_id=evaluation_spec_id,
                            strategy_id=strategy_id,
                        ),
                        strategy_id,
                    )
                    for strategy_id in spec.strategy_ids
                ]
            )
        targets = sorted(targets, key=_target_key)
        return tuple(targets[: max(int(limit), 0)])

    def get_trading_strategy(self, strategy_id: str):
        strategy = self.trading_strategies.get(strategy_id)
        if strategy is None:
            return None
        return _ManifestState(strategy_id=strategy_id, trading_strategy=strategy)

    def get_signal_discovery_spec(self, signal_discovery_id: str):
        signal_discovery = self.signal_discoveries.get(signal_discovery_id)
        if signal_discovery is None:
            return None
        return _ManifestState(
            signal_discovery_id=signal_discovery_id,
            definition=signal_discovery,
        )


@dataclass(frozen=True)
class _ManifestState:
    def __init__(self, **values):
        object.__setattr__(self, "_values", dict(values))

    def __getattr__(self, name: str):
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _subject_set_from_document(document: dict[str, object]) -> SubjectSet:
    subject_set_id = document.get("subject_set_id")
    instruments = document.get("instruments", [])
    observation_specs = document.get("observation_specs")
    bindings = document.get("bindings")
    universe_policy = document.get("universe_policy", {})
    if not isinstance(subject_set_id, str) or not subject_set_id:
        raise ValueError("subject set manifest is missing subject_set_id")
    if instruments is not None and not isinstance(instruments, list):
        raise ValueError(f"subject set manifest instruments must be a list: {subject_set_id}")
    if not isinstance(observation_specs, list) or not observation_specs:
        raise ValueError(f"subject set manifest is missing observation_specs: {subject_set_id}")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError(f"subject set manifest is missing bindings: {subject_set_id}")
    if not isinstance(universe_policy, dict):
        raise ValueError(
            f"subject set manifest universe_policy must be an object: {subject_set_id}"
        )
    parsed_instruments: list[InstrumentSpec] = []
    for item in instruments:
        if not isinstance(item, dict):
            raise ValueError(f"subject set manifest instruments must be objects: {subject_set_id}")
        instrument_id = item.get("instrument_id")
        instrument_type = item.get("instrument_type")
        asset = item.get("asset")
        if not isinstance(instrument_id, str) or not instrument_id:
            raise ValueError(
                f"subject set manifest instrument is missing instrument_id: {subject_set_id}"
            )
        if not isinstance(instrument_type, str) or not instrument_type:
            raise ValueError(
                f"subject set manifest instrument is missing instrument_type: {subject_set_id}"
            )
        if not isinstance(asset, str) or not asset:
            raise ValueError(f"subject set manifest instrument is missing asset: {subject_set_id}")
        parsed_instruments.append(
            InstrumentSpec(
                instrument_id=instrument_id,
                instrument_type=instrument_type,
                asset=asset,
                venue=None if item.get("venue") is None else str(item.get("venue")),
                quote_ccy=(None if item.get("quote_ccy") is None else str(item.get("quote_ccy"))),
                collateral_ccy=(
                    None if item.get("collateral_ccy") is None else str(item.get("collateral_ccy"))
                ),
                contract_family=(
                    None
                    if item.get("contract_family") is None
                    else str(item.get("contract_family"))
                ),
                underlying_id=(
                    None if item.get("underlying_id") is None else str(item.get("underlying_id"))
                ),
                asset_class=(
                    None if item.get("asset_class") is None else str(item.get("asset_class"))
                ),
                region=(None if item.get("region") is None else str(item.get("region"))),
                liquidity_tier=(
                    None if item.get("liquidity_tier") is None else str(item.get("liquidity_tier"))
                ),
                cluster=(None if item.get("cluster") is None else str(item.get("cluster"))),
                expiry=None if item.get("expiry") is None else str(item.get("expiry")),
                roll_rule=(None if item.get("roll_rule") is None else str(item.get("roll_rule"))),
                multiplier=(
                    None if item.get("multiplier") is None else float(item.get("multiplier"))
                ),
                margin_model=(
                    None if item.get("margin_model") is None else str(item.get("margin_model"))
                ),
            )
        )
    parsed_specs: list[ObservationSpec] = []
    for item in observation_specs:
        if not isinstance(item, dict):
            raise ValueError(
                f"subject set manifest observation_specs must be objects: {subject_set_id}"
            )
        observation_spec_id = item.get("observation_spec_id")
        observable_id = item.get("observable_id")
        if not isinstance(observation_spec_id, str) or not observation_spec_id:
            raise ValueError(
                f"subject set manifest observation spec is missing observation_spec_id: {subject_set_id}"
            )
        if not isinstance(observable_id, str) or not observable_id:
            raise ValueError(
                f"subject set manifest observation spec is missing observable_id: {subject_set_id}"
            )
        provided_observable_ids = item.get("provided_observable_ids", [])
        if not isinstance(provided_observable_ids, list) or any(
            not isinstance(value, str) or not value for value in provided_observable_ids
        ):
            raise ValueError(
                "subject set manifest observation spec provided_observable_ids must be a list of strings: "
                f"{subject_set_id}"
            )
        parsed_specs.append(
            ObservationSpec(
                observation_spec_id=observation_spec_id,
                observable_id=observable_id,
                adapter_kind=str(item.get("adapter_kind", "signal_noise_asset_observable")),
                source_id=str(item.get("source_id", "signal_noise")),
                resolution=str(item.get("resolution", "1d")),
                provided_observable_ids=tuple(provided_observable_ids),
                research_price_observable_id=(
                    None
                    if item.get("research_price_observable_id") is None
                    else str(item.get("research_price_observable_id"))
                ),
                tradable_price_observable_id=(
                    None
                    if item.get("tradable_price_observable_id") is None
                    else str(item.get("tradable_price_observable_id"))
                ),
                metadata_observable_ids=tuple(
                    str(value)
                    for value in item.get("metadata_observable_ids", [])
                    if isinstance(value, str) and value
                ),
            )
        )
    parsed_bindings: list[SubjectObservationBinding] = []
    for item in bindings:
        if not isinstance(item, dict):
            raise ValueError(f"subject set manifest bindings must be objects: {subject_set_id}")
        binding_subject_id = item.get("subject_id")
        asset = item.get("asset")
        observation_spec_id = item.get("observation_spec_id")
        if not isinstance(binding_subject_id, str) or not binding_subject_id:
            raise ValueError(
                f"subject set manifest binding is missing subject_id: {subject_set_id}"
            )
        if not isinstance(asset, str) or not asset:
            raise ValueError(f"subject set manifest binding is missing asset: {subject_set_id}")
        if not isinstance(observation_spec_id, str) or not observation_spec_id:
            raise ValueError(
                f"subject set manifest binding is missing observation_spec_id: {subject_set_id}"
            )
        parsed_bindings.append(
            SubjectObservationBinding(
                subject_id=binding_subject_id,
                subject_kind=str(item.get("subject_kind", "asset")),
                asset=asset,
                observation_spec_id=observation_spec_id,
                instrument_id=(
                    None if item.get("instrument_id") is None else str(item.get("instrument_id"))
                ),
            )
        )
    subject_set = SubjectSet(
        subject_set_id=subject_set_id,
        instruments=tuple(parsed_instruments),
        observation_specs=tuple(parsed_specs),
        bindings=tuple(parsed_bindings),
        universe_policy=UniversePolicySpec(
            base_currency=(
                None
                if universe_policy.get("base_currency") is None
                else str(universe_policy.get("base_currency"))
            ),
            trading_calendar=(
                None
                if universe_policy.get("trading_calendar") is None
                else str(universe_policy.get("trading_calendar"))
            ),
            benchmark_id=(
                None
                if universe_policy.get("benchmark_id") is None
                else str(universe_policy.get("benchmark_id"))
            ),
        ),
    )
    validate_subject_set_universe_contract(subject_set)
    return subject_set


def _signal_discovery_from_document(
    document: dict[str, object],
) -> SignalDiscoverySpec:
    signal_discovery_id = document.get("signal_discovery_id")
    if not isinstance(signal_discovery_id, str) or not signal_discovery_id:
        raise ValueError("signal discovery spec manifest is missing signal_discovery_id")
    return SignalDiscoverySpec.from_document(
        signal_discovery_id=signal_discovery_id,
        document=document,
    )


def _generated_signal_discovery_from_document(
    document: dict[str, object],
) -> SignalDiscoverySpec:
    generation_spec = SignalDiscoveryGenerationSpec.from_document(document)
    return generate_signal_discovery(generation_spec)


def _evaluation_spec_from_document(
    document: dict[str, object],
) -> tuple[str, EvaluationSpec]:
    evaluation_spec_id = document.get("evaluation_spec_id")
    if not isinstance(evaluation_spec_id, str) or not evaluation_spec_id:
        raise ValueError("evaluation spec manifest is missing evaluation_spec_id")
    return evaluation_spec_id, EvaluationSpec.from_document(document)


def _trading_strategy_from_document(
    document: dict[str, object],
) -> TradingStrategySpec:
    strategy_id = document.get("strategy_id")
    trading_strategy_document = document.get("trading_strategy")
    if (not isinstance(strategy_id, str) or not strategy_id) and isinstance(
        trading_strategy_document, dict
    ):
        nested_strategy_id = trading_strategy_document.get("strategy_id")
        if isinstance(nested_strategy_id, str) and nested_strategy_id:
            strategy_id = nested_strategy_id
    if not isinstance(strategy_id, str) or not strategy_id:
        raise ValueError("strategy spec manifest is missing strategy_id")
    if not isinstance(trading_strategy_document, dict):
        raise ValueError("strategy spec manifest is missing trading_strategy")
    return TradingStrategySpec.from_document(
        {
            **trading_strategy_document,
            "strategy_id": str(strategy_id),
        }
    )


def cmd_apply_runtime_manifest(args: argparse.Namespace) -> int:
    manifest = _load_runtime_manifest(args.manifest)
    observable_documents = manifest.get("observables", [])
    specification_documents = manifest.get("signal_specs", [])
    subject_set_documents = manifest.get("subject_sets", [])
    manifest_instruments = [
        instrument
        for document in subject_set_documents
        if isinstance(document, dict)
        for instrument in document.get("instruments", [])
        if isinstance(instrument, dict)
    ]
    manifest_instrument_types = sorted(
        {
            str(item.get("instrument_type"))
            for item in manifest_instruments
            if item.get("instrument_type") is not None
        }
    )
    signal_discovery_documents = manifest.get("signal_discoveries", [])
    strategy_spec_documents = manifest.get("strategy_specs", [])
    generated_signal_discovery_documents = manifest.get(
        "generated_signal_discoveries",
        [],
    )
    evaluation_spec_documents = manifest.get("evaluation_specs", [])
    if not isinstance(observable_documents, list):
        raise ValueError("runtime manifest observables must be a list")
    if not isinstance(specification_documents, list):
        raise ValueError("runtime manifest signal_specs must be a list")
    if not isinstance(subject_set_documents, list):
        raise ValueError("runtime manifest subject_sets must be a list")
    if not isinstance(signal_discovery_documents, list):
        raise ValueError("runtime manifest signal_discoveries must be a list")
    if not isinstance(strategy_spec_documents, list):
        raise ValueError("runtime manifest strategy_specs must be a list")
    if not isinstance(generated_signal_discovery_documents, list):
        raise ValueError("runtime manifest generated_signal_discoveries must be a list")
    if not isinstance(evaluation_spec_documents, list):
        raise ValueError("runtime manifest evaluation_specs must be a list")
    with _runtime_store(args.db) as (cfg, store):
        store.ensure_schema()
        created_observables = 0
        created_specifications = 0
        created_subject_sets = 0
        created_signal_discoveries = 0
        created_strategy_specs = 0
        created_evaluation_specs = 0
        for item in observable_documents:
            if not isinstance(item, dict):
                raise ValueError("runtime manifest observables must contain objects")
            definition = ObservableDefinition.from_document(item)
            _, created = store.register_observable(
                definition.observable_id,
                definition=definition,
            )
            if created:
                created_observables += 1
        for item in specification_documents:
            if not isinstance(item, dict):
                raise ValueError("runtime manifest signal_specs must contain objects")
            signal_id = item.get("signal_id")
            if not isinstance(signal_id, str) or not signal_id:
                raise ValueError("runtime manifest signal spec is missing signal_id")
            definition = SignalSpec.from_document(
                signal_id=signal_id,
                document=item,
            )
            store.register_observable(definition.required_observable_id)
            _, created = store.register_signal_spec(
                definition.signal_id,
                definition=definition,
            )
            if created:
                created_specifications += 1
        registered_subject_sets = []
        for item in subject_set_documents:
            if not isinstance(item, dict):
                raise ValueError("runtime manifest subject_sets must contain objects")
            subject_set = _subject_set_from_document(item)
            for observation_spec in subject_set.observation_specs:
                if store.get_observable(observation_spec.observable_id) is None:
                    raise ValueError(
                        "unknown observable for subject set observation spec: "
                        f"{observation_spec.observable_id}"
                    )
            state = store.upsert_subject_set(
                subject_set.subject_set_id or "",
                definition=subject_set,
            )
            registered_subject_sets.append(state)
            created_subject_sets += 1
        registered_signal_discoveries = []
        registered_strategy_specs = []
        all_signal_discovery_documents = list(signal_discovery_documents)
        generated_specification_documents: list[dict[str, object]] = []
        for item in generated_signal_discovery_documents:
            if not isinstance(item, dict):
                raise ValueError(
                    "runtime manifest generated_signal_discoveries must contain objects"
                )
            generated_signal_discovery = _generated_signal_discovery_from_document(item)
            for definition in materialize_signal_specs(generated_signal_discovery):
                generated_specification_documents.append(
                    {
                        "signal_id": definition.signal_id,
                        **definition.to_document(),
                    }
                )
            all_signal_discovery_documents.append(
                {
                    "signal_discovery_id": generated_signal_discovery.signal_discovery_id,
                    **generated_signal_discovery.to_document(),
                }
            )
        for item in generated_specification_documents:
            signal_id = item.get("signal_id")
            if not isinstance(signal_id, str) or not signal_id:
                raise ValueError("generated signal spec is missing signal_id")
            definition = SignalSpec.from_document(
                signal_id=signal_id,
                document=item,
            )
            store.register_observable(definition.required_observable_id)
            _, created = store.register_signal_spec(
                definition.signal_id,
                definition=definition,
            )
            if created:
                created_specifications += 1
        for item in all_signal_discovery_documents:
            if not isinstance(item, dict):
                raise ValueError("runtime manifest signal_discoveries must contain objects")
            signal_discovery = _signal_discovery_from_document(item)
            if store.get_subject_set(signal_discovery.subject_set_id) is None:
                raise ValueError(
                    "unknown subject set for signal discovery spec: "
                    f"{signal_discovery.subject_set_id}"
                )
            for signal_spec_id in signal_discovery.signal_spec_ids:
                if store.get_signal_spec(signal_spec_id) is None:
                    raise ValueError(f"unknown signal spec for signal discovery: {signal_spec_id}")
            available_specifications = [
                state.definition for state in store.list_signal_specs(limit=10_000)
            ]
            resolved_specification_ids = signal_discovery.resolve_signal_spec_ids(
                available_specifications
            )
            if not resolved_specification_ids:
                raise ValueError(
                    "signal discovery spec does not resolve any specifications: "
                    f"{signal_discovery.signal_discovery_id}"
                )
            state = store.upsert_signal_discovery_spec(
                signal_discovery.signal_discovery_id,
                definition=signal_discovery,
            )
            registered_signal_discoveries.append(state)
            created_signal_discoveries += 1
        for item in strategy_spec_documents:
            if not isinstance(item, dict):
                raise ValueError("runtime manifest strategy_specs must contain objects")
            trading_strategy = _trading_strategy_from_document(item)
            state = store.upsert_trading_strategy(trading_strategy=trading_strategy)
            registered_strategy_specs.append(state)
            created_strategy_specs += 1
        registered_evaluation_specs = []
        for item in evaluation_spec_documents:
            if not isinstance(item, dict):
                raise ValueError("runtime manifest evaluation_specs must contain objects")
            evaluation_spec_id, evaluation_spec = _evaluation_spec_from_document(item)
            for strategy_id in evaluation_spec.strategy_ids:
                if store.get_trading_strategy(strategy_id) is None:
                    raise ValueError(
                        f"unknown strategy for evaluation spec strategy_ids: {strategy_id}"
                    )
            state = store.upsert_evaluation_spec(
                evaluation_spec_id,
                definition=evaluation_spec,
            )
            registered_evaluation_specs.append(state)
            created_evaluation_specs += 1
    print("Applied runtime manifest")
    print(f"  DB:             {cfg.db_path}")
    print(f"  Manifest:       {Path(args.manifest)}")
    print(
        "  InstrumentTypes: "
        + ("none" if not manifest_instrument_types else ", ".join(manifest_instrument_types))
    )
    print(f"  Observables:    total={len(observable_documents)} created={created_observables}")
    print(
        f"  Specifications: total={len(specification_documents)} created={created_specifications}"
    )
    print(f"  SubjectSets:    total={len(subject_set_documents)} upserted={created_subject_sets}")
    print(
        "  SignalDiscoveries: "
        f"total={len(signal_discovery_documents)} upserted={created_signal_discoveries}"
    )
    print(
        f"  StrategySpecs:  total={len(strategy_spec_documents)} upserted={created_strategy_specs}"
    )
    print(
        "  EvalSpecs:      "
        f"total={len(evaluation_spec_documents)} upserted={created_evaluation_specs}"
    )
    if registered_subject_sets:
        print_subject_sets(registered_subject_sets)
    if registered_signal_discoveries:
        print_signal_discovery_specs(registered_signal_discoveries)
    if registered_evaluation_specs:
        print_evaluation_specs(registered_evaluation_specs)
    return 0


def cmd_run_evaluation(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        run_result_state = run_evaluation_use_case(
            store=store,
            evaluation_spec_id=str(args.evaluation_spec_id),
            strategy_ids=(
                None if args.strategy_id is None else tuple(str(item) for item in args.strategy_id)
            ),
            base_url=DEFAULT_SIGNAL_NOISE_BASE_URL if args.base_url is None else str(args.base_url),
        )
    _print_evaluation_run_summary(run_result_state)
    return 0


def cmd_run_walk_forward_evaluation(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (_cfg, store):
        run_result_state = run_walk_forward_evaluation_use_case(
            store=store,
            evaluation_spec_id=str(args.evaluation_spec_id),
            strategy_ids=(
                None if args.strategy_id is None else tuple(str(item) for item in args.strategy_id)
            ),
            base_url=DEFAULT_SIGNAL_NOISE_BASE_URL if args.base_url is None else str(args.base_url),
            evaluation_targets=getattr(args, "evaluation_targets", None),
        )
    _print_evaluation_run_summary(run_result_state)
    return 0


def _print_evaluation_run_summary(run_result_state) -> None:
    run_result = (
        run_result_state.run_result if hasattr(run_result_state, "run_result") else run_result_state
    )
    print("alpha-os evaluation run")
    print(f"  RunResult:    {run_result.evaluation_run_result_id}")
    print(f"  Evaluation spec:  {run_result.evaluation_spec_id}")
    print(f"  Results: {len(run_result.results)}")


def main(argv: list[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return cmd_init_db(args)
        if args.command == "apply-manifest":
            return cmd_apply_runtime_manifest(args)
        if args.command == "list-manifests":
            return cmd_list_runtime_manifests(args)
        if args.command == "run-evaluation":
            return cmd_run_evaluation(args)
        if args.command == "run-walk-forward":
            return cmd_run_walk_forward_evaluation(args)
    except ValueError as exc:
        parser.error(str(exc))
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
