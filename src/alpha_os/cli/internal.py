from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
from pathlib import Path

from ..cli_output import (
    print_evaluation_cases,
    print_evaluation_specs,
    format_snapshot_replay_artifacts,
    print_evaluation_snapshot,
    print_signal_competition_summary,
    print_signal_details,
    print_signal_metric,
    print_signal_discovery_specs,
    print_meta_aggregation_comparison,
    print_meta_prediction_metrics,
    print_meta_predictions,
    print_observables,
    print_portfolio_decisions,
    print_signal_specs,
    print_subject_set_backend_checks,
    print_subject_sets,
    print_target_summaries,
)
from ..evaluation_application import (
    run_evaluation_use_case,
    run_walk_forward_evaluation_use_case,
)
from ..evaluation_cost_config import (
    EvaluationRebalanceFrictionPolicySpec,
    TradingEnvironment,
)
from ..evaluation_spec import EvaluationSpec
from ..portfolio_construction_config import (
    PortfolioConstructionSizingSpec,
    PortfolioConstructionSpec,
)
from ..strategy_variant import (
    StrategyVariantConfig as _StrategyVariantConfig,
    strategy_variant_config_from_strategy as _strategy_variant_config_from_strategy,
)
from ..observables import ObservableDefinition
from ..evaluation_runtime import (
    apply_evaluation,
    apply_evaluations_batch,
    update_evaluation_state,
)
from ..metrics_service import refresh_target_metrics
from ..meta_aggregation_service import refresh_target_meta_predictions
from ..meta_metrics_service import refresh_target_meta_prediction_metrics
from ..evaluation_generation import (
    generate_evaluation_input_from_signal_noise,
    generate_evaluation_inputs_from_signal_noise,
    write_evaluation_input,
    write_evaluation_inputs,
)
from ..config import (
    DEFAULT_SIGNAL_NOISE_BASE_URL,
    DEFAULT_SUBJECT_ID,
    DEFAULT_TARGET,
    default_runtime_asset,
    load_runtime_config,
)
from ..signal_registry import (
    SignalDefinition,
    SignalSpec,
    build_signal_spec,
    executable_signal_from_document,
)
from ..signal_generator import (
    SignalDiscoveryGenerationSpec,
    generate_signal_discovery,
    materialize_signal_specs,
)
from ..signal_discovery import SignalDiscoverySpec
from ..evaluation_inputs import (
    EvaluationInput,
    load_evaluation_input,
    load_evaluation_inputs,
)
from ..meta_aggregation_service import DEFAULT_PRIMARY_AGGREGATION_KIND
from ..portfolio_decision import (
    InstrumentSpec,
    ObservationSpec,
    CostInput,
    PortfolioDecisionAssumptions,
    PortfolioState,
    RiskInput,
    SubjectObservationBinding,
    SubjectSet,
    UniversePolicySpec,
)
from ..portfolio_decision_service import (
    apply_decision_output_constraints,
    RuntimeDecisionBuildConfig,
    build_portfolio_decision_input,
    build_portfolio_decision_input_from_compressed_belief,
    build_runtime_portfolio_state,
    persist_portfolio_decision_output,
)
from ..portfolio_sizing_policy import (
    ConstrainedOptimizerSizingPolicy,
    HistoricalModelSizingPolicy,
    SignalWeightedSizingPolicy,
    SignedMeanVarianceSizingPolicy,
    apply_portfolio_sizing_policy,
)
from ..observation_adapters import resolve_observation_metadata
from ..store import EvaluationStore, _utc_now
from ..subject_set_backfill_service import (
    resolve_subject_set_for_build,
)
from ..trading_strategy import (
    TradingStrategySpec,
)
from ..universe_contract import validate_subject_set_universe_contract
from ..signal_client import build_signal_client


@dataclass(frozen=True)
class _RuntimeManifestCatalogEntry:
    path: Path
    category: str
    instrument_types: tuple[str, ...]
    subject_kinds: tuple[str, ...]
    subject_set_count: int
    signal_discovery_count: int
    evaluation_spec_count: int


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

    def internal_parser(name: str, **kwargs) -> argparse.ArgumentParser:
        kwargs["help"] = argparse.SUPPRESS
        return sub.add_parser(name, **kwargs)

    init_db = internal_parser("init-db")
    init_db.add_argument("--db", type=str, default=None)
    init = sub.add_parser("init", help="Initialize an alpha-os runtime database")
    init.add_argument("--db", type=str, default=None)

    register = internal_parser("register-signal-candidate")
    register.add_argument("--db", type=str, default=None)
    register.add_argument(
        "--signal-id",
        "--signal-candidate-id",
        dest="signal_id",
        type=str,
        required=True,
    )
    register.add_argument("--target-id", type=str, default=None)

    if include_runtime_parsers:
        apply_manifest = sub.add_parser(
            "apply-manifest",
            help=(
                "Apply runtime manifest resources including observables, signal specs, "
                "subject sets, strategy specs, evaluation specs, and evaluation cases"
            ),
        )
        apply_manifest.add_argument("--db", type=str, default=None)
        apply_manifest.add_argument("--manifest", type=str, required=True)

        run_diagnostic_evaluation = sub.add_parser(
            "run-diagnostic-evaluation",
            help="Apply a diagnostic manifest and run its lightweight evaluation spec",
        )
        run_diagnostic_evaluation.add_argument("--db", type=str, default=None)
        run_diagnostic_evaluation.add_argument(
            "--manifest",
            type=str,
            default="global_macro_tradeable_daily_diagnostic",
        )
        run_diagnostic_evaluation.add_argument(
            "--evaluation-spec-id",
            type=str,
            default="global_macro_tradeable_daily_diagnostic_eval",
        )
        run_diagnostic_evaluation.add_argument("--base-url", type=str, default=None)
        run_diagnostic_evaluation.add_argument("--details", action="store_true")
        run_diagnostic_evaluation.add_argument(
            "--dry-run",
            action="store_true",
            help=(
                "Apply manifests and validate the diagnostic case plan without "
                "running signal discovery, backtests, or run result generation"
            ),
        )
        run_diagnostic_evaluation.add_argument(
            "--check",
            action="store_true",
            help="Fail when the dry-run diagnostic plan violates lightweight contracts",
        )

        sub.add_parser(
            "list-runtime-manifests",
            help="List checked-in runtime manifests with categories",
        )

    inspect_resources = internal_parser("inspect-runtime-resources")
    inspect_resources.add_argument("--db", type=str, default=None)
    inspect_resources.add_argument("--observable-limit", type=int, default=50)
    inspect_resources.add_argument(
        "--signal-spec-limit",
        "--signal-candidate-spec-limit",
        dest="signal_spec_limit",
        type=int,
        default=50,
    )
    inspect_resources.add_argument("--subject-set-limit", type=int, default=50)
    inspect_resources.add_argument(
        "--signal-discovery-limit",
        dest="signal_discovery_limit",
        type=int,
        default=50,
    )
    inspect_resources.add_argument("--evaluation-spec-limit", type=int, default=50)

    register_specification = internal_parser("debug-register-signal-candidate-spec")
    register_specification.add_argument("--db", type=str, default=None)
    register_specification.add_argument(
        "--signal-id",
        "--signal-candidate-id",
        dest="signal_id",
        type=str,
        required=True,
    )
    register_specification.add_argument(
        "--base-signal-id",
        "--base-signal-candidate-id",
        dest="base_signal_id",
        type=str,
        default=None,
    )
    register_specification.add_argument("--target-id", type=str, default=None)
    register_specification.add_argument("--observable-id", type=str, default=None)

    show_specifications = internal_parser("debug-show-signal-candidate-specs")
    show_specifications.add_argument("--db", type=str, default=None)
    show_specifications.add_argument("--limit", type=int, default=20)

    register_observable = internal_parser("debug-register-observable")
    register_observable.add_argument("--db", type=str, default=None)
    register_observable.add_argument("--observable-id", type=str, required=True)
    register_observable.add_argument("--family", type=str, default=None)
    register_observable.add_argument("--value-kind", type=str, default=None)
    register_observable.add_argument("--resolution", type=str, default="1d")

    show_observables = internal_parser("debug-show-observables")
    show_observables.add_argument("--db", type=str, default=None)
    show_observables.add_argument("--limit", type=int, default=50)

    deactivate = internal_parser("deactivate-signal-candidate")
    deactivate.add_argument("--db", type=str, default=None)
    deactivate.add_argument(
        "--signal-id",
        "--signal-candidate-id",
        dest="signal_id",
        type=str,
        required=True,
    )

    activate = internal_parser("activate-signal-candidate")
    activate.add_argument("--db", type=str, default=None)
    activate.add_argument(
        "--signal-id",
        "--signal-candidate-id",
        dest="signal_id",
        type=str,
        required=True,
    )

    record = internal_parser("debug-record-prediction")
    record.add_argument("--db", type=str, default=None)
    record.add_argument("--date", type=str, required=True)
    record.add_argument(
        "--signal-id",
        "--signal-candidate-id",
        dest="signal_id",
        type=str,
        required=True,
    )
    record.add_argument("--prediction", type=float, required=True)
    record.add_argument("--evaluation-id", type=str, default=None)
    record.add_argument("--target-id", type=str, default=None)
    record.add_argument("--subject-id", type=str, default=None)

    finalize = internal_parser("debug-finalize-observation")
    finalize.add_argument("--db", type=str, default=None)
    finalize.add_argument("--date", type=str, required=True)
    finalize.add_argument("--observation", type=float, required=True)
    finalize.add_argument("--evaluation-id", type=str, default=None)
    finalize.add_argument("--target-id", type=str, default=None)
    finalize.add_argument("--subject-id", type=str, default=None)

    update = internal_parser("debug-update-state")
    update.add_argument("--db", type=str, default=None)
    update.add_argument("--date", type=str, required=True)
    update.add_argument(
        "--signal-id",
        "--signal-candidate-id",
        dest="signal_id",
        type=str,
        required=True,
    )
    update.add_argument("--evaluation-id", type=str, default=None)
    update.add_argument("--target-id", type=str, default=None)
    update.add_argument("--subject-id", type=str, default=None)

    generate_input = internal_parser("debug-generate-evaluation-input")
    generate_input.add_argument("--db", type=str, default=None)
    generate_input.add_argument("--date", type=str, required=True)
    generate_input.add_argument(
        "--signal-id",
        "--signal-candidate-id",
        dest="signal_id",
        type=str,
        required=True,
    )
    generate_input.add_argument("--out", type=str, required=True)
    generate_input.add_argument("--base-url", type=str, default=DEFAULT_SIGNAL_NOISE_BASE_URL)

    generate_inputs = internal_parser("debug-generate-evaluation-inputs")
    generate_inputs.add_argument("--db", type=str, default=None)
    generate_inputs.add_argument("--start-date", type=str, required=True)
    generate_inputs.add_argument("--end-date", type=str, required=True)
    generate_inputs.add_argument(
        "--signal-id",
        "--signal-candidate-id",
        dest="signal_id",
        type=str,
        required=True,
    )
    generate_inputs.add_argument("--out", type=str, required=True)
    generate_inputs.add_argument("--base-url", type=str, default=DEFAULT_SIGNAL_NOISE_BASE_URL)

    run = internal_parser("debug-apply-evaluation")
    run.add_argument("--db", type=str, default=None)
    run.add_argument("--date", type=str, default=None)
    run.add_argument(
        "--signal-id",
        "--signal-candidate-id",
        dest="signal_id",
        type=str,
        default=None,
    )
    run.add_argument("--prediction", type=float, default=None)
    run.add_argument("--observation", type=float, default=None)
    run.add_argument("--evaluation-id", type=str, default=None)
    run.add_argument("--target-id", type=str, default=None)
    run.add_argument("--subject-id", type=str, default=None)
    run.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a JSON object with date, signal_id, prediction, observation, target_id",
    )

    batch = internal_parser("debug-apply-evaluations")
    batch.add_argument("--db", type=str, default=None)
    batch.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a JSON array of evaluation input objects",
    )

    backfill = internal_parser("debug-apply-backfill")
    backfill.add_argument("--db", type=str, default=None)
    backfill.add_argument("--start-date", type=str, required=True)
    backfill.add_argument("--end-date", type=str, required=True)
    backfill.add_argument(
        "--signal-id",
        "--signal-candidate-id",
        dest="signal_id",
        type=str,
        required=True,
    )
    backfill.add_argument("--base-url", type=str, default=DEFAULT_SIGNAL_NOISE_BASE_URL)
    backfill.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write the generated evaluation-input JSON array",
    )

    backfill_many = internal_parser("debug-apply-signal-candidates-backfill")
    backfill_many.add_argument("--db", type=str, default=None)
    backfill_many.add_argument("--start-date", type=str, required=True)
    backfill_many.add_argument("--end-date", type=str, required=True)
    backfill_many.add_argument(
        "--signal-id",
        "--signal-candidate-id",
        type=str,
        action="append",
        dest="signal_id",
        required=True,
        help="Repeat to include multiple active signals",
    )
    backfill_many.add_argument("--base-url", type=str, default=DEFAULT_SIGNAL_NOISE_BASE_URL)

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
        "run-walk-forward-evaluation",
        aliases=["run-walk-forward"],
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

    inspect_subject_set = internal_parser("inspect-subject-set")
    inspect_subject_set.add_argument("--db", type=str, default=None)
    inspect_subject_set.add_argument("--subject-set-id", type=str, default=None)
    inspect_subject_set.add_argument("--portfolio-id", type=str, default=None)
    inspect_subject_set.add_argument("--target-id", type=str, default=None)
    inspect_subject_set.add_argument("--evaluation-limit", type=int, default=5)
    inspect_subject_set.add_argument("--prediction-limit", type=int, default=5)
    inspect_subject_set.add_argument("--decision-limit", type=int, default=10)
    inspect_subject_set.add_argument("--details", action="store_true")

    status = internal_parser("debug-status")
    status.add_argument("--db", type=str, default=None)
    status.add_argument("--subject-set-id", type=str, default=None)

    show = internal_parser("debug-show-evaluations")
    show.add_argument("--db", type=str, default=None)
    show.add_argument("--limit", type=int, default=10)
    show.add_argument("--subject-set-id", type=str, default=None)

    meta = internal_parser("debug-show-meta-predictions")
    meta.add_argument("--db", type=str, default=None)
    meta.add_argument("--limit", type=int, default=10)
    meta.add_argument("--subject-set-id", type=str, default=None)

    compare_meta = internal_parser("debug-compare-meta-aggregations")
    compare_meta.add_argument("--db", type=str, default=None)
    compare_meta.add_argument("--target-id", type=str, default=None)
    compare_meta.add_argument("--subject-set-id", type=str, default=None)

    register_subject_set = internal_parser("debug-register-subject-set")
    register_subject_set.add_argument("--db", type=str, default=None)
    register_subject_set.add_argument("--subject-set-id", type=str, required=True)
    register_subject_set.add_argument(
        "--observation-spec",
        action="append",
        default=[],
        required=True,
        help="Observation spec in observation_spec_id=observable_id form",
    )
    register_subject_set.add_argument(
        "--subject-binding",
        action="append",
        default=[],
        required=True,
        help=("Subject binding in subject_id=subject_kind=asset=observation_spec_id form"),
    )
    register_subject_set.add_argument("--base-currency", type=str, default="USD")
    register_subject_set.add_argument("--trading-calendar", type=str, default="multi_venue")
    register_subject_set.add_argument("--benchmark-id", type=str, default=None)

    show_subject_sets = internal_parser("debug-show-subject-sets")
    show_subject_sets.add_argument("--db", type=str, default=None)
    show_subject_sets.add_argument("--limit", type=int, default=20)

    check_subject_set_backend = internal_parser("check-subject-set-backend")
    check_subject_set_backend.add_argument("--db", type=str, default=None)
    check_subject_set_backend.add_argument("--subject-set-id", type=str, required=True)
    check_subject_set_backend.add_argument("--base-url", type=str, default=None)

    build_decision = internal_parser("decide-portfolio")
    _add_decide_portfolio_arguments(
        build_decision,
        require_compressed_belief=True,
    )

    debug_build_decision = internal_parser("debug-decide-portfolio-runtime")
    _add_decide_portfolio_arguments(
        debug_build_decision,
        require_compressed_belief=False,
    )

    show_decisions = internal_parser("debug-show-portfolio-decisions")
    show_decisions.add_argument("--db", type=str, default=None)
    show_decisions.add_argument("--portfolio-id", type=str, default=None)
    show_decisions.add_argument("--subject-set-id", type=str, default=None)
    show_decisions.add_argument("--target-id", type=str, default=None)
    show_decisions.add_argument("--aggregation-kind", type=str, default=None)
    show_decisions.add_argument("--limit", type=int, default=10)
    show_decisions.add_argument("--details", action="store_true")

    sub._choices_actions = [
        action for action in sub._choices_actions if action.dest in public_commands
    ]
    return parser


def _default_evaluation_id(*, subject_id: str, target_id: str, date: str) -> str:
    return f"{subject_id}:{target_id}:{date}"


def _add_decide_portfolio_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_compressed_belief: bool,
) -> None:
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--portfolio-id", type=str, default="default")
    parser.add_argument("--target-id", type=str, default=None)
    parser.add_argument("--subject-id", type=str, default=None)
    parser.add_argument("--subject-set-id", type=str, default=None)
    parser.add_argument(
        "--compressed-belief-id",
        type=str,
        required=require_compressed_belief,
        default=None,
    )
    parser.add_argument("--strategy-id", type=str, default=None)
    parser.add_argument(
        "--observation-spec",
        action="append",
        default=[],
        help="Additional observation spec in observation_spec_id=observable_id form",
    )
    parser.add_argument(
        "--subject-binding",
        action="append",
        default=[],
        help=(
            "Additional subject binding in subject_id=subject_kind=asset=observation_spec_id form"
        ),
    )
    parser.add_argument(
        "--aggregation-kind",
        type=str,
        default=DEFAULT_PRIMARY_AGGREGATION_KIND,
    )
    parser.add_argument(
        "--sizing-method",
        type=str,
        choices=(
            "signal_weighted",
            "signed_mean_variance",
            "equal_weight",
            "minimum_variance",
            "risk_budgeting",
            "hierarchical_risk_parity",
            "conviction_adjusted_hierarchical_risk_parity",
        ),
        default=None,
    )
    parser.add_argument(
        "--sizing-engine",
        type=str,
        choices=("rule_based", "optimizer", "history_based"),
        default=None,
    )
    parser.add_argument("--risk-window", type=int, default=20)
    parser.add_argument("--capital-base", type=float, default=None)
    parser.add_argument("--gross-exposure-cap", type=float, default=None)
    parser.add_argument("--gross-limit", type=float, default=None)
    parser.add_argument("--net-limit", type=float, default=None)
    parser.add_argument("--rebalance-step", type=int, default=None)
    parser.add_argument("--turnover-friction", type=float, default=None)
    parser.add_argument("--market-impact-bps", type=float, default=None)
    parser.add_argument("--fee-bps", type=float, default=None)
    parser.add_argument("--bid-ask-spread-bps", type=float, default=None)
    parser.add_argument("--funding-bps-per-step", type=float, default=None)
    parser.add_argument("--borrow-fee-bps-per-step", type=float, default=None)
    parser.add_argument("--no-trade-band", type=float, default=None)


@contextmanager
def _runtime_store(db_path: str | None) -> Iterator[tuple[object, EvaluationStore]]:
    cfg = load_runtime_config(db_path=db_path)
    store = EvaluationStore(cfg.db_path)
    try:
        yield cfg, store
    finally:
        store.close()


def _active_signal_definition(
    store: EvaluationStore,
    *,
    signal_id: str,
) -> SignalDefinition:
    signal = store.get_signal(signal_id)
    if signal is None:
        raise ValueError(f"signal must exist before generation: {signal_id}")
    if signal.status != "active":
        raise ValueError(f"signal must be active before generation: {signal_id}")
    if signal.definition is None:
        raise ValueError(
            f"active signal does not define an executable generation rule: {signal_id}"
        )
    return SignalDefinition.from_document(
        signal_id=signal.signal_id,
        document=signal.definition,
        asset=signal.asset,
    )


def _executable_signal(
    store: EvaluationStore,
    *,
    signal_id: str,
):
    store.ensure_schema()
    signal = store.get_signal(signal_id)
    if signal is None:
        raise ValueError(f"signal must exist before execution: {signal_id}")
    return executable_signal_from_document(
        signal_id=signal.signal_id,
        asset=signal.asset,
        document=signal.definition,
        target_id=signal.target_id,
    )


def _generate_backfill_inputs_for_signal(
    store: EvaluationStore,
    *,
    signal_id: str,
    start_date: str,
    end_date: str,
    base_url: str,
) -> list[EvaluationInput]:
    definition = _active_signal_definition(
        store,
        signal_id=signal_id,
    )
    return generate_evaluation_inputs_from_signal_noise(
        start_date=start_date,
        end_date=end_date,
        signal_id=signal_id,
        base_url=base_url,
        definition=definition,
    )


def _unique_signal_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _portfolio_decision_assumptions_from_args(
    args: argparse.Namespace,
    *,
    subject_ids: tuple[str, ...],
    trading_strategy: TradingStrategySpec | None = None,
) -> PortfolioDecisionAssumptions:
    risk_inputs: list[RiskInput] = []
    if args.gross_exposure_cap is not None:
        risk_inputs.append(
            RiskInput(
                name="gross_exposure_cap",
                subject_id=None,
                value=float(args.gross_exposure_cap),
                unit="weight",
            )
        )
    rebalance_friction_policy = (
        None if trading_strategy is None else trading_strategy.portfolio.rebalance_friction_policy
    )
    trading_environment = (
        None if trading_strategy is None else trading_strategy.portfolio.trading_environment
    )
    cost_inputs: list[CostInput] = []
    turnover_friction = (
        rebalance_friction_policy.turnover_friction
        if args.turnover_friction is None and rebalance_friction_policy is not None
        else args.turnover_friction
    )
    if turnover_friction is not None:
        cost_inputs.append(
            CostInput(
                name="turnover_friction",
                subject_id=None,
                value=float(turnover_friction),
                basis="per_turnover",
                unit="weight",
            )
        )
    market_impact_bps = (
        trading_environment.market_impact_bps
        if args.market_impact_bps is None and trading_environment is not None
        else args.market_impact_bps
    )
    if market_impact_bps is not None:
        for subject_id in subject_ids:
            cost_inputs.append(
                CostInput(
                    name="market_impact",
                    subject_id=subject_id,
                    value=float(market_impact_bps),
                    basis="per_notional",
                    unit="bps",
                )
            )
    fee_bps = (
        trading_environment.fee_bps
        if getattr(args, "fee_bps", None) is None and trading_environment is not None
        else getattr(args, "fee_bps", None)
    )
    if fee_bps is not None:
        cost_inputs.append(
            CostInput(
                name="fee_bps",
                subject_id=None,
                value=float(fee_bps),
                basis="per_notional",
                unit="bps",
            )
        )
    bid_ask_spread_bps = (
        trading_environment.bid_ask_spread_bps
        if getattr(args, "bid_ask_spread_bps", None) is None and trading_environment is not None
        else getattr(args, "bid_ask_spread_bps", None)
    )
    if bid_ask_spread_bps is not None:
        cost_inputs.append(
            CostInput(
                name="bid_ask_spread_bps",
                subject_id=None,
                value=float(bid_ask_spread_bps),
                basis="per_notional",
                unit="bps",
            )
        )
    funding_bps_per_step = (
        trading_environment.funding_bps_per_step
        if getattr(args, "funding_bps_per_step", None) is None and trading_environment is not None
        else getattr(args, "funding_bps_per_step", None)
    )
    if funding_bps_per_step is not None:
        cost_inputs.append(
            CostInput(
                name="funding_bps_per_step",
                subject_id=None,
                value=float(funding_bps_per_step),
                basis="per_notional_per_step",
                unit="bps",
            )
        )
    borrow_fee_bps_per_step = (
        trading_environment.borrow_fee_bps_per_step
        if getattr(args, "borrow_fee_bps_per_step", None) is None
        and trading_environment is not None
        else getattr(args, "borrow_fee_bps_per_step", None)
    )
    if borrow_fee_bps_per_step is not None:
        cost_inputs.append(
            CostInput(
                name="borrow_fee_bps_per_step",
                subject_id=None,
                value=float(borrow_fee_bps_per_step),
                basis="per_short_notional_per_step",
                unit="bps",
            )
        )
    no_trade_band = (
        rebalance_friction_policy.no_trade_band
        if args.no_trade_band is None and rebalance_friction_policy is not None
        else args.no_trade_band
    )
    if no_trade_band is not None:
        for subject_id in subject_ids:
            cost_inputs.append(
                CostInput(
                    name="no_trade_band",
                    subject_id=subject_id,
                    value=float(no_trade_band),
                    basis="per_delta_weight",
                    unit="weight",
                )
            )
    return PortfolioDecisionAssumptions(
        risk_inputs=tuple(risk_inputs),
        cost_inputs=tuple(cost_inputs),
    )


def _portfolio_state_from_args(
    portfolio_state: PortfolioState,
    args: argparse.Namespace,
) -> PortfolioState:
    return PortfolioState(
        portfolio_id=portfolio_state.portfolio_id,
        as_of=portfolio_state.as_of,
        positions=portfolio_state.positions,
        capital_base=(
            portfolio_state.capital_base if args.capital_base is None else float(args.capital_base)
        ),
        gross_limit=(
            portfolio_state.gross_limit if args.gross_limit is None else float(args.gross_limit)
        ),
        net_limit=(portfolio_state.net_limit if args.net_limit is None else float(args.net_limit)),
        rebalance_step=(
            portfolio_state.rebalance_step
            if args.rebalance_step is None
            else int(args.rebalance_step)
        ),
        holding_period_days=portfolio_state.holding_period_days,
        recent_turnover=portfolio_state.recent_turnover,
        current_drawdown=portfolio_state.current_drawdown,
    )


def _evaluation_trading_config_for_signal_discovery(
    store: EvaluationStore,
    *,
    signal_discovery_id: str | None,
) -> _StrategyVariantConfig | None:
    if signal_discovery_id is None:
        return None
    matching = [
        state.trading_strategy
        for state in store.list_trading_strategies(limit=1000)
        if state.trading_strategy.signal_discovery_id == signal_discovery_id
    ]
    if not matching:
        return None
    if len(matching) == 1:
        return _strategy_variant_config_from_strategy(matching[0])
    exact = [item for item in matching if item.strategy_id.startswith("strategy:")]
    if len(exact) == 1:
        return _strategy_variant_config_from_strategy(exact[0])
    return None


def _evaluation_trading_config_for_compressed_belief(
    store: EvaluationStore,
    *,
    compressed_belief_id: str,
) -> _StrategyVariantConfig | None:
    belief_state = store.get_compressed_belief(compressed_belief_id)
    if belief_state is None:
        return None
    return _evaluation_trading_config_for_signal_discovery(
        store,
        signal_discovery_id=belief_state.belief.signal_discovery_id,
    )


def _resolved_trading_strategy_for_args(
    store: EvaluationStore,
    args: argparse.Namespace,
) -> TradingStrategySpec | None:
    raw_strategy_id = getattr(args, "strategy_id", None)
    strategy_id = None if raw_strategy_id is None else str(raw_strategy_id).strip()
    if not strategy_id:
        return None
    state = store.get_trading_strategy(strategy_id)
    if state is None:
        raise ValueError(f"unknown trading strategy: {strategy_id}")
    return state.trading_strategy


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


def _portfolio_construction_for_decision_args(
    *,
    args: argparse.Namespace,
    base: PortfolioConstructionSpec | None,
) -> PortfolioConstructionSpec | None:
    if base is None and args.gross_exposure_cap is None and args.rebalance_step is None:
        return None
    base_construction = PortfolioConstructionSpec() if base is None else base
    return PortfolioConstructionSpec(
        construction_kind=base_construction.construction_kind,
        sizing_policy=base_construction.sizing_policy,
        rebalance_interval_steps=(
            base_construction.rebalance_interval_steps
            if args.rebalance_step is None
            else int(args.rebalance_step)
        ),
        long_only=base_construction.long_only,
        direction_mode=base_construction.direction_mode,
        active_overlay=base_construction.active_overlay,
        gross_exposure_cap=(
            base_construction.gross_exposure_cap
            if args.gross_exposure_cap is None
            else float(args.gross_exposure_cap)
        ),
        asset_class_weight_caps=dict(base_construction.asset_class_weight_caps),
        cluster_weight_caps=dict(base_construction.cluster_weight_caps),
    )


def _resolved_decision_sizing_spec(
    *,
    args: argparse.Namespace,
    trading_strategy: TradingStrategySpec | None,
    base: PortfolioConstructionSpec | None,
) -> PortfolioConstructionSizingSpec:
    if args.sizing_method is not None:
        return PortfolioConstructionSizingSpec(
            sizing_method=str(args.sizing_method),
            sizing_engine=(None if args.sizing_engine is None else str(args.sizing_engine)),
        )
    strategy_sizing_method = (
        None
        if trading_strategy is None
        else trading_strategy.portfolio.portfolio_construction.sizing_method
    )
    if strategy_sizing_method is not None:
        return PortfolioConstructionSizingSpec(
            sizing_method=str(strategy_sizing_method),
            sizing_engine=(None if args.sizing_engine is None else str(args.sizing_engine)),
        )
    if base is not None:
        return PortfolioConstructionSizingSpec(
            sizing_method=base.sizing_method,
            sizing_engine=(
                base.sizing_engine if args.sizing_engine is None else str(args.sizing_engine)
            ),
        )
    return PortfolioConstructionSizingSpec(
        sizing_method="signal_weighted",
        sizing_engine=(None if args.sizing_engine is None else str(args.sizing_engine)),
    )


def _portfolio_sizing_policy_from_spec(
    sizing_spec: PortfolioConstructionSizingSpec,
) -> (
    SignalWeightedSizingPolicy
    | ConstrainedOptimizerSizingPolicy
    | SignedMeanVarianceSizingPolicy
    | HistoricalModelSizingPolicy
):
    portfolio_construction = PortfolioConstructionSpec(sizing_policy=sizing_spec)
    if portfolio_construction.sizing_method == "signal_weighted":
        if portfolio_construction.sizing_engine == "optimizer":
            return ConstrainedOptimizerSizingPolicy()
        return SignalWeightedSizingPolicy()
    if portfolio_construction.sizing_method == "signed_mean_variance":
        return SignedMeanVarianceSizingPolicy()
    if portfolio_construction.sizing_method in {
        "equal_weight",
        "minimum_variance",
        "risk_budgeting",
        "hierarchical_risk_parity",
        "conviction_adjusted_hierarchical_risk_parity",
    }:
        return HistoricalModelSizingPolicy(model_type=portfolio_construction.sizing_method)
    raise ValueError(
        "unsupported sizing configuration: "
        f"{portfolio_construction.sizing_method}/"
        f"{portfolio_construction.sizing_engine}"
    )


def _portfolio_construction_with_sizing_spec(
    portfolio_construction: PortfolioConstructionSpec | None,
    *,
    sizing_spec: PortfolioConstructionSizingSpec,
) -> PortfolioConstructionSpec:
    base = PortfolioConstructionSpec() if portfolio_construction is None else portfolio_construction
    return PortfolioConstructionSpec(
        construction_kind=base.construction_kind,
        sizing_policy=sizing_spec,
        rebalance_interval_steps=base.rebalance_interval_steps,
        long_only=base.long_only,
        direction_mode=base.direction_mode,
        active_overlay=base.active_overlay,
        gross_exposure_cap=base.gross_exposure_cap,
        asset_class_weight_caps=dict(base.asset_class_weight_caps),
        cluster_weight_caps=dict(base.cluster_weight_caps),
    )


def _portfolio_construction_for_decision_strategy(
    *,
    trading_strategy: TradingStrategySpec | None,
    base: PortfolioConstructionSpec | None,
) -> PortfolioConstructionSpec | None:
    if trading_strategy is None:
        return base
    construction = trading_strategy.portfolio.portfolio_construction
    base_construction = PortfolioConstructionSpec() if base is None else base
    return PortfolioConstructionSpec(
        construction_kind=base_construction.construction_kind,
        sizing_policy=base_construction.sizing_policy,
        rebalance_interval_steps=trading_strategy.portfolio.rebalance_interval_steps,
        long_only=construction.long_only,
        direction_mode=construction.direction_mode,
        active_overlay=base_construction.active_overlay,
        gross_exposure_cap=(
            base_construction.gross_exposure_cap
            if construction.gross_exposure_cap is None
            else float(construction.gross_exposure_cap)
        ),
        asset_class_weight_caps=(
            dict(base_construction.asset_class_weight_caps)
            if not construction.asset_class_weight_caps
            else dict(construction.asset_class_weight_caps)
        ),
        cluster_weight_caps=(
            dict(base_construction.cluster_weight_caps)
            if not construction.cluster_weight_caps
            else dict(construction.cluster_weight_caps)
        ),
    )


def _subject_set_from_args(
    args: argparse.Namespace,
) -> SubjectSet | None:
    observation_specs: list[ObservationSpec] = []
    seen_observation_spec_ids: set[str] = set()
    for raw_value in getattr(args, "observation_spec", []):
        value = str(raw_value).strip()
        parts = value.split("=")
        if len(parts) != 2:
            raise ValueError(
                "observation-spec must be provided as observation_spec_id=observable_id"
            )
        observation_spec_id, observable_id = (item.strip() for item in parts)
        if not observation_spec_id or not observable_id:
            raise ValueError(
                "observation-spec must be provided as observation_spec_id=observable_id"
            )
        if observation_spec_id in seen_observation_spec_ids:
            raise ValueError(f"duplicate observation-spec mapping: {observation_spec_id}")
        observation_specs.append(
            ObservationSpec(
                observation_spec_id=observation_spec_id,
                observable_id=observable_id,
                adapter_kind="signal_noise_asset_observable",
            )
        )
        seen_observation_spec_ids.add(observation_spec_id)

    items: list[SubjectObservationBinding] = []
    seen_subject_ids: set[str] = set()
    for raw_value in getattr(args, "subject_binding", []):
        value = str(raw_value).strip()
        parts = value.split("=")
        if len(parts) == 3:
            subject_id, asset, observation_spec_id = (item.strip() for item in parts)
            subject_kind = "asset"
        elif len(parts) == 4:
            subject_id, subject_kind, asset, observation_spec_id = (item.strip() for item in parts)
        else:
            raise ValueError(
                "subject-binding must be provided as "
                "subject_id=subject_kind=asset=observation_spec_id"
            )
        if not subject_id or not subject_kind or not asset or not observation_spec_id:
            raise ValueError(
                "subject-binding must be provided as "
                "subject_id=subject_kind=asset=observation_spec_id"
            )
        if subject_id in seen_subject_ids:
            raise ValueError(f"duplicate subject-binding mapping: {subject_id}")
        items.append(
            SubjectObservationBinding(
                subject_id=subject_id,
                asset=asset,
                observation_spec_id=observation_spec_id,
                subject_kind=subject_kind,
            )
        )
        seen_subject_ids.add(subject_id)
    if not items:
        return None
    subject_set_id = getattr(args, "subject_set_id", None)
    return SubjectSet(
        subject_set_id=None if subject_set_id is None else str(subject_set_id),
        observation_specs=tuple(observation_specs),
        bindings=tuple(items),
        universe_policy=UniversePolicySpec(
            base_currency=str(getattr(args, "base_currency", None) or "USD"),
            trading_calendar=str(getattr(args, "trading_calendar", None) or "multi_venue"),
            benchmark_id=str(
                getattr(args, "benchmark_id", None) or subject_set_id or "runtime_inline"
            ),
        ),
    )


def _resolved_subject_set_for_decision_build(
    store: EvaluationStore,
    *,
    args: argparse.Namespace,
    trading_strategy: TradingStrategySpec | None,
) -> SubjectSet | None:
    subject_set = resolve_subject_set_for_build(
        store,
        args,
        inline_subject_set=_subject_set_from_args(args),
    )
    if subject_set is not None:
        return subject_set
    if args.subject_id is not None:
        return None
    if trading_strategy is None or trading_strategy.subject_set_id is None:
        return None
    state = store.get_subject_set(trading_strategy.subject_set_id)
    if state is None:
        raise ValueError(f"unknown subject set: {trading_strategy.subject_set_id}")
    validate_subject_set_universe_contract(state.definition)
    return state.definition


def _resolved_target_id_for_decision_build(
    cfg,
    args: argparse.Namespace,
    *,
    trading_strategy: TradingStrategySpec | None,
) -> str:
    if args.target_id is not None:
        return str(args.target_id)
    if trading_strategy is not None and trading_strategy.target_id is not None:
        return str(trading_strategy.target_id)
    return cfg.target_id


def _resolved_subject_set_scope(
    store: EvaluationStore,
    *,
    subject_set_id: str | None,
) -> tuple[SubjectSet | None, tuple[str, ...]]:
    if subject_set_id is None:
        return None, ()
    state = store.get_subject_set(str(subject_set_id))
    if state is None:
        raise ValueError(f"unknown subject set: {subject_set_id}")
    validate_subject_set_universe_contract(state.definition)
    return state.definition, state.definition.subject_ids


def _subject_set_backend_checks(
    subject_set: SubjectSet,
    *,
    base_url: str,
) -> list[dict[str, str | bool | None]]:
    client = build_signal_client(base_url=base_url)
    return _subject_set_backend_checks_with_client(subject_set, client=client)


def _subject_set_backend_checks_with_client(
    subject_set: SubjectSet,
    *,
    client,
) -> list[dict[str, str | bool | None]]:
    checks: list[dict[str, str | bool | None]] = []
    for binding in subject_set.bindings:
        observation_spec = subject_set.observation_spec_for_subject(binding.subject_id)
        metadata = resolve_observation_metadata(
            observation_spec,
            asset=binding.asset,
            base_url=client.base_url,
            client=client,
        )
        checks.append(
            {
                "subject_id": binding.subject_id,
                "subject_kind": binding.subject_kind,
                "asset": binding.asset,
                "observable_id": observation_spec.observable_id,
                "source_id": observation_spec.source_id,
                "resolution": observation_spec.resolution,
                "available": bool(metadata.get("available")),
                "category": None
                if metadata.get("category") is None
                else str(metadata.get("category", "")),
                "signal_type": None
                if metadata.get("signal_type") is None
                else str(metadata.get("signal_type", "")),
                "last_updated": metadata.get("last_updated"),
            }
        )
    return checks


def _ensure_subject_set_backend_available(
    subject_set: SubjectSet,
    *,
    base_url: str,
) -> None:
    validate_subject_set_universe_contract(subject_set)
    client = build_signal_client(base_url=base_url)
    if not client.health():
        return
    checks = _subject_set_backend_checks_with_client(subject_set, client=client)
    missing = [item for item in checks if not bool(item["available"])]
    if not missing:
        return
    joined = ", ".join(
        f"{item['subject_id']}->{item['observable_id']}@{item['source_id']}" for item in missing
    )
    raise ValueError(f"subject set contains unavailable backend observations: {joined}")


def _latest_snapshot_for_subjects(
    store: EvaluationStore,
    *,
    subject_ids: tuple[str, ...],
) -> object | None:
    if not subject_ids:
        snapshots = store.list_evaluation_snapshots(limit=1)
        return snapshots[0] if snapshots else None
    for snapshot in store.list_evaluation_snapshots(limit=200):
        if snapshot.subject_id in subject_ids:
            return snapshot
    return None


def _list_signals_for_subjects(
    store: EvaluationStore,
    *,
    subject_ids: tuple[str, ...],
    default_subject_id: str,
) -> list:
    scoped_subject_ids = subject_ids or (default_subject_id,)
    signals: list = []
    for subject_id in scoped_subject_ids:
        signals.extend(store.list_signals(subject_id=subject_id, target_id=None))
    return signals


def _list_meta_predictions_for_subjects(
    store: EvaluationStore,
    *,
    subject_ids: tuple[str, ...],
    default_subject_id: str,
    fallback_assets: tuple[str, ...] = (),
    limit: int,
) -> list:
    scoped_subject_ids = subject_ids or (default_subject_id,)
    items = []
    for subject_id in scoped_subject_ids:
        items.extend(store.list_meta_predictions(subject_id=subject_id, limit=limit))
    if not items and fallback_assets:
        for asset in fallback_assets:
            items.extend(store.list_meta_predictions(asset=asset, limit=limit))
    ordered = sorted(
        items,
        key=lambda item: (item.updated_at, item.evaluation_id, item.aggregation_kind),
        reverse=True,
    )
    return ordered[: max(int(limit), 1)]


def _list_meta_metrics_for_subjects(
    store: EvaluationStore,
    *,
    subject_ids: tuple[str, ...],
    default_subject_id: str,
    fallback_assets: tuple[str, ...] = (),
    target_id: str | None = None,
) -> list:
    scoped_subject_ids = subject_ids or (default_subject_id,)
    items = []
    for subject_id in scoped_subject_ids:
        items.extend(
            store.list_meta_prediction_metrics(
                subject_id=subject_id,
                target_id=target_id,
            )
        )
    if not items and fallback_assets:
        for asset in fallback_assets:
            items.extend(
                store.list_meta_prediction_metrics(
                    asset=asset,
                    target_id=target_id,
                )
            )
    return items


def _list_evaluation_snapshots_for_subjects(
    store: EvaluationStore,
    *,
    subject_ids: tuple[str, ...],
    limit: int,
) -> list:
    if not subject_ids:
        return store.list_evaluation_snapshots(limit=limit)
    max_limit = max(int(limit), 1)
    scanned = store.list_evaluation_snapshots(limit=max_limit * max(len(subject_ids), 1) * 20)
    filtered = [item for item in scanned if item.subject_id in subject_ids]
    return filtered[:max_limit]


def _list_portfolio_decisions_for_subject_set(
    store: EvaluationStore,
    *,
    portfolio_id: str | None,
    subject_set: SubjectSet | None,
    target_id: str | None,
    aggregation_kind: str | None,
    limit: int,
) -> list:
    max_limit = max(int(limit), 1)
    scanned = store.list_portfolio_decisions(
        portfolio_id=portfolio_id,
        target_id=target_id,
        aggregation_kind=aggregation_kind,
        limit=max_limit
        if subject_set is None
        else max_limit * max(len(subject_set.bindings), 1) * 20,
    )
    if subject_set is None:
        return scanned
    allowed_subject_ids = set(subject_set.subject_ids)
    filtered = [item for item in scanned if item.subject_id in allowed_subject_ids]
    return filtered[:max_limit]


def _resolve_evaluation_input(
    args: argparse.Namespace,
    *,
    default_target_id: str,
    default_subject_id: str,
) -> EvaluationInput:
    if args.input:
        evaluation_input = load_evaluation_input(args.input)
        if (
            args.evaluation_id is not None
            or args.target_id is not None
            or args.subject_id is not None
        ):
            evaluation_input = EvaluationInput(
                date=evaluation_input.date,
                signal_id=evaluation_input.signal_id,
                prediction=evaluation_input.prediction,
                observation=evaluation_input.observation,
                evaluation_id=(
                    evaluation_input.evaluation_id
                    if args.evaluation_id is None
                    else str(args.evaluation_id)
                ),
                subject_id=(
                    evaluation_input.subject_id if args.subject_id is None else str(args.subject_id)
                ),
                target_id=(
                    evaluation_input.target_id if args.target_id is None else str(args.target_id)
                ),
                funding_cost_bps=evaluation_input.funding_cost_bps,
                borrow_fee_bps=evaluation_input.borrow_fee_bps,
                roll_cost_bps=evaluation_input.roll_cost_bps,
                contract_multiplier=evaluation_input.contract_multiplier,
            )
        return evaluation_input

    required = {
        "date": args.date,
        "signal_id": args.signal_id,
        "prediction": args.prediction,
        "observation": args.observation,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"apply-evaluation requires --input or manual values for: {joined}")
    return EvaluationInput(
        date=str(args.date),
        signal_id=str(args.signal_id),
        prediction=float(args.prediction),
        observation=float(args.observation),
        evaluation_id=None if args.evaluation_id is None else str(args.evaluation_id),
        subject_id=default_subject_id if args.subject_id is None else str(args.subject_id),
        target_id=default_target_id if args.target_id is None else str(args.target_id),
    )


def cmd_init_db(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        store.ensure_schema()
    print(f"Initialized runtime db: {cfg.db_path}")
    print(f"  Subject:  {cfg.default_subject_id}")
    print(f"  Target:   {cfg.target_id}")
    return 0


def cmd_register_signal(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        target_id = cfg.target_id if args.target_id is None else str(args.target_id)
        signal, created = store.register_signal(
            args.signal_id,
            target_id=target_id,
        )
    outcome = "created" if created else "existing"
    print(f"Signal [{outcome}] {signal.signal_id}")
    print_signal_details(signal)
    return 0


def _load_runtime_manifest(path: str | Path) -> dict[str, object]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("runtime manifest must be a JSON object")
    return raw


def _runtime_manifest_evaluation_case_ids(manifest_path: Path) -> tuple[str, ...]:
    manifest = _load_runtime_manifest(manifest_path)
    raw_cases = manifest.get("evaluation_cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("runtime manifest evaluation_cases must be a list")
    case_ids: list[str] = []
    for item in raw_cases:
        if not isinstance(item, dict):
            raise ValueError("runtime manifest evaluation_cases must contain objects")
        case_id = item.get("evaluation_case_id")
        if isinstance(case_id, str) and case_id:
            case_ids.append(case_id)
    return tuple(case_ids)


def _build_evaluation_result_key(
    *,
    evaluation_spec_id: str,
    strategy_id: str,
) -> str:
    return f"{evaluation_spec_id}:{strategy_id}"


def _case_key(case: tuple[str, str]) -> str:
    return case[0]


def _case_strategy_id(case: tuple[str, str]) -> str:
    return case[1]


def _select_evaluation_cases(
    read_port,
    *,
    evaluation_spec_id: str,
    strategy_ids: tuple[str, ...] | None,
    evaluation_case_ids: tuple[str, ...] | None = None,
) -> tuple[tuple[str, str], ...]:
    cases = tuple(
        read_port.list_evaluation_cases(
            evaluation_spec_id=evaluation_spec_id,
            limit=10_000,
        )
    )
    if not cases:
        raise ValueError(
            "evaluation spec requires at least one evaluation case: "
            f"{evaluation_spec_id}"
        )
    if evaluation_case_ids is not None:
        allowed_case_ids = set(evaluation_case_ids)
        cases = tuple(
            case for case in cases if _case_key(case) in allowed_case_ids
        )
        if not cases:
            raise ValueError(
                "evaluation spec does not contain requested evaluation cases: "
                f"{evaluation_spec_id}"
            )
    if strategy_ids:
        allowed_strategy_ids = set(strategy_ids)
        cases = tuple(
            case
            for case in cases
            if _case_strategy_id(case) in allowed_strategy_ids
        )
        if not cases:
            raise ValueError(
                "evaluation spec does not contain requested strategies: "
                f"{evaluation_spec_id}"
            )
    return tuple(
        sorted(
            cases,
            key=lambda case: (
                _case_strategy_id(case),
                _case_key(case),
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
        self.evaluation_cases_by_spec: dict[str, dict[str, tuple[str, str]]] = {}
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
        evaluation_case_documents = manifest.get("evaluation_cases", [])
        for name, documents in (
            ("observables", observable_documents),
            ("signal_specs", specification_documents),
            ("subject_sets", subject_set_documents),
            ("signal_discoveries", signal_discovery_documents),
            ("strategy_specs", strategy_spec_documents),
            ("generated_signal_discoveries", generated_signal_discovery_documents),
            ("evaluation_specs", evaluation_spec_documents),
            ("evaluation_cases", evaluation_case_documents),
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
            self.signal_discoveries[signal_discovery.signal_discovery_id] = (
                signal_discovery
            )
        for item in strategy_spec_documents:
            if not isinstance(item, dict):
                raise ValueError("runtime manifest strategy_specs must contain objects")
            trading_strategy = _trading_strategy_from_document(item)
            self.trading_strategies[trading_strategy.strategy_id] = trading_strategy
        for item in evaluation_spec_documents:
            if not isinstance(item, dict):
                raise ValueError("runtime manifest evaluation_specs must contain objects")
            evaluation_spec_id, evaluation_spec = _evaluation_spec_from_document(item)
            self.evaluation_specs[evaluation_spec_id] = evaluation_spec
        for item in evaluation_case_documents:
            if not isinstance(item, dict):
                raise ValueError("runtime manifest evaluation_cases must contain objects")
            evaluation_spec_id = item.get("evaluation_spec_id")
            if not isinstance(evaluation_spec_id, str) or not evaluation_spec_id:
                raise ValueError("evaluation case manifest is missing evaluation_spec_id")
            if self.get_evaluation_spec(evaluation_spec_id) is None:
                raise ValueError(f"unknown evaluation spec for evaluation case: {evaluation_spec_id}")
            evaluation_case = _evaluation_case_from_document(
                self,
                evaluation_spec_id=evaluation_spec_id,
                document=item,
            )
            self.evaluation_cases_by_spec.setdefault(evaluation_spec_id, {})[
                _case_key(evaluation_case)
            ] = evaluation_case

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

    def list_evaluation_cases(
        self,
        *,
        evaluation_spec_id: str | None = None,
        limit: int = 100,
    ):
        if evaluation_spec_id is None:
            cases = [
                case
                for spec_cases in self.evaluation_cases_by_spec.values()
                for case in spec_cases.values()
            ]
        else:
            cases = list(
                self.evaluation_cases_by_spec.get(evaluation_spec_id, {}).values()
            )
        cases = sorted(cases, key=_case_key)
        return tuple(cases[: max(int(limit), 0)])

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


def _evaluation_case_from_document(
    store: EvaluationStore,
    *,
    evaluation_spec_id: str,
    document: dict[str, object],
) -> tuple[str, str]:
    allowed_fields = {"evaluation_case_id", "evaluation_spec_id", "strategy_id"}
    unsupported_fields = sorted(set(document) - allowed_fields)
    if unsupported_fields:
        raise ValueError(
            "evaluation case manifest only supports evaluation_case_id, "
            "evaluation_spec_id, and strategy_id: "
            + ",".join(unsupported_fields)
        )
    strategy_id = document.get("strategy_id")
    if not isinstance(strategy_id, str) or not strategy_id:
        raise ValueError("evaluation case manifest is missing strategy_id")
    if store.get_trading_strategy(strategy_id) is None:
        raise ValueError(f"unknown strategy for evaluation case: {strategy_id}")
    if document.get("evaluation_spec_id") != evaluation_spec_id:
        raise ValueError(
            "evaluation case manifest evaluation_spec_id does not match selected "
            f"evaluation spec: {evaluation_spec_id}"
        )
    evaluation_case_id = document.get("evaluation_case_id")
    if not isinstance(evaluation_case_id, str) or not evaluation_case_id:
        evaluation_case_id = _build_evaluation_result_key(
            evaluation_spec_id=evaluation_spec_id,
            strategy_id=strategy_id,
        )
    return evaluation_case_id, strategy_id


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
    evaluation_case_documents = manifest.get("evaluation_cases", [])
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
    if not isinstance(evaluation_case_documents, list):
        raise ValueError("runtime manifest evaluation_cases must be a list")
    with _runtime_store(args.db) as (cfg, store):
        store.ensure_schema()
        created_observables = 0
        created_specifications = 0
        created_subject_sets = 0
        created_signal_discoveries = 0
        created_strategy_specs = 0
        created_evaluation_specs = 0
        created_evaluation_cases = 0
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
            state = store.upsert_evaluation_spec(
                evaluation_spec_id,
                definition=evaluation_spec,
            )
            registered_evaluation_specs.append(state)
            created_evaluation_specs += 1
        registered_evaluation_cases = []
        for item in evaluation_case_documents:
            if not isinstance(item, dict):
                raise ValueError("runtime manifest evaluation_cases must contain objects")
            evaluation_spec_id = item.get("evaluation_spec_id")
            if not isinstance(evaluation_spec_id, str) or not evaluation_spec_id:
                raise ValueError("evaluation case manifest is missing evaluation_spec_id")
            if store.get_evaluation_spec(evaluation_spec_id) is None:
                raise ValueError(f"unknown evaluation spec for evaluation case: {evaluation_spec_id}")
            evaluation_case = _evaluation_case_from_document(
                store,
                evaluation_spec_id=evaluation_spec_id,
                document=item,
            )
            registered_evaluation_cases.append(evaluation_case)
            created_evaluation_cases += 1
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
    print(
        "  EvalTargets:    "
        f"total={len(evaluation_case_documents)} applied={created_evaluation_cases}"
    )
    if registered_subject_sets:
        print_subject_sets(registered_subject_sets)
    if registered_signal_discoveries:
        print_signal_discovery_specs(registered_signal_discoveries)
    if registered_evaluation_specs:
        print_evaluation_specs(registered_evaluation_specs)
    if registered_evaluation_cases:
        print_evaluation_cases(registered_evaluation_cases)
    return 0


def cmd_inspect_runtime_resources(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        store.ensure_schema()
        observables = store.list_observables(limit=int(args.observable_limit))
        specifications = store.list_signal_specs(limit=int(args.signal_spec_limit))
        subject_sets = store.list_subject_sets(limit=int(args.subject_set_limit))
        signal_discoveries = store.list_signal_discovery_specs(
            limit=int(args.signal_discovery_limit)
        )
        evaluation_specs = store.list_evaluation_specs(
            limit=int(args.evaluation_spec_limit)
        )
    print("alpha-os runtime resources")
    print(f"  DB:       {cfg.db_path}")
    print_observables(observables)
    print_signal_specs(specifications)
    print_subject_sets(subject_sets)
    print_signal_discovery_specs(signal_discoveries)
    print_evaluation_specs(evaluation_specs)
    return 0


def cmd_register_signal_spec(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        target_id = cfg.target_id if args.target_id is None else str(args.target_id)
        definition = _build_signal_spec_from_args(
            args,
            target_id=target_id,
        )
        store.register_observable(definition.required_observable_id)
        signal_spec, created = store.register_signal_spec(
            definition.signal_id,
            definition=definition,
        )
    outcome = "created" if created else "existing"
    print(f"Signal Spec [{outcome}] {signal_spec.signal_id}")
    print_signal_specs([signal_spec])
    return 0


def cmd_show_signal_specs(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (_cfg, store):
        specifications = store.list_signal_specs(limit=args.limit)
    print_signal_specs(specifications)
    return 0


def cmd_register_observable(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (_cfg, store):
        definition = _observable_definition_from_args(args)
        observable, created = store.register_observable(
            definition.observable_id,
            definition=definition,
        )
    outcome = "created" if created else "existing"
    print(f"Observable [{outcome}] {observable.observable_id}")
    print_observables([observable])
    return 0


def cmd_show_observables(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (_cfg, store):
        store.ensure_schema()
        observables = store.list_observables(limit=args.limit)
    print_observables(observables)
    return 0


def _build_signal_spec_from_args(
    args: argparse.Namespace,
    *,
    target_id: str,
) -> SignalSpec:
    base_signal_id = (
        str(args.signal_id) if args.base_signal_id is None else str(args.base_signal_id)
    )
    return build_signal_spec(
        base_signal_id=base_signal_id,
        signal_id=str(args.signal_id),
        target_id=target_id,
        required_observable_id=(None if args.observable_id is None else str(args.observable_id)),
    )


def _observable_definition_from_args(args: argparse.Namespace) -> ObservableDefinition:
    if args.family is None or args.value_kind is None:
        raise ValueError("register-observable requires --family and --value-kind")
    return ObservableDefinition(
        observable_id=str(args.observable_id),
        family=str(args.family),
        value_kind=str(args.value_kind),
        default_resolution=str(args.resolution),
        params={},
    )


def _cmd_change_signal_status(
    args: argparse.Namespace,
    *,
    action: str,
    verb: str,
) -> int:
    with _runtime_store(args.db) as (_cfg, store):
        signal = store.set_signal_status(
            args.signal_id,
            action=action,
        )
        refresh_target_metrics(
            store,
            subject_id=signal.subject_id,
            asset=signal.asset,
            target_id=signal.target_id,
        )
        refresh_target_meta_predictions(
            store,
            subject_id=signal.subject_id,
            asset=signal.asset,
            target_id=signal.target_id,
        )
        refresh_target_meta_prediction_metrics(
            store,
            subject_id=signal.subject_id,
            asset=signal.asset,
            target_id=signal.target_id,
        )
    print(f"Signal [{verb}] {signal.signal_id}")
    print_signal_details(signal)
    return 0


def cmd_deactivate_signal(args: argparse.Namespace) -> int:
    return _cmd_change_signal_status(
        args,
        action="deactivate",
        verb="deactivated",
    )


def cmd_activate_signal(args: argparse.Namespace) -> int:
    return _cmd_change_signal_status(
        args,
        action="activate",
        verb="activated",
    )


def cmd_record_prediction(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        target_id = cfg.target_id if args.target_id is None else str(args.target_id)
        executable = _executable_signal(store, signal_id=str(args.signal_id))
        subject_id = executable.subject_id if args.subject_id is None else str(args.subject_id)
        evaluation_id = args.evaluation_id or _default_evaluation_id(
            subject_id=subject_id,
            target_id=target_id,
            date=args.date,
        )
        prediction, created = store.record_prediction(
            evaluation_id=evaluation_id,
            signal_id=args.signal_id,
            prediction_value=args.prediction,
            subject_id=subject_id,
            asset=executable.asset,
            target_id=target_id,
        )
    outcome = "created" if created else "existing"
    print(f"Prediction [{outcome}] {prediction.evaluation_id}")
    print(f"  Subject:  {subject_id}")
    print(f"  Asset:    {prediction.asset}")
    print(f"  Target:   {prediction.target_id}")
    print(f"  Signal:   {prediction.signal_id}")
    print(f"  Value:    {prediction.value:.6f}")
    return 0


def cmd_finalize_observation(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        target_id = cfg.target_id if args.target_id is None else str(args.target_id)
        subject_id = cfg.default_subject_id if args.subject_id is None else str(args.subject_id)
        evaluation_id = args.evaluation_id or _default_evaluation_id(
            subject_id=subject_id,
            target_id=target_id,
            date=args.date,
        )
        observation, created = store.finalize_observation(
            evaluation_id=evaluation_id,
            observation_value=args.observation,
            subject_id=subject_id,
            asset=default_runtime_asset(subject_id),
            target_id=target_id,
        )
    outcome = "created" if created else "existing"
    print(f"Observation [{outcome}] {observation.evaluation_id}")
    print(f"  Subject:  {subject_id}")
    print(f"  Asset:    {observation.asset}")
    print(f"  Target:   {observation.target_id}")
    print(f"  Value:    {observation.value:.6f}")
    return 0


def cmd_update_state(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        target_id = cfg.target_id if args.target_id is None else str(args.target_id)
        executable = _executable_signal(store, signal_id=str(args.signal_id))
        subject_id = executable.subject_id if args.subject_id is None else str(args.subject_id)
        evaluation_id = args.evaluation_id or _default_evaluation_id(
            subject_id=subject_id,
            target_id=target_id,
            date=args.date,
        )
        snapshot, created = update_evaluation_state(
            store,
            evaluation_id=evaluation_id,
            signal_id=args.signal_id,
            subject_id=subject_id,
            target_id=target_id,
        )
        metric = store.get_signal_metric(args.signal_id)
    print_evaluation_snapshot(snapshot, created=created)
    print_signal_metric(metric)
    return 0


def cmd_generate_evaluation_input(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (_cfg, store):
        store.ensure_schema()
        definition = _active_signal_definition(
            store,
            signal_id=args.signal_id,
        )
    evaluation_input = generate_evaluation_input_from_signal_noise(
        date=args.date,
        signal_id=args.signal_id,
        base_url=args.base_url,
        definition=definition,
    )
    output_path = write_evaluation_input(args.out, evaluation_input)
    print(f"Generated evaluation input: {output_path}")
    print(f"  Subject:  {evaluation_input.subject_id}")
    print(f"  Target:   {evaluation_input.target_id}")
    print(f"  Date:     {evaluation_input.date}")
    print(f"  Signal:   {evaluation_input.signal_id}")
    print(
        f"  Signal:   pred={evaluation_input.prediction:.6f} obs={evaluation_input.observation:.6f}"
    )
    return 0


def cmd_generate_evaluation_inputs(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (_cfg, store):
        store.ensure_schema()
        definition = _active_signal_definition(
            store,
            signal_id=args.signal_id,
        )
    evaluation_inputs = generate_evaluation_inputs_from_signal_noise(
        start_date=args.start_date,
        end_date=args.end_date,
        signal_id=args.signal_id,
        base_url=args.base_url,
        definition=definition,
    )
    output_path = write_evaluation_inputs(args.out, evaluation_inputs)
    print(f"Generated evaluation inputs: {output_path}")
    print(f"  Count:    {len(evaluation_inputs)}")
    if evaluation_inputs:
        print(f"  Subject:  {evaluation_inputs[0].subject_id}")
        print(f"  Target:   {evaluation_inputs[0].target_id}")
        print(f"  Range:    {evaluation_inputs[0].date} -> {evaluation_inputs[-1].date}")
    else:
        print(f"  Subject:  {DEFAULT_SUBJECT_ID}")
        print(f"  Target:   {DEFAULT_TARGET}")
    return 0


def cmd_apply_evaluation(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        default_subject_id = (
            cfg.default_subject_id
            if args.signal_id is None
            else _executable_signal(store, signal_id=str(args.signal_id)).subject_id
        )
        evaluation_input = _resolve_evaluation_input(
            args,
            default_target_id=cfg.target_id,
            default_subject_id=default_subject_id,
        )
        input_source = "json_file" if args.input else "manual"
        evaluation_id = evaluation_input.evaluation_id or _default_evaluation_id(
            subject_id=evaluation_input.subject_id,
            target_id=evaluation_input.target_id,
            date=evaluation_input.date,
        )
        snapshot, created = apply_evaluation(
            store,
            evaluation_id=evaluation_id,
            signal_id=evaluation_input.signal_id,
            prediction_value=evaluation_input.prediction,
            observation_value=evaluation_input.observation,
            target_id=evaluation_input.target_id,
            subject_id=evaluation_input.subject_id,
            input_source=input_source,
            funding_cost_bps=evaluation_input.funding_cost_bps,
            borrow_fee_bps=evaluation_input.borrow_fee_bps,
            roll_cost_bps=evaluation_input.roll_cost_bps,
            contract_multiplier=evaluation_input.contract_multiplier,
        )
        metric = store.get_signal_metric(evaluation_input.signal_id)
    print_evaluation_snapshot(snapshot, created=created)
    print_signal_metric(metric)
    return 0


def cmd_apply_evaluations(args: argparse.Namespace) -> int:
    cfg = load_runtime_config(db_path=args.db)
    evaluation_inputs = load_evaluation_inputs(args.input)
    return _apply_evaluation_inputs(
        cfg.db_path,
        evaluation_inputs,
        input_source="json_batch",
    )


def _apply_evaluation_inputs(
    db_path: Path,
    evaluation_inputs: list[EvaluationInput],
    *,
    input_source: str,
    input_range_start: str | None = None,
    input_range_end: str | None = None,
) -> int:
    with _runtime_store(str(db_path)) as (_cfg, store):
        latest_snapshot, created_count, existing_count = apply_evaluations_batch(
            store,
            evaluation_inputs=evaluation_inputs,
            input_source=input_source,
            input_range_start=input_range_start,
            input_range_end=input_range_end,
        )
        if evaluation_inputs:
            latest_metric = store.get_signal_metric(evaluation_inputs[-1].signal_id)
        else:
            latest_metric = None

    print(
        "Batch complete: "
        f"evaluations={len(evaluation_inputs)} created={created_count} existing={existing_count}"
    )
    if latest_snapshot is not None:
        print(f"  Latest:   {latest_snapshot.evaluation_id} / {latest_snapshot.signal_id}")
        print_signal_metric(latest_metric)
    return 0


def cmd_apply_backfill(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        store.ensure_schema()
        evaluation_inputs = _generate_backfill_inputs_for_signal(
            store,
            signal_id=args.signal_id,
            start_date=args.start_date,
            end_date=args.end_date,
            base_url=args.base_url,
        )
    if args.out is not None:
        output_path = write_evaluation_inputs(args.out, evaluation_inputs)
        print(f"Wrote evaluation inputs: {output_path}")
    return _apply_evaluation_inputs(
        cfg.db_path,
        evaluation_inputs,
        input_source="signal_noise_backfill",
        input_range_start=args.start_date,
        input_range_end=args.end_date,
    )


def cmd_apply_signals_backfill(args: argparse.Namespace) -> int:
    signal_ids = _unique_signal_ids(args.signal_id)

    with _runtime_store(args.db) as (cfg, store):
        store.ensure_schema()
        all_evaluation_inputs: list[EvaluationInput] = []
        for signal_id in signal_ids:
            evaluation_inputs = _generate_backfill_inputs_for_signal(
                store,
                signal_id=signal_id,
                start_date=args.start_date,
                end_date=args.end_date,
                base_url=args.base_url,
            )
            all_evaluation_inputs.extend(evaluation_inputs)
        latest_snapshot, created_count, existing_count = apply_evaluations_batch(
            store,
            evaluation_inputs=all_evaluation_inputs,
            input_source="signal_noise_backfill",
            input_range_start=args.start_date,
            input_range_end=args.end_date,
        )
        print(
            "Batch complete: "
            f"signals={len(signal_ids)} "
            f"evaluations={len(all_evaluation_inputs)} "
            f"created={created_count} existing={existing_count}"
        )
        if latest_snapshot is not None:
            print(f"  Latest:   {latest_snapshot.evaluation_id} / {latest_snapshot.signal_id}")
        print_signal_competition_summary(
            store,
            signal_ids=signal_ids,
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = load_runtime_config(db_path=args.db)
    store = EvaluationStore(cfg.db_path)
    try:
        store.ensure_schema()
        subject_set, scoped_subject_ids = _resolved_subject_set_scope(
            store,
            subject_set_id=args.subject_set_id,
        )
        signals = _list_signals_for_subjects(
            store,
            subject_ids=scoped_subject_ids,
            default_subject_id=cfg.default_subject_id,
        )
        metrics = (
            []
            if not signals
            else store.list_signal_metrics(signal_ids=[item.signal_id for item in signals])
        )
        latest = _latest_snapshot_for_subjects(
            store,
            subject_ids=scoped_subject_ids,
        )
    finally:
        store.close()

    print("alpha-os status")
    print(f"  DB:       {Path(cfg.db_path)}")
    if subject_set is None:
        print(f"  Subject:  {cfg.default_subject_id}")
    else:
        print(f"  SubjectSet: {subject_set.subject_set_id}")
        print("  Subjects: " + ", ".join(subject_set.subject_ids))
        print("  Assets:   " + ", ".join(binding.asset for binding in subject_set.bindings))
    print("  Targets:  all")
    if latest is None and not signals:
        print("  Latest:   no evaluations recorded")
        return 0
    if latest is not None:
        print(f"  Latest:   {latest.evaluation_id} / {latest.signal_id}")
    else:
        print("  Latest:   no evaluations recorded")
    total = len(signals)
    active = sum(1 for item in signals if item.status == "active")
    inactive = sum(1 for item in signals if item.status == "inactive")
    print(f"  Signal:   total={total} active={active} inactive={inactive}")
    tracked = len(metrics)
    mean_corr = 0.0 if tracked == 0 else sum(item.corr for item in metrics) / tracked
    mmcs = [item.mmc for item in metrics if item.mmc is not None]
    mean_mmc_text = "n/a" if not mmcs else f"{sum(mmcs) / len(mmcs):.6f}"
    print(f"  Metrics:  tracked={tracked} mean_corr={mean_corr:.6f} mean_mmc={mean_mmc_text}")
    print_target_summaries(
        signals,
        {item.signal_id: item for item in metrics},
    )
    return 0


def cmd_inspect_subject_set(args: argparse.Namespace) -> int:
    cfg = load_runtime_config(db_path=args.db)
    store = EvaluationStore(cfg.db_path)
    try:
        store.ensure_schema()
        subject_set, scoped_subject_ids = _resolved_subject_set_scope(
            store,
            subject_set_id=args.subject_set_id,
        )
        signals = _list_signals_for_subjects(
            store,
            subject_ids=scoped_subject_ids,
            default_subject_id=cfg.default_subject_id,
        )
        metrics = (
            []
            if not signals
            else store.list_signal_metrics(signal_ids=[item.signal_id for item in signals])
        )
        latest = _latest_snapshot_for_subjects(
            store,
            subject_ids=scoped_subject_ids,
        )
        snapshots = _list_evaluation_snapshots_for_subjects(
            store,
            subject_ids=scoped_subject_ids,
            limit=int(args.evaluation_limit),
        )
        fallback_assets = (
            () if subject_set is None else tuple(binding.asset for binding in subject_set.bindings)
        )
        meta_predictions = _list_meta_predictions_for_subjects(
            store,
            subject_ids=scoped_subject_ids,
            default_subject_id=cfg.default_subject_id,
            fallback_assets=fallback_assets,
            limit=int(args.prediction_limit),
        )
        meta_metrics = _list_meta_metrics_for_subjects(
            store,
            subject_ids=scoped_subject_ids,
            default_subject_id=cfg.default_subject_id,
            fallback_assets=fallback_assets,
            target_id=None if args.target_id is None else str(args.target_id),
        )
        decisions = _list_portfolio_decisions_for_subject_set(
            store,
            portfolio_id=None if args.portfolio_id is None else str(args.portfolio_id),
            subject_set=subject_set,
            target_id=None if args.target_id is None else str(args.target_id),
            aggregation_kind=None,
            limit=int(args.decision_limit),
        )
    finally:
        store.close()

    print("alpha-os subject-set inspection")
    print(f"  DB:       {Path(cfg.db_path)}")
    if subject_set is None:
        print(f"  Subject:  {cfg.default_subject_id}")
    else:
        print(f"  SubjectSet: {subject_set.subject_set_id}")
        print("  Subjects: " + ", ".join(subject_set.subject_ids))
        print("  Assets:   " + ", ".join(binding.asset for binding in subject_set.bindings))
    print("  Status:")
    if latest is None and not signals:
        print("    Latest: no evaluations recorded")
    else:
        if latest is None:
            print("    Latest: no evaluations recorded")
        else:
            print(f"    Latest: {latest.evaluation_id} / {latest.signal_id}")
        total = len(signals)
        active = sum(1 for item in signals if item.status == "active")
        inactive = sum(1 for item in signals if item.status == "inactive")
        print(f"    Signal: total={total} active={active} inactive={inactive}")
        tracked = len(metrics)
        mean_corr = 0.0 if tracked == 0 else sum(item.corr for item in metrics) / tracked
        mmcs = [item.mmc for item in metrics if item.mmc is not None]
        mean_mmc_text = "n/a" if not mmcs else f"{sum(mmcs) / len(mmcs):.6f}"
        print(f"    Metrics: tracked={tracked} mean_corr={mean_corr:.6f} mean_mmc={mean_mmc_text}")
        print_target_summaries(
            signals,
            {item.signal_id: item for item in metrics},
        )
    print("  Evaluations:")
    if not snapshots:
        print("    none")
    else:
        for snapshot in snapshots:
            range_text = "-"
            if snapshot.input_range_start or snapshot.input_range_end:
                start = snapshot.input_range_start or "-"
                end = snapshot.input_range_end or "-"
                range_text = f"{start}->{end}"
            observation_text = "-"
            if snapshot.observable_id is not None and snapshot.adapter_kind is not None:
                observation_text = f"{snapshot.observable_id}@{snapshot.adapter_kind}"
                if snapshot.observation_spec_id is not None:
                    observation_text = (
                        f"{snapshot.observation_spec_id}={snapshot.observable_id}@"
                        f"{snapshot.adapter_kind}"
                    )
            replay_artifacts = format_snapshot_replay_artifacts(snapshot)
            replay_text = "" if replay_artifacts is None else f" replay={replay_artifacts}"
            print(
                f"    {snapshot.evaluation_id} hyp={snapshot.signal_id} "
                f"source={snapshot.input_source or '-'} observation={observation_text} "
                f"range={range_text} pred={snapshot.prediction_value:.6f} "
                f"obs={snapshot.observation_value:.6f} edge={snapshot.signed_edge:.6f}"
                f"{replay_text}"
            )
    print("  Meta:")
    print_meta_predictions(meta_predictions)
    print_meta_prediction_metrics(meta_metrics)
    print("  Decisions:")
    print_portfolio_decisions(decisions, show_details=bool(args.details))
    return 0


def cmd_show_evaluations(args: argparse.Namespace) -> int:
    cfg = load_runtime_config(db_path=args.db)
    store = EvaluationStore(cfg.db_path)
    try:
        store.ensure_schema()
        subject_set, scoped_subject_ids = _resolved_subject_set_scope(
            store,
            subject_set_id=args.subject_set_id,
        )
        snapshots = _list_evaluation_snapshots_for_subjects(
            store,
            subject_ids=scoped_subject_ids,
            limit=args.limit,
        )
    finally:
        store.close()

    print("alpha-os evaluations")
    print(f"  DB:       {Path(cfg.db_path)}")
    if subject_set is None:
        print(f"  Subject:  {cfg.default_subject_id}")
    else:
        print(f"  SubjectSet: {subject_set.subject_set_id}")
        print("  Subjects: " + ", ".join(subject_set.subject_ids))
        print("  Assets:   " + ", ".join(binding.asset for binding in subject_set.bindings))
    print(f"  Count:    {len(snapshots)}")
    for snapshot in snapshots:
        range_text = "-"
        if snapshot.input_range_start or snapshot.input_range_end:
            start = snapshot.input_range_start or "-"
            end = snapshot.input_range_end or "-"
            range_text = f"{start}->{end}"
        observation_text = "-"
        if snapshot.observable_id is not None and snapshot.adapter_kind is not None:
            observation_text = f"{snapshot.observable_id}@{snapshot.adapter_kind}"
            if snapshot.observation_spec_id is not None:
                observation_text = (
                    f"{snapshot.observation_spec_id}={snapshot.observable_id}@"
                    f"{snapshot.adapter_kind}"
                )
        replay_artifacts = format_snapshot_replay_artifacts(snapshot)
        replay_text = "" if replay_artifacts is None else f" replay={replay_artifacts}"
        print(
            f"  {snapshot.evaluation_id} "
            f"hyp={snapshot.signal_id} "
            f"source={snapshot.input_source or '-'} "
            f"observation={observation_text} "
            f"range={range_text} "
            f"pred={snapshot.prediction_value:.6f} "
            f"obs={snapshot.observation_value:.6f} "
            f"edge={snapshot.signed_edge:.6f}"
            f"{replay_text}"
        )
    return 0


def cmd_show_meta_predictions(args: argparse.Namespace) -> int:
    cfg = load_runtime_config(db_path=args.db)
    store = EvaluationStore(cfg.db_path)
    try:
        store.ensure_schema()
        subject_set, scoped_subject_ids = _resolved_subject_set_scope(
            store,
            subject_set_id=args.subject_set_id,
        )
        fallback_assets = (
            () if subject_set is None else tuple(binding.asset for binding in subject_set.bindings)
        )
        meta_predictions = _list_meta_predictions_for_subjects(
            store,
            subject_ids=scoped_subject_ids,
            default_subject_id=cfg.default_subject_id,
            fallback_assets=fallback_assets,
            limit=args.limit,
        )
        meta_metrics = _list_meta_metrics_for_subjects(
            store,
            subject_ids=scoped_subject_ids,
            default_subject_id=cfg.default_subject_id,
            fallback_assets=fallback_assets,
        )
    finally:
        store.close()

    print(f"  DB:       {Path(cfg.db_path)}")
    if subject_set is None:
        print(f"  Subject:  {cfg.default_subject_id}")
    else:
        print(f"  SubjectSet: {subject_set.subject_set_id}")
        print("  Subjects: " + ", ".join(subject_set.subject_ids))
        print("  Assets:   " + ", ".join(binding.asset for binding in subject_set.bindings))
    print_meta_predictions(meta_predictions)
    print_meta_prediction_metrics(meta_metrics)
    return 0


def cmd_compare_meta_aggregations(args: argparse.Namespace) -> int:
    cfg = load_runtime_config(db_path=args.db)
    store = EvaluationStore(cfg.db_path)
    try:
        store.ensure_schema()
        subject_set, scoped_subject_ids = _resolved_subject_set_scope(
            store,
            subject_set_id=args.subject_set_id,
        )
        fallback_assets = (
            () if subject_set is None else tuple(binding.asset for binding in subject_set.bindings)
        )
        metrics = _list_meta_metrics_for_subjects(
            store,
            subject_ids=scoped_subject_ids,
            default_subject_id=cfg.default_subject_id,
            fallback_assets=fallback_assets,
            target_id=None if args.target_id is None else str(args.target_id),
        )
    finally:
        store.close()

    print(f"  DB:       {Path(cfg.db_path)}")
    if subject_set is None:
        print(f"  Subject:  {cfg.default_subject_id}")
    else:
        print(f"  SubjectSet: {subject_set.subject_set_id}")
        print("  Subjects: " + ", ".join(subject_set.subject_ids))
        print("  Assets:   " + ", ".join(binding.asset for binding in subject_set.bindings))
    print_meta_aggregation_comparison(metrics)
    return 0


def cmd_register_subject_set(args: argparse.Namespace) -> int:
    cfg = load_runtime_config(db_path=args.db)
    subject_set = _subject_set_from_args(args)
    if subject_set is None:
        raise ValueError("subject set must include at least one subject-binding")
    store = EvaluationStore(cfg.db_path)
    try:
        store.ensure_schema()
        for observation_spec in subject_set.observation_specs:
            if store.get_observable(observation_spec.observable_id) is None:
                raise ValueError(
                    "unknown observable for subject set observation spec: "
                    f"{observation_spec.observable_id}"
                )
        registered_subject_set = SubjectSet(
            subject_set_id=str(args.subject_set_id),
            instruments=subject_set.instruments,
            observation_specs=subject_set.observation_specs,
            bindings=subject_set.bindings,
            universe_policy=subject_set.universe_policy,
        )
        validate_subject_set_universe_contract(registered_subject_set)
        state = store.upsert_subject_set(
            str(args.subject_set_id),
            definition=registered_subject_set,
        )
    finally:
        store.close()
    print(f"  DB:       {cfg.db_path}")
    print_subject_sets([state])
    return 0


def cmd_show_subject_sets(args: argparse.Namespace) -> int:
    cfg = load_runtime_config(db_path=args.db)
    store = EvaluationStore(cfg.db_path)
    try:
        store.ensure_schema()
        subject_sets = store.list_subject_sets(limit=int(args.limit))
    finally:
        store.close()
    print(f"  DB:       {cfg.db_path}")
    print_subject_sets(subject_sets)
    return 0


def cmd_check_subject_set_backend(args: argparse.Namespace) -> int:
    cfg = load_runtime_config(db_path=args.db)
    store = EvaluationStore(cfg.db_path)
    try:
        store.ensure_schema()
        state = store.get_subject_set(str(args.subject_set_id))
        if state is None:
            raise ValueError(f"unknown subject set: {args.subject_set_id}")
        validate_subject_set_universe_contract(state.definition)
        base_url = DEFAULT_SIGNAL_NOISE_BASE_URL if args.base_url is None else str(args.base_url)
        checks = _subject_set_backend_checks(
            state.definition,
            base_url=base_url,
        )
    finally:
        store.close()
    print(f"  DB:       {cfg.db_path}")
    print_subject_set_backend_checks(
        str(args.subject_set_id),
        checks,
        base_url=base_url,
    )
    return 0 if all(bool(item["available"]) for item in checks) else 2


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


@dataclass(frozen=True)
class _SignalDiscoveryEvaluationGroup:
    signal_discovery_id: str | None
    base_url: str
    evaluation_cases: tuple[tuple[str, str], ...]


def _group_evaluation_cases_by_signal_discovery(
    store: EvaluationStore,
    evaluation_cases: tuple[tuple[str, str], ...],
    *,
    base_url: str,
) -> tuple[_SignalDiscoveryEvaluationGroup, ...]:
    def strategy_lookup(strategy_id: str) -> TradingStrategySpec | None:
        strategy_state = store.get_trading_strategy(strategy_id)
        if strategy_state is None:
            return None
        return strategy_state.trading_strategy

    return _group_evaluation_cases_by_signal_discovery_with_strategy_lookup(
        evaluation_cases,
        strategy_lookup=strategy_lookup,
        base_url=base_url,
    )


def _group_evaluation_cases_by_signal_discovery_with_strategy_lookup(
    evaluation_cases: tuple[tuple[str, str], ...],
    *,
    strategy_lookup,
    base_url: str,
) -> tuple[_SignalDiscoveryEvaluationGroup, ...]:
    grouped: dict[str | None, list[tuple[str, str]]] = {}
    for evaluation_case in evaluation_cases:
        trading_strategy = strategy_lookup(_case_strategy_id(evaluation_case))
        if trading_strategy is None:
            raise ValueError(f"evaluation case strategy does not exist: {_case_strategy_id(evaluation_case)}")
        signal_discovery_id = trading_strategy.signal_discovery_id
        grouped.setdefault(signal_discovery_id, []).append(evaluation_case)
    groups: list[_SignalDiscoveryEvaluationGroup] = []
    for signal_discovery_id, grouped_cases in sorted(
        grouped.items(),
        key=lambda item: "" if item[0] is None else item[0],
    ):
        groups.append(
            _SignalDiscoveryEvaluationGroup(
                signal_discovery_id=signal_discovery_id,
                base_url=base_url,
                evaluation_cases=tuple(grouped_cases),
            )
        )
    return tuple(groups)


def cmd_run_evaluation(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        run_result_state = run_evaluation_use_case(
            store=store,
            evaluation_spec_id=str(args.evaluation_spec_id),
            strategy_ids=(
                None
                if args.strategy_id is None
                else tuple(str(item) for item in args.strategy_id)
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
                None
                if args.strategy_id is None
                else tuple(str(item) for item in args.strategy_id)
            ),
            base_url=DEFAULT_SIGNAL_NOISE_BASE_URL if args.base_url is None else str(args.base_url),
            evaluation_cases=getattr(args, "evaluation_cases", None),
        )
    _print_evaluation_run_summary(run_result_state)
    return 0


_DIAGNOSTIC_PROFILE_METRIC_GROUPS = {
    "prediction_diagnostics": (
        "mean_signal_forward_corr",
        "mean_signal_hit_rate",
        "mean_long_short_forward_spread",
        "mean_prediction_coverage",
    ),
    "portfolio_construction_trace": (
        "case_vol_stage_mean_gross_delta",
        "net_case_stage_mean_net_delta",
        "top_k_stage_mean_active_count_delta",
    ),
    "portfolio_concentration": (
        "mean_effective_n",
        "mean_active_position_count",
        "mean_top3_gross_share",
    ),
    "execution_trace": (
        "mean_desired_turnover",
        "mean_executed_turnover",
        "mean_turnover_suppression",
        "mean_skipped_trade_count",
        "mean_expected_execution_cost",
        "mean_trade_utility",
        "negative_utility_trade_fraction",
        "utility_rejected_turnover",
        "priority_filled_turnover",
        "partial_fill_count",
    ),
    "cost_drag": (
        "cost_to_gross_pnl",
        "execution_cost_to_gross_pnl",
        "total_execution_cost_notional",
        "top_cost_subjects",
        "top_cost_clusters",
    ),
    "signal_churn": (
        "mean_signal_abs_change",
        "mean_signal_sign_flip_rate",
        "mean_desired_weight_change",
    ),
    "decision_quality": (
        "mean_decision_net_return",
        "mean_decision_drawdown",
        "mean_decision_turnover",
        "total_decision_cost_notional",
    ),
}


_DIAGNOSTIC_DRY_RUN_EXPECTED_CASE_COUNT = 14
_DIAGNOSTIC_DRY_RUN_REQUIRED_CASE_IDS = {
    "global_macro_tradeable_daily_diagnostic_equal_weight_hold_case",
    "global_macro_tradeable_daily_diagnostic_equal_weight_monthly_hold_case",
    "global_macro_tradeable_daily_diagnostic_mean_reversion_case",
    "global_macro_tradeable_daily_diagnostic_mean_reversion_constrained_case",
    "global_macro_tradeable_daily_diagnostic_mean_reversion_optimizer_case",
    "global_macro_tradeable_daily_diagnostic_utility_looser_benefit_case",
    "global_macro_tradeable_daily_diagnostic_utility_tighter_benefit_case",
    "global_macro_tradeable_daily_diagnostic_utility_looser_budget_case",
    "global_macro_tradeable_daily_diagnostic_utility_no_budget_case",
}
def _diagnostic_optimizer_backend(
    sizing_method: str | None,
    sizing_engine: str | None,
) -> str:
    if sizing_method == "signed_mean_variance" and sizing_engine == "optimizer":
        return "cvxpy_signed_mean_variance"
    if sizing_method == "signal_weighted" and sizing_engine == "optimizer":
        return "cvxpy_constrained_optimizer"
    if sizing_engine == "history_based":
        if sizing_method == "equal_weight":
            return "history_based_equal_weight"
        return f"skfolio:{sizing_method or '-'}"
    return "rule_based_signal_weighted"


def _print_diagnostic_evaluation_focus(run_result_state) -> None:
    run_result = (
        run_result_state.run_result
        if hasattr(run_result_state, "run_result")
        else run_result_state
    )
    print("alpha-os diagnostic focus")
    print(f"  RunResult:   {run_result.evaluation_run_result_id}")
    print(f"  Evaluation spec: {run_result.evaluation_spec_id}")
    for result_key, result in run_result.results.items():
        print(f"  Result: {result_key}")
        metric_group_result_map = {
            item.metric_group_name: item for item in result.metric_group_results
        }
        for metric_group_name, metric_names in _DIAGNOSTIC_PROFILE_METRIC_GROUPS.items():
            metric_group_result = metric_group_result_map.get(metric_group_name)
            if metric_group_result is None:
                print(f"    {metric_group_name}: missing")
                continue
            metrics = [
                f"{name}={metric_group_result.metrics[name]}"
                for name in metric_names
                if name in metric_group_result.metrics
            ]
            print(f"    {metric_group_name}: " + (" ".join(metrics) if metrics else "no_metrics"))


def _print_diagnostic_evaluation_dry_run(
    *,
    evaluation_spec_state,
    evaluation_cases: tuple[tuple[str, str], ...],
    signal_discovery_groups: tuple[_SignalDiscoveryEvaluationGroup, ...],
    trading_configs_by_case_id: dict[str, _StrategyVariantConfig],
    strategies_by_case_id: dict[str, TradingStrategySpec],
) -> None:
    evaluation_spec = evaluation_spec_state.definition
    print("alpha-os diagnostic dry run")
    print(f"  Evaluation spec: {evaluation_spec_state.evaluation_spec_id}")
    print(f"  Folds:    {len(evaluation_spec.resolved_evaluation_folds)}")
    print(f"  Cases:    {len(evaluation_cases)}")
    print(f"  Groups:   {len(signal_discovery_groups)}")
    print("  MetricGroups: " + ",".join(evaluation_spec.metric_group_names))
    for group in signal_discovery_groups:
        print(
            "  SignalDiscoveryGroup: "
            f"{group.signal_discovery_id or '-'} "
            f"has_signal_discovery={str(group.signal_discovery_id is not None).lower()} "
            f"cases={len(group.evaluation_cases)} "
            f"base_url={group.base_url}"
        )
    for case in evaluation_cases:
        trading_config = trading_configs_by_case_id[_case_key(case)]
        construction = trading_config.portfolio_construction
        friction = trading_config.rebalance_friction_policy
        trading_environment = trading_config.trading_environment
        print(
            "  Result: "
            f"{_case_key(case)} "
            f"strategy={_case_strategy_id(case)} "
            f"signal_discovery={strategies_by_case_id[_case_key(case)].signal_discovery_id or '-'} "
            f"construction={construction.construction_kind} "
            f"holding_style={'equal_weight_hold' if construction.construction_kind == 'hold_baseline' else '-'} "
            f"sizing={construction.sizing_method} "
            f"engine={construction.sizing_engine or '-'} "
            f"optimizer_backend={_diagnostic_optimizer_backend(construction.sizing_method, construction.sizing_engine)} "
            f"rebalance_steps={construction.rebalance_interval_steps} "
            f"execution_mode={friction.execution_mode} "
            f"turnover_budget={friction.turnover_budget if friction.turnover_budget is not None else '-'} "
            f"benefit_scale={friction.benefit_scale} "
            f"min_trade_utility={friction.min_trade_utility} "
            f"partial_fill_enabled={str(friction.partial_fill_enabled).lower()} "
            f"cost_bps={trading_environment.market_impact_bps + trading_environment.fee_bps + trading_environment.bid_ask_spread_bps}"
        )


def _check_diagnostic_evaluation_dry_run(
    *,
    manifest_path: Path,
    evaluation_spec_state,
    evaluation_cases: tuple[tuple[str, str], ...],
    signal_discovery_groups: tuple[_SignalDiscoveryEvaluationGroup, ...],
    trading_configs_by_case_id: dict[str, _StrategyVariantConfig],
    strategies_by_case_id: dict[str, TradingStrategySpec],
) -> None:
    evaluation_spec = evaluation_spec_state.definition
    if not evaluation_spec.resolved_evaluation_folds:
        raise ValueError("diagnostic dry run check failed: evaluation_spec has no folds")
    if not evaluation_cases:
        raise ValueError("diagnostic dry run check failed: evaluation_spec has no cases")
    if not signal_discovery_groups:
        raise ValueError(
            "diagnostic dry run check failed: no signal discovery groups resolved"
        )
    case_ids = {_case_key(case) for case in evaluation_cases}
    for group in signal_discovery_groups:
        if not group.evaluation_cases:
            raise ValueError(
                "diagnostic dry run check failed: signal discovery group has no cases: "
                f"{group.signal_discovery_id or '-'}"
            )
    for case in evaluation_cases:
        if not _case_strategy_id(case):
            raise ValueError(
                "diagnostic dry run check failed: case has no strategy: "
                f"{_case_key(case)}"
            )
    if manifest_path.stem != "global_macro_tradeable_daily_diagnostic":
        return
    if len(evaluation_cases) != _DIAGNOSTIC_DRY_RUN_EXPECTED_CASE_COUNT:
        raise ValueError(
            "diagnostic dry run check failed: expected "
            f"{_DIAGNOSTIC_DRY_RUN_EXPECTED_CASE_COUNT} cases, got "
            f"{len(evaluation_cases)}"
        )
    missing_case_ids = sorted(_DIAGNOSTIC_DRY_RUN_REQUIRED_CASE_IDS - case_ids)
    if missing_case_ids:
        raise ValueError(
            "diagnostic dry run check failed: missing diagnostic cases: "
            + ",".join(missing_case_ids)
        )
    cases_by_id = {_case_key(case): case for case in evaluation_cases}
    configs_by_id = trading_configs_by_case_id
    looser_benefit = cases_by_id[
        "global_macro_tradeable_daily_diagnostic_utility_looser_benefit_case"
    ]
    if configs_by_id[_case_key(looser_benefit)].rebalance_friction_policy.benefit_scale != 2.0:
        raise ValueError(
            "diagnostic dry run check failed: looser benefit lane must use "
            "benefit_scale=2.0"
        )
    legacy_case = cases_by_id.get(
        "global_macro_tradeable_daily_diagnostic_legacy_proportional_execution_case"
    )
    if (
        legacy_case is None
        or configs_by_id[_case_key(legacy_case)].rebalance_friction_policy.execution_mode
        != "threshold"
    ):
        raise ValueError(
            "diagnostic dry run check failed: legacy proportional lane must use "
            "execution_mode=threshold"
        )
    equal_weight_hold_case = cases_by_id[
        "global_macro_tradeable_daily_diagnostic_equal_weight_hold_case"
    ]
    equal_weight_hold_construction = configs_by_id[
        _case_key(equal_weight_hold_case)
    ].portfolio_construction
    if equal_weight_hold_construction.construction_kind != "hold_baseline":
        raise ValueError(
            "diagnostic dry run check failed: equal-weight hold lane must use "
            "construction_kind=hold_baseline"
        )
    if equal_weight_hold_construction.rebalance_interval_steps != 252:
        raise ValueError(
            "diagnostic dry run check failed: equal-weight hold lane must use "
            "rebalance_interval_steps=252"
        )
    equal_weight_monthly_hold_case = cases_by_id[
        "global_macro_tradeable_daily_diagnostic_equal_weight_monthly_hold_case"
    ]
    if (
        configs_by_id[
            _case_key(equal_weight_monthly_hold_case)
        ].portfolio_construction.construction_kind
        != "hold_baseline"
    ):
        raise ValueError(
            "diagnostic dry run check failed: equal-weight monthly hold lane must use "
            "construction_kind=hold_baseline"
        )
    if (
        configs_by_id[
            _case_key(equal_weight_monthly_hold_case)
        ].portfolio_construction.rebalance_interval_steps
        != 21
    ):
        raise ValueError(
            "diagnostic dry run check failed: equal-weight monthly hold lane must use "
            "rebalance_interval_steps=21"
        )
    orthogonal_case = cases_by_id[
        "global_macro_tradeable_daily_diagnostic_mean_reversion_case"
    ]
    if (
        strategies_by_case_id[_case_key(orthogonal_case)].signal_discovery_id
        != "global_macro_tradeable_daily_diagnostic_mean_reversion_search"
    ):
        raise ValueError(
            "diagnostic dry run check failed: orthogonal lane must use "
            "mean-reversion signal discovery"
        )
    constrained_case = cases_by_id[
        "global_macro_tradeable_daily_diagnostic_mean_reversion_constrained_case"
    ]
    if (
        strategies_by_case_id[_case_key(constrained_case)].signal_discovery_id
        != "global_macro_tradeable_daily_diagnostic_mean_reversion_search"
    ):
        raise ValueError(
            "diagnostic dry run check failed: constrained mean-reversion lane "
            "must use mean-reversion signal discovery"
        )
    constrained_config = configs_by_id[_case_key(constrained_case)]
    if constrained_config.portfolio_construction.rebalance_interval_steps != 10:
        raise ValueError(
            "diagnostic dry run check failed: constrained mean-reversion lane "
            "must use rebalance_interval_steps=10"
        )
    constrained_intent = constrained_config.portfolio_construction.portfolio_intent
    if (
        constrained_intent.effective_n_floor != 10.0
        or constrained_intent.top_gross_share_cap_n != 3
        or constrained_intent.top_gross_share_cap != 0.4
    ):
        raise ValueError(
            "diagnostic dry run check failed: constrained mean-reversion lane "
            "must use the moderate concentration constraints"
        )
    constrained_friction = constrained_config.rebalance_friction_policy
    if (
        constrained_friction.no_trade_band != 0.01
        or constrained_friction.execution_cost_aversion != 3.0
        or constrained_friction.turnover_budget != 0.025
        or not constrained_friction.partial_fill_enabled
    ):
        raise ValueError(
            "diagnostic dry run check failed: constrained mean-reversion lane "
            "must use the moderate friction controls"
        )
    optimizer_case = cases_by_id[
        "global_macro_tradeable_daily_diagnostic_mean_reversion_optimizer_case"
    ]
    if (
        strategies_by_case_id[_case_key(optimizer_case)].signal_discovery_id
        != "global_macro_tradeable_daily_diagnostic_mean_reversion_search"
    ):
        raise ValueError(
            "diagnostic dry run check failed: optimizer mean-reversion lane "
            "must use mean-reversion signal discovery"
        )
    if (
        configs_by_id[_case_key(optimizer_case)].portfolio_construction.sizing_method != "signed_mean_variance"
        or configs_by_id[_case_key(optimizer_case)].portfolio_construction.sizing_engine != "optimizer"
    ):
        raise ValueError(
            "diagnostic dry run check failed: optimizer mean-reversion lane "
            "must use signed_mean_variance/optimizer sizing"
        )
    optimizer_config = configs_by_id[_case_key(optimizer_case)]
    if optimizer_config.portfolio_construction.rebalance_interval_steps != 21:
        raise ValueError(
            "diagnostic dry run check failed: optimizer mean-reversion lane "
            "must use rebalance_interval_steps=21"
        )
def _trading_strategies_for_evaluation_cases(
    read_port,
    evaluation_cases: tuple[tuple[str, str], ...],
) -> dict[str, TradingStrategySpec]:
    strategies: dict[str, TradingStrategySpec] = {}
    for case in evaluation_cases:
        state = read_port.get_trading_strategy(_case_strategy_id(case))
        if state is None:
            raise ValueError(
                f"evaluation case strategy does not exist: {_case_strategy_id(case)}"
            )
        strategies[_case_strategy_id(case)] = state.trading_strategy
    return strategies


def cmd_run_diagnostic_evaluation(args: argparse.Namespace) -> int:
    if bool(args.check) and not bool(args.dry_run):
        raise ValueError("run-diagnostic-evaluation --check requires --dry-run")
    manifest_path = _resolve_runtime_manifest_path(args.manifest)
    manifest_evaluation_case_ids = _runtime_manifest_evaluation_case_ids(manifest_path)
    if bool(args.dry_run):
        timestamp = _utc_now()
        read_port = _RuntimeManifestReadPort(
            manifest_paths=_runtime_manifest_paths_with_extends(manifest_path),
            created_at=timestamp,
        )
        evaluation_spec_state = read_port.get_evaluation_spec(str(args.evaluation_spec_id))
        if evaluation_spec_state is None:
            raise ValueError(f"evaluation spec does not exist: {args.evaluation_spec_id}")
        evaluation_cases = _select_evaluation_cases(
            read_port,
            evaluation_spec_id=str(args.evaluation_spec_id),
            strategy_ids=None,
            evaluation_case_ids=manifest_evaluation_case_ids,
        )
        effective_strategies = _trading_strategies_for_evaluation_cases(
            read_port,
            evaluation_cases,
        )
        signal_discovery_groups = _group_evaluation_cases_by_signal_discovery_with_strategy_lookup(
            evaluation_cases,
            strategy_lookup=lambda strategy_id: effective_strategies.get(strategy_id),
            base_url=DEFAULT_SIGNAL_NOISE_BASE_URL if args.base_url is None else str(args.base_url),
        )
        trading_configs_by_case_id = {
            _case_key(case): _strategy_variant_config_from_strategy(
                effective_strategies[_case_strategy_id(case)]
            )
            for case in evaluation_cases
        }
        strategies_by_case_id = {
            _case_key(case): effective_strategies[_case_strategy_id(case)]
            for case in evaluation_cases
        }
        _print_diagnostic_evaluation_dry_run(
            evaluation_spec_state=evaluation_spec_state,
            evaluation_cases=evaluation_cases,
            signal_discovery_groups=signal_discovery_groups,
            trading_configs_by_case_id=trading_configs_by_case_id,
            strategies_by_case_id=strategies_by_case_id,
        )
        if bool(args.check):
            _check_diagnostic_evaluation_dry_run(
                manifest_path=manifest_path,
                evaluation_spec_state=evaluation_spec_state,
                evaluation_cases=evaluation_cases,
                signal_discovery_groups=signal_discovery_groups,
                trading_configs_by_case_id=trading_configs_by_case_id,
                strategies_by_case_id=strategies_by_case_id,
            )
            print("  DryRunCheck: passed")
        return 0
    for base_manifest_path in _extended_runtime_manifest_paths(manifest_path):
        cmd_apply_runtime_manifest(
            argparse.Namespace(db=args.db, manifest=str(base_manifest_path))
        )
    cmd_apply_runtime_manifest(argparse.Namespace(db=args.db, manifest=str(manifest_path)))
    read_port = _RuntimeManifestReadPort(
        manifest_paths=_runtime_manifest_paths_with_extends(manifest_path),
        created_at=_utc_now(),
    )
    run_args = argparse.Namespace(
        db=args.db,
        evaluation_spec_id=str(args.evaluation_spec_id),
        strategy_id=None,
        base_url=args.base_url,
        details=args.details,
        evaluation_cases=_select_evaluation_cases(
            read_port,
            evaluation_spec_id=str(args.evaluation_spec_id),
            strategy_ids=None,
            evaluation_case_ids=manifest_evaluation_case_ids,
        ),
    )
    result = cmd_run_walk_forward_evaluation(run_args)
    if bool(args.details):
        with _runtime_store(args.db) as (_cfg, store):
            store.ensure_schema()
            run_result_state = store.get_latest_evaluation_run_result()
            if run_result_state is None:
                raise ValueError("diagnostic evaluation run result was not persisted")
        _print_diagnostic_evaluation_focus(run_result_state)
    return result


def _print_evaluation_run_summary(run_result_state) -> None:
    run_result = (
        run_result_state.run_result
        if hasattr(run_result_state, "run_result")
        else run_result_state
    )
    print("alpha-os evaluation run")
    print(f"  RunResult:    {run_result.evaluation_run_result_id}")
    print(f"  Evaluation spec:  {run_result.evaluation_spec_id}")
    print(f"  Results: {len(run_result.results)}")


def _format_runtime_strategy_summary(
    trading_strategy: TradingStrategySpec,
    *,
    sizing_method: str,
    sizing_engine: str,
) -> str:
    portfolio = trading_strategy.portfolio
    construction = portfolio.portfolio_construction
    return (
        f"{trading_strategy.strategy_id} "
        f"selection={portfolio.selection_kind} "
        f"sizing={sizing_method} "
        f"engine={sizing_engine} "
        f"rebalance=every_{portfolio.rebalance_interval_steps}_steps "
        f"top_k={'-' if portfolio.top_k is None else portfolio.top_k} "
        f"long_only={str(construction.long_only).lower()} "
        "gross_exposure_cap="
        f"{'-' if construction.gross_exposure_cap is None else construction.gross_exposure_cap}"
    )


def _run_portfolio_decision_from_input(
    *,
    store: EvaluationStore,
    cfg,
    target_id: str,
    runtime_asset: str,
    subject_id: str,
    subject_set: SubjectSet | None,
    decision_input,
    portfolio_state,
    config: RuntimeDecisionBuildConfig,
    assumptions: PortfolioDecisionAssumptions,
    sizing_policy,
    sizing_method: str,
    sizing_engine: str,
    portfolio_id: str,
    trading_strategy: TradingStrategySpec | None = None,
    portfolio_construction: PortfolioConstructionSpec | None = None,
) -> int:
    decision_output = apply_portfolio_sizing_policy(
        decision_input,
        sizing_policy=sizing_policy,
    )
    decision_output = apply_decision_output_constraints(
        decision_output,
        portfolio_state=portfolio_state,
        subject_set=subject_set,
        portfolio_construction=portfolio_construction,
        top_k=None if trading_strategy is None else trading_strategy.portfolio.top_k,
        risk_by_subject={
            item.subject_id: max(float(item.value), 0.0)
            for item in decision_input.risk_inputs
            if item.subject_id is not None
        },
    )
    persist_portfolio_decision_output(
        store,
        decision_output=decision_output,
        target_id=target_id,
        aggregation_kind=config.aggregation_kind,
        portfolio_state=portfolio_state,
        config=config,
        assumptions=assumptions,
        decision_input=decision_input,
        sizing_method=sizing_method,
        sizing_engine=sizing_engine,
        trading_strategy=trading_strategy,
    )
    decisions = store.list_portfolio_decisions(
        portfolio_id=portfolio_id,
        target_id=target_id,
        aggregation_kind=config.aggregation_kind,
        limit=10,
    )
    print(f"  DB:       {cfg.db_path}")
    print(f"  Asset:    {runtime_asset}")
    if trading_strategy is not None:
        print(
            "  Strategy: "
            + _format_runtime_strategy_summary(
                trading_strategy,
                sizing_method=sizing_method,
                sizing_engine=sizing_engine,
            )
        )
    print_portfolio_decisions(decisions[: len(decision_output.targets)])
    return 0


def cmd_decide_portfolio(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        store.ensure_schema()
        trading_strategy = _resolved_trading_strategy_for_args(store, args)
        target_id = _resolved_target_id_for_decision_build(
            cfg,
            args,
            trading_strategy=trading_strategy,
        )
        subject_set = _resolved_subject_set_for_decision_build(
            store,
            args=args,
            trading_strategy=trading_strategy,
        )
        if args.subject_id is None:
            if subject_set is not None and subject_set.bindings:
                subject_id = subject_set.bindings[0].subject_id
                runtime_asset = subject_set.bindings[0].asset
            else:
                subject_id = cfg.default_subject_id
                runtime_asset = default_runtime_asset(subject_id)
        else:
            subject_id = str(args.subject_id)
            runtime_asset = default_runtime_asset(subject_id)
        subject_ids = tuple(
            dict.fromkeys((subject_id,) + (() if subject_set is None else subject_set.subject_ids))
        )
        assumptions = _portfolio_decision_assumptions_from_args(
            args,
            subject_ids=subject_ids,
            trading_strategy=trading_strategy,
        )
        config = RuntimeDecisionBuildConfig(
            aggregation_kind=str(args.aggregation_kind),
            risk_window=int(args.risk_window),
            subject_set=subject_set,
        )
        portfolio_state = build_runtime_portfolio_state(
            store,
            portfolio_id=str(args.portfolio_id),
            aggregation_kind=config.aggregation_kind,
        )
        portfolio_state = _portfolio_state_from_args(portfolio_state, args)
        sizing_policy = _portfolio_sizing_policy_from_args(args)
        decision_input = build_portfolio_decision_input_from_compressed_belief(
            store,
            compressed_belief_id=str(args.compressed_belief_id),
            portfolio_id=str(args.portfolio_id),
            portfolio_state=portfolio_state,
            assumptions=assumptions,
        )
        if decision_input is None:
            raise ValueError("portfolio decision could not be built from compressed belief")
        belief_config = _evaluation_trading_config_for_compressed_belief(
            store,
            compressed_belief_id=str(args.compressed_belief_id),
        )
        portfolio_construction = _portfolio_construction_for_decision_strategy(
            trading_strategy=trading_strategy,
            base=_portfolio_construction_for_decision_args(
                args=args,
                base=(None if belief_config is None else belief_config.portfolio_construction),
            ),
        )
        sizing_spec = _resolved_decision_sizing_spec(
            args=args,
            trading_strategy=trading_strategy,
            base=portfolio_construction,
        )
        portfolio_construction = _portfolio_construction_with_sizing_spec(
            portfolio_construction,
            sizing_spec=sizing_spec,
        )
        sizing_policy = _portfolio_sizing_policy_from_spec(sizing_spec)
        return _run_portfolio_decision_from_input(
            store=store,
            cfg=cfg,
            target_id=target_id,
            runtime_asset=runtime_asset,
            subject_id=subject_id,
            subject_set=subject_set,
            decision_input=decision_input,
            portfolio_state=portfolio_state,
            config=config,
            assumptions=assumptions,
            sizing_policy=sizing_policy,
            sizing_method=sizing_spec.sizing_method,
            sizing_engine=sizing_spec.sizing_engine,
            portfolio_id=str(args.portfolio_id),
            trading_strategy=trading_strategy,
            portfolio_construction=portfolio_construction,
        )


def cmd_debug_decide_portfolio_runtime(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        store.ensure_schema()
        trading_strategy = _resolved_trading_strategy_for_args(store, args)
        target_id = _resolved_target_id_for_decision_build(
            cfg,
            args,
            trading_strategy=trading_strategy,
        )
        subject_set = _resolved_subject_set_for_decision_build(
            store,
            args=args,
            trading_strategy=trading_strategy,
        )
        if args.subject_id is None:
            if subject_set is not None and subject_set.bindings:
                subject_id = subject_set.bindings[0].subject_id
                runtime_asset = subject_set.bindings[0].asset
            else:
                subject_id = cfg.default_subject_id
                runtime_asset = default_runtime_asset(subject_id)
        else:
            subject_id = str(args.subject_id)
            runtime_asset = default_runtime_asset(subject_id)
        subject_ids = tuple(
            dict.fromkeys((subject_id,) + (() if subject_set is None else subject_set.subject_ids))
        )
        assumptions = _portfolio_decision_assumptions_from_args(
            args,
            subject_ids=subject_ids,
            trading_strategy=trading_strategy,
        )
        config = RuntimeDecisionBuildConfig(
            aggregation_kind=str(args.aggregation_kind),
            risk_window=int(args.risk_window),
            subject_set=subject_set,
        )
        portfolio_state = build_runtime_portfolio_state(
            store,
            portfolio_id=str(args.portfolio_id),
            aggregation_kind=config.aggregation_kind,
        )
        portfolio_state = _portfolio_state_from_args(portfolio_state, args)
        decision_input = build_portfolio_decision_input(
            store,
            runtime_asset=runtime_asset,
            target_id=target_id,
            portfolio_id=str(args.portfolio_id),
            subject_id=subject_id,
            portfolio_state=portfolio_state,
            config=config,
            assumptions=assumptions,
        )
        if decision_input is None:
            raise ValueError("portfolio decision could not be built from current runtime state")
        portfolio_construction = _portfolio_construction_for_decision_strategy(
            trading_strategy=trading_strategy,
            base=_portfolio_construction_for_decision_args(
                args=args,
                base=None,
            ),
        )
        sizing_spec = _resolved_decision_sizing_spec(
            args=args,
            trading_strategy=trading_strategy,
            base=portfolio_construction,
        )
        portfolio_construction = _portfolio_construction_with_sizing_spec(
            portfolio_construction,
            sizing_spec=sizing_spec,
        )
        sizing_policy = _portfolio_sizing_policy_from_spec(sizing_spec)
        return _run_portfolio_decision_from_input(
            store=store,
            cfg=cfg,
            target_id=target_id,
            runtime_asset=runtime_asset,
            subject_id=subject_id,
            subject_set=subject_set,
            decision_input=decision_input,
            portfolio_state=portfolio_state,
            config=config,
            assumptions=assumptions,
            sizing_policy=sizing_policy,
            sizing_method=sizing_spec.sizing_method,
            sizing_engine=sizing_spec.sizing_engine,
            portfolio_id=str(args.portfolio_id),
            trading_strategy=trading_strategy,
            portfolio_construction=portfolio_construction,
        )


def _portfolio_sizing_policy_from_args(
    args: argparse.Namespace,
) -> (
    SignalWeightedSizingPolicy
    | ConstrainedOptimizerSizingPolicy
    | SignedMeanVarianceSizingPolicy
    | HistoricalModelSizingPolicy
):
    portfolio_construction = PortfolioConstructionSpec(
        sizing_policy=PortfolioConstructionSizingSpec(
            sizing_method=str(args.sizing_method),
            sizing_engine=(None if args.sizing_engine is None else str(args.sizing_engine)),
        ),
    )
    if portfolio_construction.sizing_method == "signal_weighted":
        if portfolio_construction.sizing_engine == "optimizer":
            return ConstrainedOptimizerSizingPolicy()
        return SignalWeightedSizingPolicy()
    if portfolio_construction.sizing_method == "signed_mean_variance":
        return SignedMeanVarianceSizingPolicy()
    if portfolio_construction.sizing_method in {
        "equal_weight",
        "minimum_variance",
        "risk_budgeting",
        "hierarchical_risk_parity",
        "conviction_adjusted_hierarchical_risk_parity",
    }:
        return HistoricalModelSizingPolicy(model_type=portfolio_construction.sizing_method)
    raise ValueError(
        "unsupported sizing configuration: "
        f"{portfolio_construction.sizing_method}/"
        f"{portfolio_construction.sizing_engine}"
    )


def _direction_mode_from_args(args: argparse.Namespace) -> str | None:
    direction_mode = getattr(args, "direction_mode", None)
    if direction_mode is not None:
        if getattr(args, "long_only", False) and direction_mode != "long_only":
            raise ValueError("--long-only conflicts with --direction-mode")
        return str(direction_mode)
    if getattr(args, "long_only", False):
        return "long_only"
    return None


def _evaluation_trading_config_from_args(
    args: argparse.Namespace,
) -> _StrategyVariantConfig:
    return _StrategyVariantConfig(
        portfolio_construction=PortfolioConstructionSpec(
            sizing_policy=PortfolioConstructionSizingSpec(
                sizing_method=str(args.sizing_method),
                sizing_engine=(None if args.sizing_engine is None else str(args.sizing_engine)),
            ),
            rebalance_interval_steps=(
                1 if args.rebalance_step is None else int(args.rebalance_step)
            ),
            long_only=bool(getattr(args, "long_only", False)),
            direction_mode=_direction_mode_from_args(args),
            gross_exposure_cap=(
                None if args.gross_exposure_cap is None else float(args.gross_exposure_cap)
            ),
        ),
        rebalance_friction_policy=EvaluationRebalanceFrictionPolicySpec(
            turnover_friction=(
                0.0 if args.turnover_friction is None else float(args.turnover_friction)
            ),
            no_trade_band=(0.0 if args.no_trade_band is None else float(args.no_trade_band)),
        ),
        trading_environment=TradingEnvironment(
            market_impact_bps=(
                0.0 if args.market_impact_bps is None else float(args.market_impact_bps)
            ),
            fee_bps=0.0 if args.fee_bps is None else float(args.fee_bps),
            bid_ask_spread_bps=0.0
            if args.bid_ask_spread_bps is None
            else float(args.bid_ask_spread_bps),
            funding_bps_per_step=(
                0.0 if args.funding_bps_per_step is None else float(args.funding_bps_per_step)
            ),
            borrow_fee_bps_per_step=(
                0.0 if args.borrow_fee_bps_per_step is None else float(args.borrow_fee_bps_per_step)
            ),
        ),
        top_k=None if getattr(args, "top_k", None) is None else int(args.top_k),
    )


def cmd_show_portfolio_decisions(args: argparse.Namespace) -> int:
    cfg = load_runtime_config(db_path=args.db)
    store = EvaluationStore(cfg.db_path)
    try:
        store.ensure_schema()
        subject_set, _scoped_assets = _resolved_subject_set_scope(
            store,
            subject_set_id=args.subject_set_id,
        )
        decisions = _list_portfolio_decisions_for_subject_set(
            store,
            portfolio_id=None if args.portfolio_id is None else str(args.portfolio_id),
            subject_set=subject_set,
            target_id=None if args.target_id is None else str(args.target_id),
            aggregation_kind=(
                None if args.aggregation_kind is None else str(args.aggregation_kind)
            ),
            limit=int(args.limit),
        )
    finally:
        store.close()
    print(f"  DB:       {Path(cfg.db_path)}")
    if subject_set is not None:
        print(f"  SubjectSet: {subject_set.subject_set_id}")
        print("  Subjects: " + ", ".join(binding.subject_id for binding in subject_set.bindings))
    print_portfolio_decisions(decisions, show_details=bool(args.details))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {"init-db", "init"}:
            return cmd_init_db(args)
        if args.command == "register-signal-candidate":
            return cmd_register_signal(args)
        if args.command == "apply-manifest":
            return cmd_apply_runtime_manifest(args)
        if args.command == "run-diagnostic-evaluation":
            return cmd_run_diagnostic_evaluation(args)
        if args.command in {"list-runtime-manifests", "list-manifests"}:
            return cmd_list_runtime_manifests(args)
        if args.command == "inspect-runtime-resources":
            return cmd_inspect_runtime_resources(args)
        if args.command == "run-evaluation":
            return cmd_run_evaluation(args)
        if args.command in {"run-walk-forward-evaluation", "run-walk-forward"}:
            return cmd_run_walk_forward_evaluation(args)
        if args.command == "debug-register-signal-candidate-spec":
            return cmd_register_signal_spec(args)
        if args.command == "debug-show-signal-candidate-specs":
            return cmd_show_signal_specs(args)
        if args.command == "debug-register-observable":
            return cmd_register_observable(args)
        if args.command == "debug-show-observables":
            return cmd_show_observables(args)
        if args.command == "deactivate-signal-candidate":
            return cmd_deactivate_signal(args)
        if args.command == "activate-signal-candidate":
            return cmd_activate_signal(args)
        if args.command == "debug-record-prediction":
            return cmd_record_prediction(args)
        if args.command == "debug-finalize-observation":
            return cmd_finalize_observation(args)
        if args.command == "debug-update-state":
            return cmd_update_state(args)
        if args.command == "debug-generate-evaluation-input":
            return cmd_generate_evaluation_input(args)
        if args.command == "debug-generate-evaluation-inputs":
            return cmd_generate_evaluation_inputs(args)
        if args.command == "debug-apply-evaluation":
            return cmd_apply_evaluation(args)
        if args.command == "debug-apply-evaluations":
            return cmd_apply_evaluations(args)
        if args.command == "debug-apply-backfill":
            return cmd_apply_backfill(args)
        if args.command == "debug-apply-signal-candidates-backfill":
            return cmd_apply_signals_backfill(args)
        if args.command == "inspect-subject-set":
            return cmd_inspect_subject_set(args)
        if args.command == "debug-status":
            return cmd_status(args)
        if args.command == "debug-show-evaluations":
            return cmd_show_evaluations(args)
        if args.command == "debug-show-meta-predictions":
            return cmd_show_meta_predictions(args)
        if args.command == "debug-compare-meta-aggregations":
            return cmd_compare_meta_aggregations(args)
        if args.command == "debug-register-subject-set":
            return cmd_register_subject_set(args)
        if args.command == "debug-show-subject-sets":
            return cmd_show_subject_sets(args)
        if args.command == "check-subject-set-backend":
            return cmd_check_subject_set_backend(args)
        if args.command == "decide-portfolio":
            return cmd_decide_portfolio(args)
        if args.command == "debug-decide-portfolio-runtime":
            return cmd_debug_decide_portfolio_runtime(args)
        if args.command == "debug-show-portfolio-decisions":
            return cmd_show_portfolio_decisions(args)
    except ValueError as exc:
        parser.error(str(exc))
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
