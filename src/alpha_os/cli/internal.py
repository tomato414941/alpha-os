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
    print_observables,
    print_portfolio_decisions,
    print_signal_specs,
    print_subject_set_backend_checks,
    print_subject_sets,
)
from ..evaluation_application import (
    run_evaluation_use_case,
    run_walk_forward_evaluation_use_case,
)
from ..evaluation_spec import EvaluationSpec
from ..portfolio_construction_config import (
    PortfolioConstructionSizingSpec,
    PortfolioConstructionSpec,
)
from ..observables import ObservableDefinition
from ..config import (
    DEFAULT_SIGNAL_NOISE_BASE_URL,
    default_runtime_asset,
    load_runtime_config,
)
from ..signal_registry import (
    SignalSpec,
    build_signal_spec,
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
from ..store import EvaluationStore
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

    def internal_parser(name: str, **kwargs) -> argparse.ArgumentParser:
        kwargs["help"] = argparse.SUPPRESS
        return sub.add_parser(name, **kwargs)

    init_db = internal_parser("init-db")
    init_db.add_argument("--db", type=str, default=None)
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
    parser.add_argument("--turnover-cost-rate", type=float, default=None)
    parser.add_argument("--market-impact-bps", type=float, default=None)
    parser.add_argument("--fee-bps", type=float, default=None)
    parser.add_argument("--bid-ask-spread-bps", type=float, default=None)
    parser.add_argument("--funding-bps-per-step", type=float, default=None)
    parser.add_argument("--borrow-fee-bps-per-step", type=float, default=None)


@contextmanager
def _runtime_store(db_path: str | None) -> Iterator[tuple[object, EvaluationStore]]:
    cfg = load_runtime_config(db_path=db_path)
    store = EvaluationStore(cfg.db_path)
    try:
        yield cfg, store
    finally:
        store.close()


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
    trading_environment = (
        None if trading_strategy is None else trading_strategy.trading_environment
    )
    cost_inputs: list[CostInput] = []
    turnover_cost_rate = (
        trading_environment.turnover_cost_rate
        if args.turnover_cost_rate is None and trading_environment is not None
        else args.turnover_cost_rate
    )
    if turnover_cost_rate is not None:
        cost_inputs.append(
            CostInput(
                name="turnover_cost_rate",
                subject_id=None,
                value=float(turnover_cost_rate),
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
        active_weight_budget=base_construction.active_weight_budget,
        gross_exposure_cap=(
            base_construction.gross_exposure_cap
            if args.gross_exposure_cap is None
            else float(args.gross_exposure_cap)
        ),
        asset_class_weight_caps=dict(base_construction.asset_class_weight_caps),
        cluster_weight_caps=dict(base_construction.cluster_weight_caps),
        effective_n_floor=base_construction.effective_n_floor,
        top_gross_share_cap_n=base_construction.top_gross_share_cap_n,
        top_gross_share_cap=base_construction.top_gross_share_cap,
        concentration_min_abs_weight=base_construction.concentration_min_abs_weight,
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
        else trading_strategy.portfolio_construction.sizing_method
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
        active_weight_budget=base.active_weight_budget,
        gross_exposure_cap=base.gross_exposure_cap,
        asset_class_weight_caps=dict(base.asset_class_weight_caps),
        cluster_weight_caps=dict(base.cluster_weight_caps),
        effective_n_floor=base.effective_n_floor,
        top_gross_share_cap_n=base.top_gross_share_cap_n,
        top_gross_share_cap=base.top_gross_share_cap,
        concentration_min_abs_weight=base.concentration_min_abs_weight,
    )


def _portfolio_construction_for_decision_strategy(
    *,
    trading_strategy: TradingStrategySpec | None,
    base: PortfolioConstructionSpec | None,
) -> PortfolioConstructionSpec | None:
    if trading_strategy is None:
        return base
    construction = trading_strategy.portfolio_construction
    base_construction = PortfolioConstructionSpec() if base is None else base
    return PortfolioConstructionSpec(
        construction_kind=base_construction.construction_kind,
        sizing_policy=base_construction.sizing_policy,
        rebalance_interval_steps=trading_strategy.rebalance_interval_steps,
        long_only=construction.long_only,
        direction_mode=construction.direction_mode,
        active_weight_budget=base_construction.active_weight_budget,
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
        effective_n_floor=(
            base_construction.effective_n_floor
            if construction.effective_n_floor is None
            else construction.effective_n_floor
        ),
        top_gross_share_cap_n=(
            base_construction.top_gross_share_cap_n
            if construction.top_gross_share_cap_n is None
            else construction.top_gross_share_cap_n
        ),
        top_gross_share_cap=(
            base_construction.top_gross_share_cap
            if construction.top_gross_share_cap is None
            else construction.top_gross_share_cap
        ),
        concentration_min_abs_weight=construction.concentration_min_abs_weight,
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
    subject_set_id = None if args.subject_set_id is None else str(args.subject_set_id).strip()
    inline_subject_set = _subject_set_from_args(args)
    if subject_set_id and inline_subject_set is not None:
        raise ValueError("subject-set-id and subject-binding cannot be used together")
    if subject_set_id:
        state = store.get_subject_set(subject_set_id)
        if state is None:
            raise ValueError(f"unknown subject set: {subject_set_id}")
        subject_set = state.definition
    else:
        subject_set = inline_subject_set
    if subject_set is not None:
        validate_subject_set_universe_contract(subject_set)
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


def cmd_inspect_runtime_resources(args: argparse.Namespace) -> int:
    with _runtime_store(args.db) as (cfg, store):
        store.ensure_schema()
        observables = store.list_observables(limit=int(args.observable_limit))
        specifications = store.list_signal_specs(limit=int(args.signal_spec_limit))
        subject_sets = store.list_subject_sets(limit=int(args.subject_set_limit))
        signal_discoveries = store.list_signal_discovery_specs(
            limit=int(args.signal_discovery_limit)
        )
        evaluation_specs = store.list_evaluation_specs(limit=int(args.evaluation_spec_limit))
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
    evaluation_targets: tuple[tuple[str, str], ...]


def _group_evaluation_targets_by_signal_discovery(
    store: EvaluationStore,
    evaluation_targets: tuple[tuple[str, str], ...],
    *,
    base_url: str,
) -> tuple[_SignalDiscoveryEvaluationGroup, ...]:
    def strategy_lookup(strategy_id: str) -> TradingStrategySpec | None:
        strategy_state = store.get_trading_strategy(strategy_id)
        if strategy_state is None:
            return None
        return strategy_state.trading_strategy

    return _group_evaluation_targets_by_signal_discovery_with_strategy_lookup(
        evaluation_targets,
        strategy_lookup=strategy_lookup,
        base_url=base_url,
    )


def _group_evaluation_targets_by_signal_discovery_with_strategy_lookup(
    evaluation_targets: tuple[tuple[str, str], ...],
    *,
    strategy_lookup,
    base_url: str,
) -> tuple[_SignalDiscoveryEvaluationGroup, ...]:
    grouped: dict[str | None, list[tuple[str, str]]] = {}
    for evaluation_case in evaluation_targets:
        trading_strategy = strategy_lookup(_target_strategy_id(evaluation_case))
        if trading_strategy is None:
            raise ValueError(
                f"evaluation target strategy does not exist: {_target_strategy_id(evaluation_case)}"
            )
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
                evaluation_targets=tuple(grouped_cases),
            )
        )
    return tuple(groups)


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


def _format_runtime_strategy_summary(
    trading_strategy: TradingStrategySpec,
    *,
    sizing_method: str,
    sizing_engine: str,
) -> str:
    construction = trading_strategy.portfolio_construction
    return (
        f"{trading_strategy.strategy_id} "
        f"selection={trading_strategy.selection_kind} "
        f"sizing={sizing_method} "
        f"engine={sizing_engine} "
        f"rebalance=every_{trading_strategy.rebalance_interval_steps}_steps "
        f"top_k={'-' if trading_strategy.top_k is None else trading_strategy.top_k} "
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
        top_k=None if trading_strategy is None else trading_strategy.top_k,
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
        if args.command == "apply-manifest":
            return cmd_apply_runtime_manifest(args)
        if args.command in {"list-runtime-manifests", "list-manifests"}:
            return cmd_list_runtime_manifests(args)
        if args.command == "inspect-runtime-resources":
            return cmd_inspect_runtime_resources(args)
        if args.command == "run-evaluation":
            return cmd_run_evaluation(args)
        if args.command in {"run-walk-forward-evaluation", "run-walk-forward"}:
            return cmd_run_walk_forward_evaluation(args)
        if args.command == "debug-show-signal-candidate-specs":
            return cmd_show_signal_specs(args)
        if args.command == "debug-register-observable":
            return cmd_register_observable(args)
        if args.command == "debug-show-observables":
            return cmd_show_observables(args)
        if args.command == "debug-register-subject-set":
            return cmd_register_subject_set(args)
        if args.command == "debug-show-subject-sets":
            return cmd_show_subject_sets(args)
        if args.command == "check-subject-set-backend":
            return cmd_check_subject_set_backend(args)
        if args.command == "decide-portfolio":
            return cmd_decide_portfolio(args)
        if args.command == "debug-show-portfolio-decisions":
            return cmd_show_portfolio_decisions(args)
    except ValueError as exc:
        parser.error(str(exc))
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
