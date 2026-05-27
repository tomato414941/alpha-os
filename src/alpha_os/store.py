from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import DEFAULT_SUBJECT_ID, DEFAULT_TARGET, default_runtime_asset
from .compression import CompressedBelief
from .evaluation_spec import EvaluationSpec
from .evaluation_run_result import EvaluationRunResult
from .signal_registry import (
    SignalSpec,
    find_signal_spec,
)
from .signal_discovery import SignalDiscoverySpec
from .observables import (
    ObservableDefinition,
    find_observable_definition,
    list_observable_definitions,
)
from .portfolio_decision import (
    InstrumentSpec,
    ObservationSpec,
    SubjectObservationBinding,
    SubjectSet,
    UniversePolicySpec,
)
from .screening import ScreeningResult
from .trading_strategy import TradingStrategySpec
from .targets import TargetDefinition, find_target_definition, list_target_definitions


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, init=False)
class SignalSpecState:
    signal_id: str
    definition_json: str
    target_id: str

    def __init__(
        self,
        *,
        signal_id: str | None = None,

        definition_json: str,
        target_id: str,
    ) -> None:
        resolved_signal_id = signal_id
        if resolved_signal_id is None:
            raise ValueError("signal spec state requires signal_id")
        object.__setattr__(self, "signal_id", str(resolved_signal_id))
        object.__setattr__(self, "definition_json", definition_json)
        object.__setattr__(self, "target_id", target_id)
    @property
    def definition(self) -> SignalSpec:
        return SignalSpec.from_document(
            signal_id=self.signal_id,
            document=json.loads(self.definition_json),
        )

    @property
    def kind(self) -> str:
        return self.definition.kind

    @property
    def lookback(self) -> int:
        return self.definition.lookback

    @property
    def horizon_days(self) -> int | None:
        return self.definition.horizon_days


@dataclass(frozen=True)
class ObservationRecord:
    evaluation_id: str
    subject_id: str
    asset: str
    target_id: str
    value: float
    recorded_at: str


@dataclass(frozen=True)
class TargetState:
    target_id: str
    definition_json: str

    @property
    def definition(self) -> TargetDefinition:
        return TargetDefinition.from_document(json.loads(self.definition_json))


@dataclass(frozen=True)
class ObservableState:
    observable_id: str
    definition_json: str

    @property
    def definition(self) -> ObservableDefinition:
        return ObservableDefinition.from_document(json.loads(self.definition_json))


@dataclass(frozen=True)
class SubjectSetState:
    subject_set_id: str
    definition_json: str

    @property
    def definition(self) -> SubjectSet:
        document = json.loads(self.definition_json)
        instruments = tuple(
            InstrumentSpec(
                instrument_id=str(item["instrument_id"]),
                instrument_type=str(item["instrument_type"]),
                asset=str(item["asset"]),
                venue=None if item.get("venue") is None else str(item.get("venue")),
                quote_ccy=(
                    None if item.get("quote_ccy") is None else str(item.get("quote_ccy"))
                ),
                collateral_ccy=(
                    None
                    if item.get("collateral_ccy") is None
                    else str(item.get("collateral_ccy"))
                ),
                contract_family=(
                    None
                    if item.get("contract_family") is None
                    else str(item.get("contract_family"))
                ),
                underlying_id=(
                    None
                    if item.get("underlying_id") is None
                    else str(item.get("underlying_id"))
                ),
                asset_class=(
                    None if item.get("asset_class") is None else str(item.get("asset_class"))
                ),
                region=(
                    None if item.get("region") is None else str(item.get("region"))
                ),
                liquidity_tier=(
                    None
                    if item.get("liquidity_tier") is None
                    else str(item.get("liquidity_tier"))
                ),
                cluster=(
                    None if item.get("cluster") is None else str(item.get("cluster"))
                ),
                expiry=None if item.get("expiry") is None else str(item.get("expiry")),
                roll_rule=(
                    None if item.get("roll_rule") is None else str(item.get("roll_rule"))
                ),
                multiplier=(
                    None if item.get("multiplier") is None else float(item.get("multiplier"))
                ),
                margin_model=(
                    None if item.get("margin_model") is None else str(item.get("margin_model"))
                ),
            )
            for item in document.get("instruments", [])
            if isinstance(item, dict)
            and isinstance(item.get("instrument_id"), str)
            and isinstance(item.get("instrument_type"), str)
            and isinstance(item.get("asset"), str)
        )
        observation_specs = tuple(
            ObservationSpec(
                observation_spec_id=str(item["observation_spec_id"]),
                observable_id=str(item["observable_id"]),
                adapter_kind=str(
                    item.get("adapter_kind", "signal_noise_asset_observable")
                ),
                source_id=str(item.get("source_id", "signal_noise")),
                resolution=str(item.get("resolution", "1d")),
                provided_observable_ids=tuple(
                    str(value)
                    for value in item.get("provided_observable_ids", [])
                    if isinstance(value, str) and value
                ),
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
            for item in document.get("observation_specs", [])
            if isinstance(item, dict)
            and isinstance(item.get("observation_spec_id"), str)
            and isinstance(item.get("observable_id"), str)
        )
        bindings = tuple(
            SubjectObservationBinding(
                subject_id=str(item["subject_id"]),
                asset=str(item["asset"]),
                observation_spec_id=str(item["observation_spec_id"]),
                subject_kind=str(item.get("subject_kind", "asset")),
                instrument_id=(
                    None if item.get("instrument_id") is None else str(item.get("instrument_id"))
                ),
            )
            for item in document.get("bindings", [])
            if isinstance(item, dict)
            and isinstance(item.get("subject_id"), str)
            and isinstance(item.get("asset"), str)
            and isinstance(item.get("observation_spec_id"), str)
        )
        if not observation_specs and bindings:
            legacy_bindings = tuple(
                item
                for item in document.get("bindings", [])
                if isinstance(item, dict)
                and isinstance(item.get("subject_id"), str)
                and isinstance(item.get("asset"), str)
                and isinstance(item.get("signal_name"), str)
            )
            observation_specs = tuple(
                ObservationSpec(
                    observation_spec_id=f"{str(item['subject_id'])}__legacy",
                    observable_id="daily_close",
                    adapter_kind="signal_noise_asset_observable",
                )
                for item in legacy_bindings
            )
            bindings = tuple(
                SubjectObservationBinding(
                    subject_id=str(item["subject_id"]),
                    asset=str(item["asset"]),
                    observation_spec_id=f"{str(item['subject_id'])}__legacy",
                    subject_kind="asset",
                )
                for item in legacy_bindings
            )
        return SubjectSet(
            subject_set_id=self.subject_set_id,
            instruments=instruments,
            observation_specs=observation_specs,
            bindings=bindings,
            universe_policy=UniversePolicySpec(
                base_currency=(
                    None
                    if not isinstance(document.get("universe_policy"), dict)
                    or document["universe_policy"].get("base_currency") is None
                    else str(document["universe_policy"].get("base_currency"))
                ),
                trading_calendar=(
                    None
                    if not isinstance(document.get("universe_policy"), dict)
                    or document["universe_policy"].get("trading_calendar") is None
                    else str(document["universe_policy"].get("trading_calendar"))
                ),
                benchmark_id=(
                    None
                    if not isinstance(document.get("universe_policy"), dict)
                    or document["universe_policy"].get("benchmark_id") is None
                    else str(document["universe_policy"].get("benchmark_id"))
                ),
            ),
        )


@dataclass(frozen=True)
class SignalDiscoverySpecState:
    signal_discovery_id: str
    definition_json: str

    @property
    def definition(self) -> SignalDiscoverySpec:
        return SignalDiscoverySpec.from_document(
            signal_discovery_id=self.signal_discovery_id,
            document=json.loads(self.definition_json),
        )


@dataclass(frozen=True)
class EvaluationSpecState:
    evaluation_spec_id: str
    definition_json: str

    @property
    def definition(self) -> EvaluationSpec:
        return EvaluationSpec.from_document(json.loads(self.definition_json))


@dataclass(frozen=True)
class ScreeningResultState:
    screening_result_id: str
    signal_discovery_id: str
    result_json: str
    created_at: str

    @property
    def result(self) -> ScreeningResult:
        return ScreeningResult.from_document(
            screening_result_id=self.screening_result_id,
            document=json.loads(self.result_json),
        )


@dataclass(frozen=True)
class CompressedBeliefState:
    compressed_belief_id: str
    signal_discovery_id: str
    belief_json: str
    created_at: str

    @property
    def belief(self) -> CompressedBelief:
        return CompressedBelief.from_document(
            compressed_belief_id=self.compressed_belief_id,
            document=json.loads(self.belief_json),
        )


@dataclass(frozen=True)
class TradingStrategyState:
    strategy_id: str
    spec_json: str
    created_at: str

    @property
    def trading_strategy(self) -> TradingStrategySpec:
        return TradingStrategySpec.from_document(
            json.loads(self.spec_json)["trading_strategy"]
        )


@dataclass(frozen=True)
class EvaluationRunResultState:
    evaluation_run_result_id: str
    evaluation_spec_id: str
    run_result_json: str
    created_at: str

    @property
    def run_result(self) -> EvaluationRunResult:
        return EvaluationRunResult.from_document(
            evaluation_run_result_id=self.evaluation_run_result_id,
            document=json.loads(self.run_result_json),
        )


@dataclass(frozen=True)
class PortfolioDecisionState:
    portfolio_id: str
    subject_id: str
    target_id: str
    aggregation_kind: str
    as_of: str
    target_weight: float
    position_delta: float
    target_notional: float | None
    target_quantity: float | None
    entry_allowed: bool
    risk_scale: float
    details_json: str | None
    created_at: str
    updated_at: str

    @property
    def details(self) -> dict[str, Any] | None:
        if self.details_json is None:
            return None
        return json.loads(self.details_json)


def _row_to_signal_spec(
    row: sqlite3.Row | None,
) -> SignalSpecState | None:
    if row is None:
        return None
    return SignalSpecState(
        signal_id=str(row["signal_id"]),
        definition_json=str(row["definition_json"]),
        target_id=str(row["target_id"]),
    )


def _row_to_signal_discovery_spec(
    row: sqlite3.Row | None,
) -> SignalDiscoverySpecState | None:
    if row is None:
        return None
    return SignalDiscoverySpecState(
        signal_discovery_id=str(row["signal_discovery_id"]),
        definition_json=str(row["definition_json"]),
    )


def _row_to_evaluation_spec(
    row: sqlite3.Row | None,
) -> EvaluationSpecState | None:
    if row is None:
        return None
    return EvaluationSpecState(
        evaluation_spec_id=str(row["evaluation_spec_id"]),
        definition_json=str(row["definition_json"]),
    )


def _row_to_observation(row: sqlite3.Row | None) -> ObservationRecord | None:
    if row is None:
        return None
    return ObservationRecord(
        evaluation_id=str(row["evaluation_id"]),
        subject_id=(
            str(row["subject_id"])
            if "subject_id" in row.keys() and row["subject_id"] is not None
            else str(row["asset"])
        ),
        asset=str(row["asset"]),
        target_id=str(row["target_id"]),
        value=float(row["value"]),
        recorded_at=str(row["recorded_at"]),
    )


def _row_to_target(row: sqlite3.Row | None) -> TargetState | None:
    if row is None:
        return None
    return TargetState(
        target_id=str(row["target_id"]),
        definition_json=str(row["definition_json"]),
    )


def _row_to_observable(row: sqlite3.Row | None) -> ObservableState | None:
    if row is None:
        return None
    return ObservableState(
        observable_id=str(row["observable_id"]),
        definition_json=str(row["definition_json"]),
    )


def _row_to_observable(row: sqlite3.Row | None) -> ObservableState | None:
    if row is None:
        return None
    return ObservableState(
        observable_id=str(row["observable_id"]),
        definition_json=str(row["definition_json"]),
    )


def _row_to_subject_set(row: sqlite3.Row | None) -> SubjectSetState | None:
    if row is None:
        return None
    return SubjectSetState(
        subject_set_id=str(row["subject_set_id"]),
        definition_json=str(row["definition_json"]),
    )


def _row_to_portfolio_decision(row: sqlite3.Row | None) -> PortfolioDecisionState | None:
    if row is None:
        return None
    return PortfolioDecisionState(
        portfolio_id=str(row["portfolio_id"]),
        subject_id=str(row["subject_id"]),
        target_id=str(row["target_id"]),
        aggregation_kind=str(row["aggregation_kind"]),
        as_of=str(row["as_of"]),
        target_weight=float(row["target_weight"]),
        position_delta=float(row["position_delta"]),
        target_notional=None
        if row["target_notional"] is None
        else float(row["target_notional"]),
        target_quantity=None
        if row["target_quantity"] is None
        else float(row["target_quantity"]),
        entry_allowed=bool(int(row["entry_allowed"])),
        risk_scale=float(row["risk_scale"]),
        details_json=None if row["details_json"] is None else str(row["details_json"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_evaluation_run_result(
    row: sqlite3.Row | None,
) -> EvaluationRunResultState | None:
    if row is None:
        return None
    return EvaluationRunResultState(
        evaluation_run_result_id=str(row["evaluation_run_result_id"]),
        evaluation_spec_id=str(row["evaluation_spec_id"]),
        run_result_json=str(row["run_result_json"]),
        created_at=str(row["created_at"]),
    )


def _row_to_screening_result(row: sqlite3.Row | None) -> ScreeningResultState | None:
    if row is None:
        return None
    return ScreeningResultState(
        screening_result_id=str(row["screening_result_id"]),
        signal_discovery_id=str(row["signal_discovery_id"]),
        result_json=str(row["result_json"]),
        created_at=str(row["created_at"]),
    )


def _row_to_compressed_belief(row: sqlite3.Row | None) -> CompressedBeliefState | None:
    if row is None:
        return None
    return CompressedBeliefState(
        compressed_belief_id=str(row["compressed_belief_id"]),
        signal_discovery_id=str(row["signal_discovery_id"]),
        belief_json=str(row["belief_json"]),
        created_at=str(row["created_at"]),
    )


def _row_to_trading_strategy(row: sqlite3.Row | None) -> TradingStrategyState | None:
    if row is None:
        return None
    return TradingStrategyState(
        strategy_id=str(row["strategy_id"]),
        spec_json=str(row["spec_json"]),
        created_at=str(row["created_at"]),
    )


class EvaluationStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS targets (
                target_id TEXT PRIMARY KEY,
                definition_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observables (
                observable_id TEXT PRIMARY KEY,
                definition_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS subject_sets (
                subject_set_id TEXT PRIMARY KEY,
                definition_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS signal_discoveries (
                signal_discovery_id TEXT PRIMARY KEY,
                definition_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_specs (
                evaluation_spec_id TEXT PRIMARY KEY,
                definition_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS signal_specs (
                signal_id TEXT PRIMARY KEY,
                target_id TEXT NOT NULL,
                definition_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                evaluation_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                target_id TEXT NOT NULL,
                value REAL NOT NULL,
                recorded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observation_frame_cache (
                cache_key TEXT PRIMARY KEY,
                frame_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS screening_results (
                screening_result_id TEXT PRIMARY KEY,
                signal_discovery_id TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS compressed_beliefs (
                compressed_belief_id TEXT PRIMARY KEY,
                signal_discovery_id TEXT NOT NULL,
                belief_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS strategy_specs (
                strategy_id TEXT PRIMARY KEY,
                spec_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_run_results (
                evaluation_run_result_id TEXT PRIMARY KEY,
                evaluation_spec_id TEXT NOT NULL,
                run_result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS portfolio_decisions (
                portfolio_id TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                aggregation_kind TEXT NOT NULL,
                as_of TEXT NOT NULL,
                target_weight REAL NOT NULL,
                position_delta REAL NOT NULL,
                target_notional REAL,
                target_quantity REAL,
                entry_allowed INTEGER NOT NULL,
                risk_scale REAL NOT NULL,
                details_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (portfolio_id, subject_id, target_id, aggregation_kind, as_of)
            );
            """
        )
        self._seed_builtin_targets()
        self._seed_builtin_observables()
        self._ensure_subject_first_runtime_schema()
        self._ensure_signal_spec_schema()
        self.conn.commit()

    def _ensure_subject_first_runtime_schema(self) -> None:
        table_columns = {
            "observations": {
                str(row["name"])
                for row in self.conn.execute("PRAGMA table_info(observations)").fetchall()
            }
        }
        required_columns = {
            "observations": "TEXT NOT NULL DEFAULT ''",
        }
        for key, definition in required_columns.items():
            table_name = key
            if "subject_id" in table_columns[table_name]:
                continue
            self.conn.execute(
                f"""
                ALTER TABLE {table_name}
                ADD COLUMN subject_id {definition}
                """
            )
        self._backfill_runtime_subject_ids()

    def _ensure_signal_spec_schema(self) -> None:
        self._seed_builtin_signal_specs()

    def _backfill_runtime_subject_ids(self) -> None:
        self.conn.execute(
            """
            UPDATE observations
            SET subject_id = asset
            WHERE subject_id = ''
            """
        )

    def _seed_builtin_targets(self) -> None:
        timestamp = _utc_now()
        for definition in list_target_definitions():
            self.conn.execute(
                """
                INSERT INTO targets (target_id, definition_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(target_id) DO NOTHING
                """,
                (
                    definition.target_id,
                    json.dumps(definition.to_document(), sort_keys=True),
                    timestamp,
                    timestamp,
                ),
            )

    def _seed_builtin_observables(self) -> None:
        timestamp = _utc_now()
        for definition in list_observable_definitions():
            self.conn.execute(
                """
                INSERT INTO observables (observable_id, definition_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(observable_id) DO NOTHING
                """,
                (
                    definition.observable_id,
                    json.dumps(definition.to_document(), sort_keys=True),
                    timestamp,
                    timestamp,
                ),
            )

    def _seed_builtin_signal_specs(self) -> None:
        timestamp = _utc_now()
        specification_ids = (
            "momentum_1d",
            "momentum_3d",
            "momentum_5d",
            "reversal_1d",
            "reversal_3d",
            "reversal_5d",
            "average_gap_3d",
            "average_gap_5d",
            "range_position_5d",
        )
        for signal_id in specification_ids:
            definition = find_signal_spec(signal_id)
            if definition is None:
                continue
            self.conn.execute(
                """
                INSERT INTO signal_specs (
                    signal_id, target_id, definition_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(signal_id) DO NOTHING
                """,
                (
                    definition.signal_id,
                    definition.target_id,
                    json.dumps(definition.to_document(), sort_keys=True),
                    timestamp,
                    timestamp,
                ),
            )

    def get_target(self, target_id: str) -> TargetState | None:
        row = self.conn.execute(
            """
            SELECT target_id, definition_json
            FROM targets
            WHERE target_id = ?
            """,
            (target_id,),
        ).fetchone()
        return _row_to_target(row)

    def get_observable(self, observable_id: str) -> ObservableState | None:
        row = self.conn.execute(
            """
            SELECT observable_id, definition_json
            FROM observables
            WHERE observable_id = ?
            """,
            (observable_id,),
        ).fetchone()
        return _row_to_observable(row)

    def get_subject_set(self, subject_set_id: str) -> SubjectSetState | None:
        row = self.conn.execute(
            """
            SELECT subject_set_id, definition_json
            FROM subject_sets
            WHERE subject_set_id = ?
            """,
            (subject_set_id,),
        ).fetchone()
        return _row_to_subject_set(row)

    def get_signal_discovery_spec(
        self,
        signal_discovery_id: str,
    ) -> SignalDiscoverySpecState | None:
        row = self.conn.execute(
            """
            SELECT signal_discovery_id, definition_json
            FROM signal_discoveries
            WHERE signal_discovery_id = ?
            """,
            (signal_discovery_id,),
        ).fetchone()
        return _row_to_signal_discovery_spec(row)

    def get_evaluation_spec(
        self,
        evaluation_spec_id: str,
    ) -> EvaluationSpecState | None:
        row = self.conn.execute(
            """
            SELECT evaluation_spec_id, definition_json
            FROM evaluation_specs
            WHERE evaluation_spec_id = ?
            """,
            (evaluation_spec_id,),
        ).fetchone()
        return _row_to_evaluation_spec(row)

    def get_observation_frame_cache(self, cache_key: str) -> str | None:
        row = self.conn.execute(
            """
            SELECT frame_json
            FROM observation_frame_cache
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def list_observables(self, *, limit: int = 100) -> list[ObservableState]:
        rows = self.conn.execute(
            """
            SELECT observable_id, definition_json
            FROM observables
            ORDER BY observable_id ASC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ).fetchall()
        return [_row_to_observable(row) for row in rows if row is not None]

    def get_signal_spec(self, signal_id: str) -> SignalSpecState | None:
        row = self.conn.execute(
            """
            SELECT signal_id, target_id, definition_json
            FROM signal_specs
            WHERE signal_id = ?
            """,
            (signal_id,),
        ).fetchone()
        return _row_to_signal_spec(row)

    def list_signal_specs(self, *, limit: int = 100) -> list[SignalSpecState]:
        rows = self.conn.execute(
            """
            SELECT signal_id, target_id, definition_json
            FROM signal_specs
            ORDER BY signal_id ASC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ).fetchall()
        return [_row_to_signal_spec(row) for row in rows if row is not None]

    def upsert_subject_set(
        self,
        subject_set_id: str,
        *,
        definition: SubjectSet,
        recorded_at: str | None = None,
    ) -> SubjectSetState:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        definition_json = json.dumps(
            {
                "instruments": [
                    {
                        "instrument_id": item.instrument_id,
                        "instrument_type": item.instrument_type,
                        "asset": item.asset,
                        "venue": item.venue,
                        "quote_ccy": item.quote_ccy,
                        "contract_family": item.contract_family,
                        "asset_class": item.asset_class,
                        "region": item.region,
                        "liquidity_tier": item.liquidity_tier,
                        "cluster": item.cluster,
                        "expiry": item.expiry,
                        "roll_rule": item.roll_rule,
                        "multiplier": item.multiplier,
                        "margin_model": item.margin_model,
                    }
                    for item in definition.instruments
                ],
                "observation_specs": [
                    {
                        "observation_spec_id": item.observation_spec_id,
                        "observable_id": item.observable_id,
                        "adapter_kind": item.adapter_kind,
                        "source_id": item.source_id,
                        "resolution": item.resolution,
                        "provided_observable_ids": list(item.provided_observable_ids),
                    }
                    for item in definition.observation_specs
                ],
                "bindings": [
                    {
                        "subject_id": item.subject_id,
                        "subject_kind": item.subject_kind,
                        "asset": item.asset,
                        "observation_spec_id": item.observation_spec_id,
                        "instrument_id": item.instrument_id,
                    }
                    for item in definition.bindings
                ],
                "universe_policy": {
                    "base_currency": definition.universe_policy.base_currency,
                    "trading_calendar": definition.universe_policy.trading_calendar,
                    "benchmark_id": definition.universe_policy.benchmark_id,
                },
            },
            sort_keys=True,
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO subject_sets (
                    subject_set_id, definition_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(subject_set_id) DO UPDATE SET
                    definition_json = excluded.definition_json,
                    updated_at = excluded.updated_at
                """,
                (
                    subject_set_id,
                    definition_json,
                    timestamp,
                    timestamp,
                ),
            )
        state = self.get_subject_set(subject_set_id)
        assert state is not None
        return state

    def upsert_signal_discovery_spec(
        self,
        signal_discovery_id: str,
        *,
        definition: SignalDiscoverySpec,
        recorded_at: str | None = None,
    ) -> SignalDiscoverySpecState:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        definition_json = json.dumps(definition.to_document(), sort_keys=True)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO signal_discoveries (
                    signal_discovery_id, definition_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(signal_discovery_id) DO UPDATE SET
                    definition_json = excluded.definition_json,
                    updated_at = excluded.updated_at
                """,
                (
                    signal_discovery_id,
                    definition_json,
                    timestamp,
                    timestamp,
                ),
            )
        state = self.get_signal_discovery_spec(signal_discovery_id)
        assert state is not None
        return state

    def upsert_evaluation_spec(
        self,
        evaluation_spec_id: str,
        *,
        definition: EvaluationSpec,
        recorded_at: str | None = None,
    ) -> EvaluationSpecState:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        definition_json = json.dumps(definition.to_document(), sort_keys=True)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO evaluation_specs (
                    evaluation_spec_id, definition_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(evaluation_spec_id) DO UPDATE SET
                    definition_json = excluded.definition_json,
                    updated_at = excluded.updated_at
                """,
                (
                    evaluation_spec_id,
                    definition_json,
                    timestamp,
                    timestamp,
                ),
            )
        state = self.get_evaluation_spec(evaluation_spec_id)
        assert state is not None
        return state

    def upsert_observation_frame_cache(
        self,
        cache_key: str,
        *,
        frame_json: str,
        recorded_at: str | None = None,
    ) -> None:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO observation_frame_cache (
                    cache_key, frame_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    frame_json = excluded.frame_json,
                    updated_at = excluded.updated_at
                """,
                (
                    cache_key,
                    frame_json,
                    timestamp,
                    timestamp,
                ),
            )

    def list_subject_sets(self, *, limit: int = 20) -> list[SubjectSetState]:
        rows = self.conn.execute(
            """
            SELECT subject_set_id, definition_json
            FROM subject_sets
            ORDER BY subject_set_id ASC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ).fetchall()
        return [_row_to_subject_set(row) for row in rows if row is not None]

    def list_signal_discovery_specs(
        self,
        *,
        limit: int = 20,
    ) -> list[SignalDiscoverySpecState]:
        rows = self.conn.execute(
            """
            SELECT signal_discovery_id, definition_json
            FROM signal_discoveries
            ORDER BY signal_discovery_id ASC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ).fetchall()
        return [
            _row_to_signal_discovery_spec(row)
            for row in rows
            if row is not None
        ]

    def list_evaluation_specs(
        self,
        *,
        limit: int = 20,
    ) -> list[EvaluationSpecState]:
        rows = self.conn.execute(
            """
            SELECT evaluation_spec_id, definition_json
            FROM evaluation_specs
            ORDER BY evaluation_spec_id ASC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ).fetchall()
        return [_row_to_evaluation_spec(row) for row in rows if row is not None]

    def register_target(
        self,
        target_id: str,
        *,
        definition: TargetDefinition | None = None,
        recorded_at: str | None = None,
    ) -> TargetState:
        self.ensure_schema()
        existing = self.get_target(target_id)
        if existing is not None:
            return existing

        resolved_definition = definition or find_target_definition(target_id)
        if resolved_definition is None:
            raise ValueError(f"target definition must exist before use: {target_id}")
        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO targets (target_id, definition_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    resolved_definition.target_id,
                    json.dumps(resolved_definition.to_document(), sort_keys=True),
                    timestamp,
                    timestamp,
                ),
            )
        target_state = self.get_target(target_id)
        assert target_state is not None
        return target_state

    def register_observable(
        self,
        observable_id: str,
        *,
        definition: ObservableDefinition | None = None,
        recorded_at: str | None = None,
    ) -> tuple[ObservableState, bool]:
        self.ensure_schema()
        existing = self.get_observable(observable_id)
        if existing is not None:
            return existing, False

        resolved_definition = definition or find_observable_definition(observable_id)
        if resolved_definition is None:
            raise ValueError(
                f"observable definition must exist before use: {observable_id}"
            )
        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO observables (observable_id, definition_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    resolved_definition.observable_id,
                    json.dumps(resolved_definition.to_document(), sort_keys=True),
                    timestamp,
                    timestamp,
                ),
            )
        state = self.get_observable(observable_id)
        assert state is not None
        return state, True

    def register_signal_spec(
        self,
        signal_id: str,
        *,
        definition: SignalSpec | None = None,
        recorded_at: str | None = None,
    ) -> tuple[SignalSpecState, bool]:
        self.ensure_schema()
        existing = self.get_signal_spec(signal_id)
        if existing is not None:
            return existing, False

        resolved_definition = definition or find_signal_spec(signal_id)
        if resolved_definition is None:
            raise ValueError(
                f"signal spec definition must exist before use: {signal_id}"
            )
        timestamp = recorded_at or _utc_now()
        self.register_target(
            resolved_definition.target_id,
            definition=resolved_definition.target,
            recorded_at=timestamp,
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO signal_specs (
                    signal_id, target_id, definition_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    resolved_definition.signal_id,
                    resolved_definition.target_id,
                    json.dumps(resolved_definition.to_document(), sort_keys=True),
                    timestamp,
                    timestamp,
                ),
            )
        state = self.get_signal_spec(signal_id)
        assert state is not None
        return state, True

    def upsert_portfolio_decision(
        self,
        *,
        portfolio_id: str,
        subject_id: str,
        target_id: str,
        aggregation_kind: str,
        as_of: str,
        target_weight: float,
        position_delta: float,
        target_notional: float | None,
        target_quantity: float | None,
        entry_allowed: bool,
        risk_scale: float,
        details_json: str | None,
        recorded_at: str | None = None,
    ) -> None:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO portfolio_decisions (
                    portfolio_id, subject_id, target_id, aggregation_kind,
                    as_of, target_weight, position_delta, target_notional,
                    target_quantity, entry_allowed, risk_scale, details_json,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(portfolio_id, subject_id, target_id, aggregation_kind, as_of)
                DO UPDATE SET
                    target_weight = excluded.target_weight,
                    position_delta = excluded.position_delta,
                    target_notional = excluded.target_notional,
                    target_quantity = excluded.target_quantity,
                    entry_allowed = excluded.entry_allowed,
                    risk_scale = excluded.risk_scale,
                    details_json = excluded.details_json,
                    updated_at = excluded.updated_at
                """,
                (
                    portfolio_id,
                    subject_id,
                    target_id,
                    aggregation_kind,
                    as_of,
                    float(target_weight),
                    float(position_delta),
                    None if target_notional is None else float(target_notional),
                    None if target_quantity is None else float(target_quantity),
                    int(bool(entry_allowed)),
                    float(risk_scale),
                    details_json,
                    timestamp,
                    timestamp,
                ),
            )

    def list_portfolio_decisions(
        self,
        *,
        portfolio_id: str | None = None,
        target_id: str | None = None,
        aggregation_kind: str | None = None,
        limit: int = 20,
    ) -> list[PortfolioDecisionState]:
        filters: list[str] = []
        params: list[Any] = []
        if portfolio_id is not None:
            filters.append("portfolio_id = ?")
            params.append(portfolio_id)
        if target_id is not None:
            filters.append("target_id = ?")
            params.append(target_id)
        if aggregation_kind is not None:
            filters.append("aggregation_kind = ?")
            params.append(aggregation_kind)
        where_clause = ""
        if filters:
            where_clause = f"WHERE {' AND '.join(filters)}"
        params.append(max(int(limit), 1))
        rows = self.conn.execute(
            f"""
            SELECT portfolio_id, subject_id, target_id, aggregation_kind,
                   as_of, target_weight, position_delta, target_notional,
                   target_quantity, entry_allowed, risk_scale, details_json,
                   created_at, updated_at
            FROM portfolio_decisions
            {where_clause}
            ORDER BY updated_at DESC, as_of DESC, subject_id ASC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [_row_to_portfolio_decision(row) for row in rows if row is not None]

    def get_latest_portfolio_decisions(
        self,
        *,
        portfolio_id: str,
        aggregation_kind: str,
    ) -> list[PortfolioDecisionState]:
        row = self.conn.execute(
            """
            SELECT MAX(as_of) AS latest_as_of
            FROM portfolio_decisions
            WHERE portfolio_id = ? AND aggregation_kind = ?
            """,
            (portfolio_id, aggregation_kind),
        ).fetchone()
        if row is None or row["latest_as_of"] is None:
            return []
        latest_as_of = str(row["latest_as_of"])
        rows = self.conn.execute(
            """
            SELECT portfolio_id, subject_id, target_id, aggregation_kind,
                   as_of, target_weight, position_delta, target_notional,
                   target_quantity, entry_allowed, risk_scale, details_json,
                   created_at, updated_at
            FROM portfolio_decisions
            WHERE portfolio_id = ? AND aggregation_kind = ? AND as_of = ?
            ORDER BY subject_id ASC
            """,
            (portfolio_id, aggregation_kind, latest_as_of),
        ).fetchall()
        return [_row_to_portfolio_decision(row) for row in rows if row is not None]

    def upsert_screening_result(
        self,
        *,
        result: ScreeningResult,
    ) -> ScreeningResultState:
        self.ensure_schema()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO screening_results (
                    screening_result_id, signal_discovery_id, result_json, created_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(screening_result_id) DO UPDATE SET
                    signal_discovery_id = excluded.signal_discovery_id,
                    result_json = excluded.result_json,
                    created_at = excluded.created_at
                """,
                (
                    result.screening_result_id,
                    result.signal_discovery_id,
                    json.dumps(result.to_document(), sort_keys=True),
                    result.created_at,
                ),
            )
        state = self.get_screening_result(result.screening_result_id)
        assert state is not None
        return state

    def get_screening_result(
        self,
        screening_result_id: str,
    ) -> ScreeningResultState | None:
        row = self.conn.execute(
            """
            SELECT screening_result_id, signal_discovery_id, result_json, created_at
            FROM screening_results
            WHERE screening_result_id = ?
            """,
            (screening_result_id,),
        ).fetchone()
        return _row_to_screening_result(row)

    def list_screening_results(
        self,
        *,
        signal_discovery_id: str | None = None,
        limit: int = 20,
    ) -> list[ScreeningResultState]:
        if signal_discovery_id is None:
            rows = self.conn.execute(
                """
                SELECT screening_result_id, signal_discovery_id, result_json, created_at
                FROM screening_results
                ORDER BY created_at DESC, screening_result_id DESC
                LIMIT ?
                """,
                (max(int(limit), 1),),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT screening_result_id, signal_discovery_id, result_json, created_at
                FROM screening_results
                WHERE signal_discovery_id = ?
                ORDER BY created_at DESC, screening_result_id DESC
                LIMIT ?
                """,
                (
                    signal_discovery_id,
                    max(int(limit), 1),
                ),
            ).fetchall()
        return [
            _row_to_screening_result(row)
            for row in rows
            if row is not None
        ]

    def upsert_compressed_belief(
        self,
        *,
        signal_discovery_id: str,
        belief: CompressedBelief,
    ) -> CompressedBeliefState:
        self.ensure_schema()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO compressed_beliefs (
                    compressed_belief_id, signal_discovery_id, belief_json, created_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(compressed_belief_id) DO UPDATE SET
                    signal_discovery_id = excluded.signal_discovery_id,
                    belief_json = excluded.belief_json,
                    created_at = excluded.created_at
                """,
                (
                    belief.compressed_belief_id,
                    signal_discovery_id,
                    json.dumps(belief.to_document(), sort_keys=True),
                    belief.created_at,
                ),
            )
        state = self.get_compressed_belief(belief.compressed_belief_id)
        assert state is not None
        return state

    def get_compressed_belief(
        self,
        compressed_belief_id: str,
    ) -> CompressedBeliefState | None:
        row = self.conn.execute(
            """
            SELECT compressed_belief_id, signal_discovery_id, belief_json, created_at
            FROM compressed_beliefs
            WHERE compressed_belief_id = ?
            """,
            (compressed_belief_id,),
        ).fetchone()
        return _row_to_compressed_belief(row)

    def list_compressed_beliefs(
        self,
        *,
        signal_discovery_id: str | None = None,
        limit: int = 20,
    ) -> list[CompressedBeliefState]:
        if signal_discovery_id is None:
            rows = self.conn.execute(
                """
                SELECT compressed_belief_id, signal_discovery_id, belief_json, created_at
                FROM compressed_beliefs
                ORDER BY created_at DESC, compressed_belief_id DESC
                LIMIT ?
                """,
                (max(int(limit), 1),),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT compressed_belief_id, signal_discovery_id, belief_json, created_at
                FROM compressed_beliefs
                WHERE signal_discovery_id = ?
                ORDER BY created_at DESC, compressed_belief_id DESC
                LIMIT ?
                """,
                (
                    signal_discovery_id,
                    max(int(limit), 1),
                ),
            ).fetchall()
        return [_row_to_compressed_belief(row) for row in rows if row is not None]

    def upsert_trading_strategy(
        self,
        *,
        trading_strategy: TradingStrategySpec,
    ) -> TradingStrategyState:
        self.ensure_schema()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO strategy_specs (
                    strategy_id, spec_json, created_at
                )
                VALUES (?, ?, ?)
                ON CONFLICT(strategy_id) DO UPDATE SET
                    spec_json = excluded.spec_json,
                    created_at = excluded.created_at
                """,
                (
                    trading_strategy.strategy_id,
                    json.dumps(
                        {"trading_strategy": trading_strategy.to_document()},
                        sort_keys=True,
                    ),
                    trading_strategy.created_at,
                ),
            )
        state = self.get_trading_strategy(trading_strategy.strategy_id)
        assert state is not None
        return state

    def get_trading_strategy(
        self,
        strategy_id: str,
    ) -> TradingStrategyState | None:
        row = self.conn.execute(
            """
            SELECT strategy_id, spec_json, created_at
            FROM strategy_specs
            WHERE strategy_id = ?
            """,
            (strategy_id,),
        ).fetchone()
        return _row_to_trading_strategy(row)

    def list_trading_strategies(
        self,
        *,
        limit: int = 20,
    ) -> list[TradingStrategyState]:
        rows = self.conn.execute(
            """
            SELECT strategy_id, spec_json, created_at
            FROM strategy_specs
            ORDER BY created_at DESC, strategy_id DESC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ).fetchall()
        return [_row_to_trading_strategy(row) for row in rows if row is not None]

    def upsert_evaluation_run_result(
        self,
        *,
        run_result: EvaluationRunResult,
    ) -> EvaluationRunResultState:
        self.ensure_schema()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO evaluation_run_results (
                    evaluation_run_result_id, evaluation_spec_id, run_result_json, created_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(evaluation_run_result_id) DO UPDATE SET
                    evaluation_spec_id = excluded.evaluation_spec_id,
                    run_result_json = excluded.run_result_json,
                    created_at = excluded.created_at
                """,
                (
                    run_result.evaluation_run_result_id,
                    run_result.evaluation_spec_id,
                    json.dumps(run_result.to_document(), sort_keys=True),
                    run_result.created_at,
                ),
            )
        state = self.get_evaluation_run_result(run_result.evaluation_run_result_id)
        assert state is not None
        return state

    def get_evaluation_run_result(
        self,
        evaluation_run_result_id: str,
    ) -> EvaluationRunResultState | None:
        row = self.conn.execute(
            """
            SELECT evaluation_run_result_id, evaluation_spec_id, run_result_json, created_at
            FROM evaluation_run_results
            WHERE evaluation_run_result_id = ?
            """,
            (evaluation_run_result_id,),
        ).fetchone()
        return _row_to_evaluation_run_result(row)

    def get_latest_evaluation_run_result(self) -> EvaluationRunResultState | None:
        row = self.conn.execute(
            """
            SELECT evaluation_run_result_id, evaluation_spec_id, run_result_json, created_at
            FROM evaluation_run_results
            ORDER BY created_at DESC, evaluation_run_result_id DESC
            LIMIT 1
            """
        ).fetchone()
        return _row_to_evaluation_run_result(row)

    def list_evaluation_run_results(
        self,
        *,
        evaluation_spec_id: str | None = None,
        limit: int = 20,
    ) -> list[EvaluationRunResultState]:
        if evaluation_spec_id is None:
            rows = self.conn.execute(
                """
                SELECT evaluation_run_result_id, evaluation_spec_id, run_result_json, created_at
                FROM evaluation_run_results
                ORDER BY created_at DESC, evaluation_run_result_id DESC
                LIMIT ?
                """,
                (max(int(limit), 1),),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT evaluation_run_result_id, evaluation_spec_id, run_result_json, created_at
                FROM evaluation_run_results
                WHERE evaluation_spec_id = ?
                ORDER BY created_at DESC, evaluation_run_result_id DESC
                LIMIT ?
                """,
                (
                    evaluation_spec_id,
                    max(int(limit), 1),
                ),
            ).fetchall()
        return [_row_to_evaluation_run_result(row) for row in rows if row is not None]

    def get_observation(self, evaluation_id: str) -> ObservationRecord | None:
        row = self.conn.execute(
            """
            SELECT evaluation_id, subject_id, asset, target_id, value, recorded_at
            FROM observations
            WHERE evaluation_id = ?
            """,
            (evaluation_id,),
        ).fetchone()
        return _row_to_observation(row)

    def list_observations_for_subject_or_asset(
        self,
        *,
        subject_id: str,
        asset: str,
        target_id: str,
        limit: int,
    ) -> list[ObservationRecord]:
        """List recent observations, preferring subject rows before asset fallback."""
        rows = self.conn.execute(
            """
            SELECT evaluation_id, subject_id, asset, target_id, value, recorded_at
            FROM observations
            WHERE subject_id = ? AND target_id = ?
            ORDER BY evaluation_id DESC
            LIMIT ?
            """,
            (subject_id, target_id, int(limit)),
        ).fetchall()
        if not rows:
            rows = self.conn.execute(
                """
                SELECT evaluation_id, subject_id, asset, target_id, value, recorded_at
                FROM observations
                WHERE asset = ? AND target_id = ?
                ORDER BY evaluation_id DESC
                LIMIT ?
                """,
                (asset, target_id, int(limit)),
            ).fetchall()
        return [
            observation
            for observation in (_row_to_observation(row) for row in rows)
            if observation is not None
        ]

    def finalize_observation(
        self,
        *,
        evaluation_id: str,
        observation_value: float,
        recorded_at: str | None = None,
        subject_id: str | None = None,
        asset: str | None = None,
        target_id: str = DEFAULT_TARGET,
    ) -> tuple[ObservationRecord, bool]:
        self.ensure_schema()
        self.register_target(target_id, recorded_at=recorded_at)
        existing = self.get_observation(evaluation_id)
        resolved_subject_id = DEFAULT_SUBJECT_ID if subject_id is None else subject_id
        resolved_asset = (
            default_runtime_asset(resolved_subject_id)
            if asset is None
            else asset
        )
        if existing is not None:
            if existing.value != float(observation_value):
                raise ValueError(
                    "observation already exists for this evaluation_id with a different value"
                )
            if (
                existing.subject_id != resolved_subject_id
                or existing.asset != resolved_asset
                or existing.target_id != target_id
            ):
                raise ValueError(
                    "observation already exists for this evaluation with different subject/asset/target"
                )
            return existing, False

        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO observations (
                    evaluation_id, subject_id, asset, target_id, value, recorded_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    resolved_subject_id,
                    resolved_asset,
                    target_id,
                    float(observation_value),
                    timestamp,
                ),
            )

        observation = self.get_observation(evaluation_id)
        assert observation is not None
        return observation, True
