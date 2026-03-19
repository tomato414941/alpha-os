"""Helpers for replaying admission gates and rebuilding registry state."""
from __future__ import annotations

import hashlib
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .lifecycle import LifecycleConfig, passes_candidate_gate
from .managed_alphas import AlphaRecord, ManagedAlphaStore, AlphaState


@dataclass(frozen=True)
class RegistryRebuildStats:
    source_name: str
    source_rows: int
    active_count: int
    dormant_count: int
    rejected_count: int
    backup_path: Path | None
    registry_db: Path


def alpha_id_for_expression(
    expression: str,
    *,
    existing_ids: dict[str, str] | None = None,
) -> str:
    if existing_ids and expression in existing_ids:
        return existing_ids[expression]
    digest = hashlib.md5(
        expression.encode(),
        usedforsecurity=False,
    ).hexdigest()[:8]
    return f"v2_{digest}"


def existing_alpha_ids_by_expression(db_path: Path) -> dict[str, str]:
    registry = ManagedAlphaStore(db_path)
    try:
        return {
            record.expression: record.alpha_id
            for record in registry.list_all()
        }
    finally:
        registry.close()


def load_registry_records(db_path: Path) -> list[AlphaRecord]:
    registry = ManagedAlphaStore(db_path)
    try:
        return registry.list_all()
    finally:
        registry.close()


def load_candidate_records(db_path: Path) -> list[AlphaRecord]:
    existing_ids = existing_alpha_ids_by_expression(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT candidate_id, expression, fitness, oos_sharpe, pbo, dsr_pvalue,
                   created_at, validated_at
            FROM candidates
            WHERE oos_sharpe IS NOT NULL
              AND pbo IS NOT NULL
              AND dsr_pvalue IS NOT NULL
            ORDER BY COALESCE(validated_at, created_at) ASC, candidate_id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    latest_by_expression: dict[str, AlphaRecord] = {}
    for (
        candidate_id,
        expression,
        fitness,
        oos_sharpe,
        pbo,
        dsr_pvalue,
        created_at,
        validated_at,
    ) in rows:
        ts = float(validated_at or created_at or time.time())
        alpha_id = alpha_id_for_expression(expression, existing_ids=existing_ids)
        latest_by_expression[expression] = AlphaRecord(
            alpha_id=alpha_id,
            expression=expression,
            state=AlphaState.CANDIDATE,
            fitness=float(fitness or 0.0),
            oos_sharpe=float(oos_sharpe or 0.0),
            pbo=float(pbo or 1.0),
            dsr_pvalue=float(dsr_pvalue or 1.0),
            created_at=ts,
            updated_at=ts,
            metadata={"candidate_id": candidate_id},
        )
    return list(latest_by_expression.values())


def load_source_records(db_path: Path, source: str) -> list[AlphaRecord]:
    if source == "alphas":
        return load_registry_records(db_path)
    if source == "candidates":
        return load_candidate_records(db_path)
    raise ValueError(f"unsupported source: {source}")


def materialize_admission_snapshot(
    records: list[AlphaRecord],
    config: LifecycleConfig,
    *,
    fail_state: str = AlphaState.REJECTED,
) -> tuple[list[AlphaRecord], dict[str, int]]:
    fail_state = AlphaState.canonical(fail_state)
    counts = {
        AlphaState.ACTIVE: 0,
        AlphaState.DORMANT: 0,
        AlphaState.REJECTED: 0,
    }
    snapshot: list[AlphaRecord] = []
    for record in records:
        state = (
            AlphaState.ACTIVE
            if passes_candidate_gate(record, config)
            else fail_state
        )
        if state not in counts:
            counts[state] = 0
        counts[state] += 1
        snapshot.append(
            AlphaRecord(
                alpha_id=record.alpha_id,
                expression=record.expression,
                state=state,
                fitness=record.fitness,
                oos_sharpe=record.oos_sharpe,
                oos_log_growth=record.oos_log_growth,
                pbo=record.pbo,
                dsr_pvalue=record.dsr_pvalue,
                turnover=record.turnover,
                correlation_avg=record.correlation_avg,
                created_at=record.created_at,
                updated_at=record.updated_at,
                metadata=record.metadata,
            )
        )
    return snapshot, counts


def backup_registry_db(db_path: Path) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup_path = db_path.with_name(f"{db_path.name}.bak-{timestamp}")
    shutil.copy2(db_path, backup_path)
    return backup_path


def apply_registry_snapshot(db_path: Path, records: list[AlphaRecord]) -> None:
    registry = ManagedAlphaStore(db_path)
    try:
        registry.replace_all(records)
    finally:
        registry.close()


def rebuild_registry(
    db_path: Path,
    config: LifecycleConfig,
    *,
    source: str = "candidates",
    fail_state: str = AlphaState.REJECTED,
    dry_run: bool = False,
    backup: bool = True,
) -> RegistryRebuildStats:
    source_records = load_source_records(db_path, source)
    snapshot, counts = materialize_admission_snapshot(
        source_records,
        config,
        fail_state=fail_state,
    )
    backup_path = None
    if not dry_run:
        if backup and db_path.exists():
            backup_path = backup_registry_db(db_path)
        apply_registry_snapshot(db_path, snapshot)
    return RegistryRebuildStats(
        source_name=source,
        source_rows=len(source_records),
        active_count=counts.get(AlphaState.ACTIVE, 0),
        dormant_count=counts.get(AlphaState.DORMANT, 0),
        rejected_count=counts.get(AlphaState.REJECTED, 0),
        backup_path=backup_path,
        registry_db=db_path,
    )
