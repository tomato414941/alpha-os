"""Admission queue helpers backed by the candidates table."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field

from ..config import asset_data_dir


@dataclass(frozen=True)
class CandidateSeed:
    expression: str
    source: str
    fitness: float = 0.0
    behavior_json: dict = field(default_factory=dict)
    created_at: float | None = None


@dataclass(frozen=True)
class PendingCandidatePruneStats:
    asset: str
    max_age_days: int
    selected_count: int
    pruned_count: int


def ensure_candidate_source_metadata(conn: sqlite3.Connection) -> None:
    try:
        conn.execute(
            "ALTER TABLE candidates ADD COLUMN source TEXT NOT NULL DEFAULT ''"
        )
    except sqlite3.OperationalError:
        pass
    conn.execute(
        "UPDATE candidates SET source = 'alpha_generator_btc' "
        "WHERE source = '' AND candidate_id LIKE 'alpha_generator_btc_%'"
    )
    conn.execute(
        "UPDATE candidates SET source = 'manual' "
        "WHERE source = '' AND candidate_id LIKE 'manual_%'"
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_candidates_source_status_created
        ON candidates(source, status, created_at DESC)
        """
    )
    conn.commit()


def queue_candidate_expressions(
    conn: sqlite3.Connection,
    expressions: list[str],
    *,
    source: str,
    fitness: float = 0.0,
    behavior_json: dict | None = None,
    created_at: float | None = None,
) -> int:
    rows = [
        CandidateSeed(
            expression=expression,
            source=source,
            fitness=float(fitness),
            behavior_json=dict(behavior_json or {}),
            created_at=created_at,
        )
        for expression in expressions
    ]
    return queue_candidates(conn, rows)


def queue_candidates(
    conn: sqlite3.Connection,
    seeds: list[CandidateSeed],
) -> int:
    if not seeds:
        return 0

    payload: dict[str, tuple[str, str, float, str, str, float]] = {}
    for seed in seeds:
        stamp = seed.created_at or time.time()
        digest = hashlib.md5(
            f"{seed.source}:{seed.expression}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:16]
        candidate_id = f"{seed.source}_{digest}"
        payload[candidate_id] = (
            candidate_id,
            seed.source,
            seed.expression,
            float(seed.fitness),
            "pending",
            json.dumps(seed.behavior_json),
            stamp,
        )

    rows = list(payload.values())
    placeholders = ", ".join("?" for _ in rows)
    existing = set()
    if rows:
        existing_rows = conn.execute(
            f"SELECT candidate_id FROM candidates WHERE candidate_id IN ({placeholders})",
            [row[0] for row in rows],
        ).fetchall()
        existing = {row["candidate_id"] for row in existing_rows}

    conn.executemany(
        """INSERT OR IGNORE INTO candidates
           (candidate_id, source, expression, fitness, status, behavior_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows) - len(existing)


def count_pending_candidates(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE status = 'pending'"
    ).fetchone()
    return row[0]


def fetch_pending_candidates(
    conn: sqlite3.Connection,
    limit: int,
) -> list[tuple[str, str, float]]:
    rows = conn.execute(
        """
        SELECT candidate_id, expression, fitness
        FROM candidates
        WHERE status = 'pending'
        ORDER BY
            CASE
                WHEN source LIKE 'alpha_generator_%' THEN 0
                WHEN source = 'manual' THEN 1
                ELSE 2
            END,
            created_at DESC,
            fitness DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return rows


def mark_candidates_validating(
    conn: sqlite3.Connection,
    candidate_ids: list[str],
) -> None:
    conn.executemany(
        "UPDATE candidates SET status = 'validating' WHERE candidate_id = ?",
        [(candidate_id,) for candidate_id in candidate_ids],
    )
    conn.commit()


def reject_candidate(
    conn: sqlite3.Connection,
    candidate_id: str,
    reason: str,
) -> None:
    conn.execute(
        "UPDATE candidates SET status = 'rejected', "
        "validated_at = ?, error_message = ? WHERE candidate_id = ?",
        (time.time(), reason[:200], candidate_id),
    )
    conn.commit()


def adopt_candidate(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    oos_sharpe: float,
    pbo: float,
    dsr_pvalue: float,
) -> None:
    conn.execute(
        "UPDATE candidates SET status = 'adopted', "
        "oos_sharpe = ?, pbo = ?, dsr_pvalue = ?, validated_at = ? "
        "WHERE candidate_id = ?",
        (oos_sharpe, pbo, dsr_pvalue, time.time(), candidate_id),
    )
    conn.commit()


def reset_candidates_to_pending(
    conn: sqlite3.Connection,
    candidate_ids: list[str],
) -> None:
    conn.executemany(
        "UPDATE candidates SET status = 'pending' WHERE candidate_id = ?",
        [(candidate_id,) for candidate_id in candidate_ids],
    )
    conn.commit()


def gc_old_candidate_results(
    conn: sqlite3.Connection,
    *,
    max_age_days: int = 30,
) -> int:
    cutoff = time.time() - max_age_days * 86400
    result = conn.execute(
        "DELETE FROM candidates WHERE status IN ('adopted', 'rejected') "
        "AND created_at < ?",
        (cutoff,),
    )
    conn.commit()
    return result.rowcount


def prune_stale_pending_candidates(
    asset: str,
    *,
    max_age_days: int,
    dry_run: bool = False,
) -> PendingCandidatePruneStats:
    cutoff = time.time() - max_age_days * 86400
    db_path = asset_data_dir(asset) / "alpha_registry.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        ensure_candidate_source_metadata(conn)
        rows = conn.execute(
            """
            SELECT candidate_id
            FROM candidates
            WHERE status = 'pending'
              AND created_at < ?
              AND source NOT LIKE 'alpha_generator_%'
              AND source != 'manual'
            """,
            (cutoff,),
        ).fetchall()
        selected_count = len(rows)
        if dry_run or not rows:
            return PendingCandidatePruneStats(
                asset=asset,
                max_age_days=max_age_days,
                selected_count=selected_count,
                pruned_count=0,
            )

        stamp = time.time()
        conn.executemany(
            """
            UPDATE candidates
            SET status = 'rejected',
                validated_at = ?,
                error_message = ?
            WHERE candidate_id = ?
            """,
            [
                (
                    stamp,
                    f"stale pending > {max_age_days}d",
                    row[0],
                )
                for row in rows
            ],
        )
        conn.commit()
        return PendingCandidatePruneStats(
            asset=asset,
            max_age_days=max_age_days,
            selected_count=selected_count,
            pruned_count=selected_count,
        )
    finally:
        conn.close()
