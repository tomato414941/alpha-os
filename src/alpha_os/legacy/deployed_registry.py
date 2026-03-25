"""Helpers for legacy deployed-alpha table operations."""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Callable

from .registry_types import AlphaRecord, DeployedAlphaEntry


def clear_deployed_alphas(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM deployed_alphas")
    conn.commit()


def list_deployed_alphas(
    conn: sqlite3.Connection,
    *,
    row_to_record: Callable[[sqlite3.Row], AlphaRecord],
) -> list[AlphaRecord]:
    rows = conn.execute(
        """
        SELECT a.*
        FROM deployed_alphas u
        JOIN alphas a ON a.alpha_id = u.alpha_id
        ORDER BY u.slot ASC
        """
    ).fetchall()
    return [row_to_record(row) for row in rows]


def list_deployed_alpha_entries(conn: sqlite3.Connection) -> list[DeployedAlphaEntry]:
    rows = conn.execute(
        """
        SELECT alpha_id, slot, deployed_at, deployment_score, metadata
        FROM deployed_alphas
        ORDER BY slot ASC
        """
    ).fetchall()
    return [
        DeployedAlphaEntry(
            alpha_id=row["alpha_id"],
            slot=row["slot"],
            deployed_at=row["deployed_at"],
            deployment_score=row["deployment_score"],
            metadata=json.loads(row["metadata"]),
        )
        for row in rows
    ]


def count_deployed_alphas(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM deployed_alphas").fetchone()
    return row[0]


def replace_deployed_alphas(
    conn: sqlite3.Connection,
    alpha_ids: list[str],
    *,
    scores: dict[str, float] | None = None,
    metadata: dict[str, dict] | None = None,
    deployed_at: float | None = None,
) -> None:
    unique_ids = list(dict.fromkeys(alpha_ids))
    if unique_ids:
        placeholders = ", ".join("?" for _ in unique_ids)
        rows = conn.execute(
            f"SELECT alpha_id FROM alphas WHERE alpha_id IN ({placeholders})",
            unique_ids,
        ).fetchall()
        existing_ids = {row["alpha_id"] for row in rows}
        missing = [alpha_id for alpha_id in unique_ids if alpha_id not in existing_ids]
        if missing:
            raise KeyError(f"Unknown alpha ids for deployed alphas: {missing[:5]}")

    stamp = deployed_at or time.time()
    clear_deployed_alphas(conn)
    if unique_ids:
        rows = []
        for slot, alpha_id in enumerate(unique_ids):
            rows.append(
                (
                    alpha_id,
                    slot,
                    stamp,
                    float((scores or {}).get(alpha_id, 0.0)),
                    json.dumps((metadata or {}).get(alpha_id, {})),
                )
            )
        conn.executemany(
            """
            INSERT INTO deployed_alphas
                (alpha_id, slot, deployed_at, deployment_score, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
    conn.commit()
