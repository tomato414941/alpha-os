"""Helpers for legacy alpha registry table operations."""
from __future__ import annotations

import json
import sqlite3
import time

from .registry_types import AlphaRecord, AlphaState


def ensure_alpha_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alphas (
            alpha_id TEXT PRIMARY KEY,
            expression TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'candidate',
            fitness REAL DEFAULT 0.0,
            oos_sharpe REAL DEFAULT 0.0,
            pbo REAL DEFAULT 1.0,
            dsr_pvalue REAL DEFAULT 1.0,
            turnover REAL DEFAULT 0.0,
            correlation_avg REAL DEFAULT 0.0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            metadata TEXT DEFAULT '{}',
            stake REAL DEFAULT 0.0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_state ON alphas(state)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_oos_sharpe ON alphas(oos_sharpe DESC)
    """)

    cols = {row[1] for row in conn.execute("PRAGMA table_info(alphas)")}
    if "stake" not in cols:
        conn.execute("ALTER TABLE alphas ADD COLUMN stake REAL DEFAULT 0.0")
        conn.execute(
            "UPDATE alphas SET stake = MAX(oos_sharpe, 0.0) WHERE state = 'active'"
        )
        conn.commit()

    try:
        conn.execute(
            "ALTER TABLE alphas ADD COLUMN oos_log_growth REAL DEFAULT 0.0"
        )
    except sqlite3.OperationalError:
        pass

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_stake ON alphas(stake DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_oos_log_growth
        ON alphas(oos_log_growth DESC)
    """)
    conn.commit()


def replace_all(conn: sqlite3.Connection, records: list[AlphaRecord]) -> None:
    values = [record_values(record) for record in records]
    conn.execute("DELETE FROM alphas")
    if values:
        conn.executemany(
            """INSERT INTO alphas
               (alpha_id, expression, state, fitness, oos_sharpe, oos_log_growth,
                pbo, dsr_pvalue, turnover, correlation_avg,
                created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values,
        )
    conn.commit()


def list_all(conn: sqlite3.Connection) -> list[AlphaRecord]:
    rows = conn.execute("SELECT * FROM alphas").fetchall()
    return [row_to_record(row) for row in rows]


def register(conn: sqlite3.Connection, record: AlphaRecord) -> None:
    now = time.time()
    if record.created_at == 0.0:
        record.created_at = now
    record.updated_at = now
    conn.execute(
        """INSERT OR REPLACE INTO alphas
           (alpha_id, expression, state, fitness, oos_sharpe, oos_log_growth,
            pbo, dsr_pvalue, turnover, correlation_avg,
            created_at, updated_at, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        record_values(record),
    )
    conn.commit()


def get(conn: sqlite3.Connection, alpha_id: str) -> AlphaRecord | None:
    row = conn.execute(
        "SELECT * FROM alphas WHERE alpha_id = ?",
        (alpha_id,),
    ).fetchone()
    if row is None:
        return None
    return row_to_record(row)


def list_by_state(conn: sqlite3.Connection, state: str) -> list[AlphaRecord]:
    canonical_state = AlphaState.canonical(state)
    rows = conn.execute(
        "SELECT * FROM alphas WHERE state = ? ORDER BY oos_sharpe DESC",
        (canonical_state,),
    ).fetchall()
    return [row_to_record(row) for row in rows]


def update_state(conn: sqlite3.Connection, alpha_id: str, new_state: str) -> None:
    canonical_state = AlphaState.canonical(new_state)
    conn.execute(
        "UPDATE alphas SET state = ?, updated_at = ? WHERE alpha_id = ?",
        (canonical_state, time.time(), alpha_id),
    )
    conn.commit()


def bulk_update_states(
    conn: sqlite3.Connection,
    alpha_ids: list[str],
    new_state: str,
) -> None:
    unique_ids = list(dict.fromkeys(alpha_ids))
    if not unique_ids:
        return
    canonical_state = AlphaState.canonical(new_state)
    stamp = time.time()
    conn.executemany(
        "UPDATE alphas SET state = ?, updated_at = ? WHERE alpha_id = ?",
        [(canonical_state, stamp, alpha_id) for alpha_id in unique_ids],
    )
    conn.commit()


def update_stake(conn: sqlite3.Connection, alpha_id: str, stake: float) -> None:
    conn.execute(
        "UPDATE alphas SET stake = ?, updated_at = ? WHERE alpha_id = ?",
        (stake, time.time(), alpha_id),
    )
    conn.commit()


def bulk_update_stakes(conn: sqlite3.Connection, stakes: dict[str, float]) -> None:
    if not stakes:
        return
    stamp = time.time()
    conn.executemany(
        "UPDATE alphas SET stake = ?, updated_at = ? WHERE alpha_id = ?",
        [(stake, stamp, alpha_id) for alpha_id, stake in stakes.items()],
    )
    conn.commit()


def top_by_stake(conn: sqlite3.Connection, n: int = 30) -> list[AlphaRecord]:
    rows = conn.execute(
        "SELECT * FROM alphas WHERE stake > 0 ORDER BY stake DESC LIMIT ?",
        (n,),
    ).fetchall()
    return [row_to_record(row) for row in rows]


def update_metrics(
    conn: sqlite3.Connection,
    alpha_id: str,
    *,
    oos_sharpe: float | None = None,
    pbo: float | None = None,
    dsr_pvalue: float | None = None,
) -> None:
    updates = []
    params: list[float | str] = []
    if oos_sharpe is not None:
        updates.append("oos_sharpe = ?")
        params.append(oos_sharpe)
    if pbo is not None:
        updates.append("pbo = ?")
        params.append(pbo)
    if dsr_pvalue is not None:
        updates.append("dsr_pvalue = ?")
        params.append(dsr_pvalue)
    if not updates:
        return
    updates.append("updated_at = ?")
    params.append(time.time())
    params.append(alpha_id)
    conn.execute(
        f"UPDATE alphas SET {', '.join(updates)} WHERE alpha_id = ?",
        params,
    )
    conn.commit()


def count(conn: sqlite3.Connection, state: str | None = None) -> int:
    if state:
        canonical_state = AlphaState.canonical(state)
        row = conn.execute(
            "SELECT COUNT(*) FROM alphas WHERE state = ?",
            (canonical_state,),
        ).fetchone()
    else:
        row = conn.execute("SELECT COUNT(*) FROM alphas").fetchone()
    return row[0]


def top(
    conn: sqlite3.Connection,
    *,
    n: int = 10,
    state: str | None = None,
    metric: str = "sharpe",
) -> list[AlphaRecord]:
    col = {"sharpe": "oos_sharpe", "log_growth": "oos_log_growth"}[metric]
    if state:
        canonical_state = AlphaState.canonical(state)
        rows = conn.execute(
            f"SELECT * FROM alphas WHERE state = ? ORDER BY {col} DESC LIMIT ?",
            (canonical_state, n),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT * FROM alphas ORDER BY {col} DESC LIMIT ?",
            (n,),
        ).fetchall()
    return [row_to_record(row) for row in rows]


def record_values(record: AlphaRecord) -> tuple:
    now = time.time()
    if record.created_at == 0.0:
        record.created_at = now
    if record.updated_at == 0.0:
        record.updated_at = now
    state = AlphaState.canonical(record.state)
    return (
        record.alpha_id,
        record.expression,
        state,
        record.fitness,
        record.oos_sharpe,
        record.oos_log_growth,
        record.pbo,
        record.dsr_pvalue,
        record.turnover,
        record.correlation_avg,
        record.created_at,
        record.updated_at,
        json.dumps(record.metadata),
    )


def row_to_record(row: sqlite3.Row) -> AlphaRecord:
    keys = row.keys()
    return AlphaRecord(
        alpha_id=row["alpha_id"],
        expression=row["expression"],
        state=AlphaState.canonical(row["state"]),
        fitness=row["fitness"],
        oos_sharpe=row["oos_sharpe"],
        oos_log_growth=row["oos_log_growth"] if "oos_log_growth" in keys else 0.0,
        pbo=row["pbo"],
        dsr_pvalue=row["dsr_pvalue"],
        turnover=row["turnover"],
        correlation_avg=row["correlation_avg"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        metadata=json.loads(row["metadata"]),
        stake=row["stake"] if "stake" in keys else 0.0,
    )
