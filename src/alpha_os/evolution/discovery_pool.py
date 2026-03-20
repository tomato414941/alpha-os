"""MAP-Elites discovery pool for alpha expressions."""
from __future__ import annotations

import json
import logging
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..dsl.expr import Expr

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryPoolConfig:
    dims: tuple[int, ...] = (8, 8, 8, 8)
    ranges: tuple[tuple[float, float], ...] = (
        (0.0, 50.0),    # persistence: half-life in days
        (0.0, 1.0),     # activity: fraction of non-trivial days
        (-0.15, 0.15),  # price_beta: momentum vs mean-reversion
        (-0.6, 0.6),    # vol_sensitivity: volatile vs calm market affinity
    )
    max_nan_ratio: float = 0.1


@dataclass
class DiscoveryPoolEntry:
    expr: Expr
    fitness: float
    behavior: np.ndarray
    survival_score: float = 0.0
    best_horizon: int = 1


@dataclass(frozen=True)
class DiscoveryPoolUpdate:
    stored: bool
    replaced: bool = False


def passes_sanity_filter(
    signal: np.ndarray, max_nan_ratio: float = 0.1
) -> bool:
    """Check if a signal passes sanity requirements for a discovery-pool entry."""
    if signal.size == 0:
        return False
    nan_ratio = np.isnan(signal).sum() / signal.size
    if nan_ratio > max_nan_ratio:
        return False
    clean = signal[~np.isnan(signal)]
    if clean.size == 0:
        return False
    if not np.all(np.isfinite(clean)):
        return False
    if np.std(clean) == 0:
        return False
    return True


def _migrate_table(conn: sqlite3.Connection) -> None:
    """Migrate legacy 'archive' table to 'discovery_pool' if needed."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS discovery_pool (
            cell_key TEXT PRIMARY KEY,
            expression TEXT NOT NULL,
            fitness REAL NOT NULL,
            behavior TEXT NOT NULL,
            best_horizon INTEGER NOT NULL DEFAULT 1
        )
    """)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "archive" in tables:
        conn.execute(
            "INSERT OR IGNORE INTO discovery_pool"
            " (cell_key, expression, fitness, behavior)"
            " SELECT * FROM archive"
        )
        conn.execute("DROP TABLE archive")
        conn.commit()
    # Add best_horizon column to existing tables
    cols = {row[1] for row in conn.execute("PRAGMA table_info(discovery_pool)")}
    if "best_horizon" not in cols:
        conn.execute("ALTER TABLE discovery_pool ADD COLUMN best_horizon INTEGER NOT NULL DEFAULT 1")
        conn.commit()


class DiscoveryPool:
    """Grid-based MAP-Elites discovery pool storing alpha expressions.

    Discovery-pool cells act as local quality frontiers.
    A valid candidate enters an empty cell or replaces the incumbent when it
    improves the cell-local survival score.
    """

    def __init__(self, config: DiscoveryPoolConfig | None = None):
        self.config = config or DiscoveryPoolConfig()
        self._grid: dict[tuple[int, ...], DiscoveryPoolEntry] = {}

    def add(self, expr: Expr, fitness: float, behavior: np.ndarray) -> bool:
        """Add expression to the pool using fitness-based replacement.

        Returns True if it was stored (new cell or better fitness than incumbent).
        """
        cell = self._to_cell(behavior)
        existing = self._grid.get(cell)
        if existing is None or fitness > existing.survival_score:
            self._grid[cell] = DiscoveryPoolEntry(
                expr=expr,
                fitness=fitness,
                behavior=behavior,
                survival_score=fitness,
            )
            return True
        return False

    def store_candidate(
        self,
        expr: Expr,
        behavior: np.ndarray,
        signal: np.ndarray,
        *,
        fitness: float = 0.0,
        survival_score: float | None = None,
        best_horizon: int = 1,
    ) -> DiscoveryPoolUpdate:
        """Store a candidate if it is valid and improves its cell frontier."""
        if not passes_sanity_filter(signal, self.config.max_nan_ratio):
            return DiscoveryPoolUpdate(stored=False)
        cell = self._to_cell(behavior)
        incumbent = self._grid.get(cell)
        candidate_score = float(fitness if survival_score is None else survival_score)
        if incumbent is not None and candidate_score <= incumbent.survival_score:
            return DiscoveryPoolUpdate(stored=False)
        self._grid[cell] = DiscoveryPoolEntry(
            expr=expr,
            fitness=float(fitness),
            behavior=behavior,
            survival_score=candidate_score,
            best_horizon=best_horizon,
        )
        return DiscoveryPoolUpdate(stored=True, replaced=incumbent is not None)

    def _to_cell(self, behavior: np.ndarray) -> tuple[int, ...]:
        indices: list[int] = []
        for i, (lo, hi) in enumerate(self.config.ranges):
            val = float(np.clip(behavior[i], lo, hi))
            dim_size = self.config.dims[i]
            idx = int((val - lo) / (hi - lo + 1e-12) * dim_size)
            indices.append(min(idx, dim_size - 1))
        return tuple(indices)

    @property
    def size(self) -> int:
        return len(self._grid)

    @property
    def capacity(self) -> int:
        cap = 1
        for d in self.config.dims:
            cap *= d
        return cap

    @property
    def coverage(self) -> float:
        return self.size / self.capacity

    def best(self, n: int = 10) -> list[tuple[Expr, float]]:
        """Top-n entries by fitness."""
        entries = sorted(
            self._grid.values(), key=lambda e: e.fitness, reverse=True
        )
        return [(e.expr, e.fitness) for e in entries[:n]]

    def sample(self, n: int, rng: random.Random | None = None) -> list[DiscoveryPoolEntry]:
        """Sample n entries uniformly from occupied cells."""
        rng = rng or random.Random()
        entries = list(self._grid.values())
        if len(entries) <= n:
            return list(entries)
        return rng.sample(entries, n)

    def elites(self) -> list[tuple[Expr, float, np.ndarray]]:
        """All discovery-pool entries as (expr, fitness, behavior) triples."""
        return [(e.expr, e.fitness, e.behavior) for e in self._grid.values()]

    def elites_with_horizon(self) -> list[tuple[Expr, float, np.ndarray, int]]:
        """All entries as (expr, fitness, behavior, best_horizon) tuples."""
        return [
            (e.expr, e.fitness, e.behavior, e.best_horizon)
            for e in self._grid.values()
        ]

    # --- Persistence (SQLite) ---

    def save_to_db(self, db_path: Path) -> int:
        """Persist the discovery pool to SQLite. Returns the number of rows written."""
        from ..dsl import to_string

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        _migrate_table(conn)
        rows = []
        for cell, entry in self._grid.items():
            rows.append((
                json.dumps(cell),
                to_string(entry.expr),
                entry.fitness,
                json.dumps(entry.behavior.tolist()),
                entry.best_horizon,
            ))
        conn.execute("DELETE FROM discovery_pool")
        conn.executemany(
            "INSERT INTO discovery_pool (cell_key, expression, fitness, behavior, best_horizon)"
            " VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()
        return len(rows)

    @classmethod
    def load_from_db(cls, db_path: Path, config: DiscoveryPoolConfig | None = None) -> DiscoveryPool:
        """Load the discovery pool from SQLite.

        Recomputes cell keys from stored behavior vectors.  When the
        number of behavior axes has changed (e.g. 3→4), old entries
        cannot be placed and are dropped — the pool effectively
        resets, keeping only entries whose behavior dimension matches.
        """
        from ..dsl import parse

        pool = cls(config=config)
        if not db_path.exists():
            return pool

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA busy_timeout=30000")
        _migrate_table(conn)
        try:
            rows = conn.execute(
                "SELECT cell_key, expression, fitness, behavior, best_horizon"
                " FROM discovery_pool"
            ).fetchall()
        except sqlite3.OperationalError:
            conn.close()
            return pool
        conn.close()

        expected_dims = len(pool.config.dims)
        n_collisions = 0
        n_skipped = 0
        for _cell_json, expr_str, fitness, behavior_json, best_horizon in rows:
            try:
                expr = parse(expr_str)
                behavior = np.array(json.loads(behavior_json))
                if len(behavior) != expected_dims:
                    n_skipped += 1
                    continue
                cell = pool._to_cell(behavior)
                incumbent = pool._grid.get(cell)
                if incumbent is not None and incumbent.fitness >= fitness:
                    n_collisions += 1
                    continue
                pool._grid[cell] = DiscoveryPoolEntry(
                    expr=expr,
                    fitness=fitness,
                    behavior=behavior,
                    survival_score=fitness,
                    best_horizon=int(best_horizon) if best_horizon else 1,
                )
            except Exception as exc:
                logger.warning("Failed to load pool entry %s: %s", expr_str, exc)

        if n_skipped > 0 or n_collisions > 0:
            logger.info(
                "Pool load: %d rows, %d loaded, %d dim-mismatch skipped, %d collisions",
                len(rows), pool.size, n_skipped, n_collisions,
            )

        return pool
