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
    dims: tuple[int, ...] = (100, 10, 10)
    ranges: tuple[tuple[float, float], ...] = (
        (0.0, 100.0),  # feature_bucket (hash mod 100)
        (0.0, 100.0),  # holding_half_life (days)
        (1.0, 20.0),   # complexity (node count)
    )
    max_nan_ratio: float = 0.1


@dataclass
class DiscoveryPoolEntry:
    expr: Expr
    fitness: float
    behavior: np.ndarray


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


class DiscoveryPool:
    """Grid-based MAP-Elites discovery pool storing alpha expressions.

    Path B mode: sanity filter only, no fitness-based competition.
    Candidates enter empty cells; occupied cells keep their incumbent.
    """

    def __init__(self, config: DiscoveryPoolConfig | None = None):
        self.config = config or DiscoveryPoolConfig()
        self._grid: dict[tuple[int, ...], DiscoveryPoolEntry] = {}

    def add(self, expr: Expr, fitness: float, behavior: np.ndarray) -> bool:
        """Add expression to the archive using fitness-based replacement.

        Returns True if it was stored (new cell or better fitness than incumbent).
        """
        cell = self._to_cell(behavior)
        existing = self._grid.get(cell)
        if existing is None or fitness > existing.fitness:
            self._grid[cell] = DiscoveryPoolEntry(
                expr=expr, fitness=fitness, behavior=behavior
            )
            return True
        return False

    def add_if_empty(
        self,
        expr: Expr,
        behavior: np.ndarray,
        signal: np.ndarray,
        *,
        fitness: float = 0.0,
    ) -> bool:
        """Add expression to the discovery pool if the cell is empty.

        Path B mode: no fitness competition. First valid occupant stays,
        but we still persist its originating fitness for later promotion.
        """
        if not passes_sanity_filter(signal, self.config.max_nan_ratio):
            return False
        cell = self._to_cell(behavior)
        if cell in self._grid:
            return False
        self._grid[cell] = DiscoveryPoolEntry(
            expr=expr,
            fitness=float(fitness),
            behavior=behavior,
        )
        return True

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

    # --- Persistence (SQLite) ---

    def save_to_db(self, db_path: Path) -> int:
        """Persist the discovery pool to SQLite. Returns the number of rows written."""
        from ..dsl import to_string

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS archive (
                cell_key TEXT PRIMARY KEY,
                expression TEXT NOT NULL,
                fitness REAL NOT NULL,
                behavior TEXT NOT NULL
            )
        """)
        rows = []
        for cell, entry in self._grid.items():
            rows.append((
                json.dumps(cell),
                to_string(entry.expr),
                entry.fitness,
                json.dumps(entry.behavior.tolist()),
            ))
        conn.execute("DELETE FROM archive")
        conn.executemany(
            "INSERT INTO archive (cell_key, expression, fitness, behavior) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()
        return len(rows)

    @classmethod
    def load_from_db(cls, db_path: Path, config: DiscoveryPoolConfig | None = None) -> DiscoveryPool:
        """Load the discovery pool from SQLite."""
        from ..dsl import parse

        archive = cls(config=config)
        if not db_path.exists():
            return archive

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            rows = conn.execute(
                "SELECT cell_key, expression, fitness, behavior FROM archive"
            ).fetchall()
        except sqlite3.OperationalError:
            conn.close()
            return archive
        conn.close()

        for cell_json, expr_str, fitness, behavior_json in rows:
            try:
                cell = tuple(json.loads(cell_json))
                expr = parse(expr_str)
                behavior = np.array(json.loads(behavior_json))
                archive._grid[cell] = DiscoveryPoolEntry(
                    expr=expr, fitness=fitness, behavior=behavior,
                )
            except Exception as exc:
                logger.warning("Failed to load archive entry %s: %s", cell_json, exc)

        return archive
