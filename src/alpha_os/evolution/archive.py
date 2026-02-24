"""MAP-Elites quality-diversity archive for alpha expressions."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ..dsl.expr import Expr


@dataclass
class ArchiveConfig:
    dims: tuple[int, ...] = (10, 10, 10, 10)
    ranges: tuple[tuple[float, float], ...] = (
        (0.0, 1.0),    # corr_to_live_book
        (0.0, 100.0),  # holding_half_life (days)
        (0.0, 2.0),    # turnover
        (1.0, 20.0),   # complexity (node count)
    )


@dataclass
class ArchiveEntry:
    expr: Expr
    fitness: float
    behavior: np.ndarray


class AlphaArchive:
    """Grid-based MAP-Elites archive storing alpha expressions.

    Each cell in the N-dimensional grid holds the single best (highest fitness)
    expression whose behavior descriptor falls into that cell.
    """

    def __init__(self, config: ArchiveConfig | None = None):
        self.config = config or ArchiveConfig()
        self._grid: dict[tuple[int, ...], ArchiveEntry] = {}

    def add(self, expr: Expr, fitness: float, behavior: np.ndarray) -> bool:
        """Add expression to archive.

        Returns True if it was stored (new cell or better fitness than incumbent).
        """
        cell = self._to_cell(behavior)
        existing = self._grid.get(cell)
        if existing is None or fitness > existing.fitness:
            self._grid[cell] = ArchiveEntry(
                expr=expr, fitness=fitness, behavior=behavior
            )
            return True
        return False

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

    def sample(self, n: int, rng: random.Random | None = None) -> list[ArchiveEntry]:
        """Sample n entries uniformly from occupied cells."""
        rng = rng or random.Random()
        entries = list(self._grid.values())
        if len(entries) <= n:
            return list(entries)
        return rng.sample(entries, n)

    def elites(self) -> list[tuple[Expr, float, np.ndarray]]:
        """All archive entries as (expr, fitness, behavior) triples."""
        return [(e.expr, e.fitness, e.behavior) for e in self._grid.values()]
