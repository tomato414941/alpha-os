"""GP evolution loop using DEAP + existing DSL Expr trees."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from ..dsl.expr import Expr
from ..dsl.generator import AlphaGenerator, _collect_nodes

logger = logging.getLogger(__name__)


@dataclass
class GPConfig:
    pop_size: int = 200
    n_generations: int = 30
    cx_prob: float = 0.5
    mut_prob: float = 0.3
    tournament_size: int = 3
    max_depth: int = 3
    elite_size: int = 5
    bloat_penalty: float = 0.01
    depth_penalty: float = 0.0
    similarity_penalty: float = 0.0


def _tree_depth(expr: Expr) -> int:
    from ..dsl.expr import UnaryOp, BinaryOp, RollingOp, PairRollingOp, ConditionalOp, LagOp

    if isinstance(expr, (UnaryOp,)):
        return 1 + _tree_depth(expr.child)
    if isinstance(expr, RollingOp):
        return 1 + _tree_depth(expr.child)
    if isinstance(expr, LagOp):
        return 1 + _tree_depth(expr.child)
    if isinstance(expr, BinaryOp):
        return 1 + max(_tree_depth(expr.left), _tree_depth(expr.right))
    if isinstance(expr, PairRollingOp):
        return 1 + max(_tree_depth(expr.left), _tree_depth(expr.right))
    if isinstance(expr, ConditionalOp):
        return 1 + max(
            _tree_depth(expr.condition_left),
            _tree_depth(expr.condition_right),
            _tree_depth(expr.then_branch),
            _tree_depth(expr.else_branch),
        )
    return 0


def _node_count(expr: Expr) -> int:
    return len(_collect_nodes(expr))


def _ast_signature(expr: Expr) -> set[str]:
    """Return a set-based AST signature for structure similarity scoring."""
    from ..dsl.expr import (
        UnaryOp,
        BinaryOp,
        RollingOp,
        PairRollingOp,
        ConditionalOp,
        LagOp,
        Feature,
        Constant,
    )

    sig: set[str] = set()

    def walk(node: Expr, depth: int) -> None:
        if isinstance(node, UnaryOp):
            sig.add(f"{depth}:U:{node.op}")
            walk(node.child, depth + 1)
        elif isinstance(node, BinaryOp):
            sig.add(f"{depth}:B:{node.op}")
            walk(node.left, depth + 1)
            walk(node.right, depth + 1)
        elif isinstance(node, RollingOp):
            sig.add(f"{depth}:R:{node.op}:{node.window}")
            walk(node.child, depth + 1)
        elif isinstance(node, PairRollingOp):
            sig.add(f"{depth}:PR:{node.op}:{node.window}")
            walk(node.left, depth + 1)
            walk(node.right, depth + 1)
        elif isinstance(node, LagOp):
            sig.add(f"{depth}:L:{node.op}:{node.window}")
            walk(node.child, depth + 1)
        elif isinstance(node, ConditionalOp):
            sig.add(f"{depth}:C:{node.op}")
            walk(node.condition_left, depth + 1)
            walk(node.condition_right, depth + 1)
            walk(node.then_branch, depth + 1)
            walk(node.else_branch, depth + 1)
        elif isinstance(node, Feature):
            sig.add(f"{depth}:F:{node.name}")
        elif isinstance(node, Constant):
            sig.add(f"{depth}:K")
        else:
            sig.add(f"{depth}:X")

    walk(expr, 0)
    return sig


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def crossover(parent1: Expr, parent2: Expr, rng: random.Random) -> tuple[Expr, Expr]:
    """Subtree crossover: swap random subtrees between two parents."""
    import copy

    c1 = copy.deepcopy(parent1)
    c2 = copy.deepcopy(parent2)

    nodes1 = _collect_nodes(c1)
    nodes2 = _collect_nodes(c2)

    if len(nodes1) < 2 or len(nodes2) < 2:
        return c1, c2

    # Pick non-root nodes for swap
    idx1 = rng.randint(1, len(nodes1) - 1)
    idx2 = rng.randint(1, len(nodes2) - 1)
    n1 = nodes1[idx1]
    n2 = nodes2[idx2]

    # Find parents and replace
    _replace_child(c1, n1, n2)
    _replace_child(c2, n2, n1)

    return c1, c2


def _replace_child(root: Expr, old: Expr, new: Expr) -> None:
    """Replace old subtree with new in root (in-place on frozen dataclass via object.__setattr__)."""
    from ..dsl.expr import UnaryOp, BinaryOp, RollingOp, PairRollingOp, ConditionalOp, LagOp

    if isinstance(root, UnaryOp):
        if root.child is old:
            object.__setattr__(root, "child", new)
        else:
            _replace_child(root.child, old, new)
    elif isinstance(root, BinaryOp):
        if root.left is old:
            object.__setattr__(root, "left", new)
        elif root.right is old:
            object.__setattr__(root, "right", new)
        else:
            _replace_child(root.left, old, new)
            _replace_child(root.right, old, new)
    elif isinstance(root, RollingOp):
        if root.child is old:
            object.__setattr__(root, "child", new)
        else:
            _replace_child(root.child, old, new)
    elif isinstance(root, PairRollingOp):
        if root.left is old:
            object.__setattr__(root, "left", new)
        elif root.right is old:
            object.__setattr__(root, "right", new)
        else:
            _replace_child(root.left, old, new)
            _replace_child(root.right, old, new)
    elif isinstance(root, LagOp):
        if root.child is old:
            object.__setattr__(root, "child", new)
        else:
            _replace_child(root.child, old, new)
    elif isinstance(root, ConditionalOp):
        if root.condition_left is old:
            object.__setattr__(root, "condition_left", new)
        elif root.condition_right is old:
            object.__setattr__(root, "condition_right", new)
        elif root.then_branch is old:
            object.__setattr__(root, "then_branch", new)
        elif root.else_branch is old:
            object.__setattr__(root, "else_branch", new)
        else:
            _replace_child(root.condition_left, old, new)
            _replace_child(root.condition_right, old, new)
            _replace_child(root.then_branch, old, new)
            _replace_child(root.else_branch, old, new)


class GPEvolver:
    """Runs GP evolution to produce alpha expressions."""

    def __init__(
        self,
        features: list[str],
        evaluate_fn,
        config: GPConfig | None = None,
        seed: int | None = None,
    ):
        self.config = config or GPConfig()
        self.generator = AlphaGenerator(features, seed=seed)
        self.evaluate_fn = evaluate_fn
        self.rng = random.Random(seed)
        self.features = features
        self._signature_memory: list[set[str]] = []

    def run(self) -> list[tuple[Expr, float]]:
        """Run GP evolution. Returns list of (expression, fitness) sorted by fitness desc."""
        cfg = self.config

        # Initialize population
        pop = self.generator.generate_random(cfg.pop_size, max_depth=cfg.max_depth)

        # Evaluate initial population
        fitnesses = self._evaluate_batch(pop)

        all_results: list[tuple[Expr, float]] = []

        for gen in range(cfg.n_generations):
            # Selection (tournament)
            selected = self._tournament_select(pop, fitnesses, len(pop))

            # Crossover + mutation
            offspring = []
            i = 0
            while i < len(selected) - 1:
                p1, p2 = selected[i], selected[i + 1]
                if self.rng.random() < cfg.cx_prob:
                    c1, c2 = crossover(p1, p2, self.rng)
                    offspring.extend([c1, c2])
                else:
                    offspring.extend([p1, p2])
                i += 2
            if i < len(selected):
                offspring.append(selected[i])

            # Mutation
            for j in range(len(offspring)):
                if self.rng.random() < cfg.mut_prob:
                    offspring[j] = self.generator.mutate(offspring[j])

            # Enforce depth limit
            offspring = [
                e if _tree_depth(e) <= cfg.max_depth
                else self.generator._random_expr(cfg.max_depth)
                for e in offspring
            ]

            # Evaluate
            off_fitnesses = self._evaluate_batch(offspring)

            # Elitism: keep best from previous generation
            combined = list(zip(pop, fitnesses)) + list(zip(offspring, off_fitnesses))
            combined.sort(key=lambda x: x[1], reverse=True)

            pop = [e for e, _ in combined[:cfg.pop_size]]
            fitnesses = [f for _, f in combined[:cfg.pop_size]]
            self._update_signature_memory(pop[: cfg.elite_size])

            all_results.extend(combined)

        # Deduplicate and sort
        seen = set()
        unique = []
        for expr, fit in sorted(all_results, key=lambda x: x[1], reverse=True):
            key = repr(expr)
            if key not in seen:
                seen.add(key)
                unique.append((expr, fit))
        return unique

    def _evaluate_batch(self, exprs: list[Expr]) -> list[float]:
        from ..alpha.evaluator import FAILED_FITNESS

        results = []
        n_failed = 0
        for expr in exprs:
            try:
                raw_fitness = self.evaluate_fn(expr)
                penalty = (
                    self.config.bloat_penalty * _node_count(expr)
                    + self.config.depth_penalty * _tree_depth(expr)
                )
                if self.config.similarity_penalty > 0:
                    sig = _ast_signature(expr)
                    sim = self._max_similarity(sig)
                    penalty += self.config.similarity_penalty * sim
                results.append(raw_fitness - penalty)
            except Exception:
                results.append(FAILED_FITNESS)
                n_failed += 1
        if n_failed:
            logger.debug("GP batch: %d/%d evaluations failed", n_failed, len(exprs))
        return results

    def _max_similarity(self, signature: set[str]) -> float:
        if not self._signature_memory:
            return 0.0
        return max(_jaccard_similarity(signature, ref) for ref in self._signature_memory)

    def _update_signature_memory(self, exprs: list[Expr]) -> None:
        if not exprs:
            return
        self._signature_memory.extend(_ast_signature(expr) for expr in exprs)

    def _tournament_select(
        self, pop: list[Expr], fitnesses: list[float], n: int
    ) -> list[Expr]:
        selected = []
        for _ in range(n):
            idxs = [self.rng.randint(0, len(pop) - 1) for _ in range(self.config.tournament_size)]
            best = max(idxs, key=lambda i: fitnesses[i])
            selected.append(pop[best])
        return selected
