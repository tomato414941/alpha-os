from __future__ import annotations

import copy
import itertools
import random

from .tokens import (
    UNARY_OPS,
    BINARY_OPS,
    ROLLING_OPS,
    PAIR_ROLLING_OPS,
    CONDITIONAL_OPS,
    LAG_OPS,
    ALLOWED_WINDOWS,
)
from .expr import (
    Expr,
    Feature,
    Constant,
    UnaryOp,
    BinaryOp,
    RollingOp,
    PairRollingOp,
    ConditionalOp,
    LagOp,
)

_UNARY_LIST = sorted(UNARY_OPS)
_BINARY_LIST = sorted(BINARY_OPS)
_ROLLING_LIST = sorted(ROLLING_OPS)
_PAIR_ROLLING_LIST = sorted(PAIR_ROLLING_OPS)
_CONDITIONAL_LIST = sorted(CONDITIONAL_OPS)
_LAG_LIST = sorted(LAG_OPS)


class AlphaGenerator:
    def __init__(
        self,
        features: list[str],
        windows: list[int] | None = None,
        seed: int | None = None,
    ) -> None:
        if not features:
            raise ValueError("features must be non-empty")
        self.features = features
        self.windows = windows or ALLOWED_WINDOWS
        self.rng = random.Random(seed)

    def generate_random(self, n: int, max_depth: int = 3) -> list[Expr]:
        """Generate n random expression trees up to max_depth."""
        return [self._random_expr(max_depth) for _ in range(n)]

    def generate_from_templates(self) -> list[Expr]:
        """Generate from predefined templates with all parameter combinations."""
        results: list[Expr] = []
        pairs = list(itertools.combinations(self.features, 2))

        for d in self.windows:
            # Momentum divergence: (sub (roc_d f1) (roc_d f2))
            for f1, f2 in pairs:
                results.append(
                    BinaryOp(
                        "sub",
                        RollingOp("roc", d, Feature(f1)),
                        RollingOp("roc", d, Feature(f2)),
                    )
                )

            # Mean reversion: (neg (sub (rank_d f) 0.5))
            for f in self.features:
                results.append(
                    UnaryOp(
                        "neg",
                        BinaryOp("sub", RollingOp("rank", d, Feature(f)), Constant(0.5)),
                    )
                )

            # Relative strength: (sub (zscore (mean_d f1)) (zscore (mean_d f2)))
            for f1, f2 in pairs:
                results.append(
                    BinaryOp(
                        "sub",
                        UnaryOp("zscore", RollingOp("mean", d, Feature(f1))),
                        UnaryOp("zscore", RollingOp("mean", d, Feature(f2))),
                    )
                )

            # Correlation change: (delta_d (corr_d2 f1 f2))
            for d2 in self.windows:
                if d2 >= d:
                    continue
                for f1, f2 in pairs:
                    results.append(
                        RollingOp(
                            "delta",
                            d,
                            PairRollingOp("corr", d2, Feature(f1), Feature(f2)),
                        )
                    )

        return results

    def mutate(self, expr: Expr) -> Expr:
        """Mutate an expression: swap a feature, change a window, replace an op."""
        expr = copy.deepcopy(expr)
        nodes = _collect_nodes(expr)
        target = self.rng.choice(nodes)

        if isinstance(target, Feature):
            object.__setattr__(target, "name", self.rng.choice(self.features))
        elif isinstance(target, (RollingOp, PairRollingOp, LagOp)):
            new_window = self.rng.choice(self.windows)
            object.__setattr__(target, "window", new_window)
        elif isinstance(target, UnaryOp):
            object.__setattr__(target, "op", self.rng.choice(_UNARY_LIST))
        elif isinstance(target, BinaryOp):
            object.__setattr__(target, "op", self.rng.choice(_BINARY_LIST))
        elif isinstance(target, ConditionalOp):
            object.__setattr__(target, "op", self.rng.choice(_CONDITIONAL_LIST))

        return expr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_expr(self, max_depth: int) -> Expr:
        if max_depth <= 1:
            return self._random_leaf()

        kind = self.rng.choices(
            ["leaf", "unary", "binary", "rolling", "pair_rolling", "lag", "conditional"],
            weights=[2, 2, 3, 4, 1, 2, 1],
        )[0]

        if kind == "leaf":
            return self._random_leaf()
        if kind == "unary":
            return UnaryOp(
                self.rng.choice(_UNARY_LIST),
                self._random_expr(max_depth - 1),
            )
        if kind == "binary":
            return BinaryOp(
                self.rng.choice(_BINARY_LIST),
                self._random_expr(max_depth - 1),
                self._random_expr(max_depth - 1),
            )
        if kind == "rolling":
            return RollingOp(
                self.rng.choice(_ROLLING_LIST),
                self.rng.choice(self.windows),
                self._random_expr(max_depth - 1),
            )
        if kind == "pair_rolling":
            return PairRollingOp(
                self.rng.choice(_PAIR_ROLLING_LIST),
                self.rng.choice(self.windows),
                self._random_expr(max_depth - 1),
                self._random_expr(max_depth - 1),
            )
        if kind == "lag":
            return LagOp(
                self.rng.choice(_LAG_LIST),
                self.rng.choice(self.windows),
                self._random_expr(max_depth - 1),
            )
        # conditional
        return ConditionalOp(
            self.rng.choice(_CONDITIONAL_LIST),
            self._random_expr(max_depth - 1),
            self._random_expr(max_depth - 1),
            self._random_expr(max_depth - 1),
            self._random_expr(max_depth - 1),
        )

    def _random_leaf(self) -> Expr:
        if self.rng.random() < 0.9:
            return Feature(self.rng.choice(self.features))
        return Constant(round(self.rng.uniform(-2, 2), 2))


def _collect_nodes(expr: Expr) -> list[Expr]:
    nodes: list[Expr] = [expr]
    if isinstance(expr, UnaryOp):
        nodes.extend(_collect_nodes(expr.child))
    elif isinstance(expr, BinaryOp):
        nodes.extend(_collect_nodes(expr.left))
        nodes.extend(_collect_nodes(expr.right))
    elif isinstance(expr, RollingOp):
        nodes.extend(_collect_nodes(expr.child))
    elif isinstance(expr, PairRollingOp):
        nodes.extend(_collect_nodes(expr.left))
        nodes.extend(_collect_nodes(expr.right))
    elif isinstance(expr, LagOp):
        nodes.extend(_collect_nodes(expr.child))
    elif isinstance(expr, ConditionalOp):
        nodes.extend(_collect_nodes(expr.condition_left))
        nodes.extend(_collect_nodes(expr.condition_right))
        nodes.extend(_collect_nodes(expr.then_branch))
        nodes.extend(_collect_nodes(expr.else_branch))
    return nodes
