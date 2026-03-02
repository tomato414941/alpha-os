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

            # --- if_gt templates ---

            # Trend signal: (if_gt f (lag_d f) 1.0 -1.0)
            # Price > lagged price = uptrend → long; else short
            for f in self.features:
                results.append(
                    ConditionalOp(
                        "if_gt",
                        Feature(f),
                        LagOp("lag", d, Feature(f)),
                        Constant(1.0),
                        Constant(-1.0),
                    )
                )

            # Momentum regime: (if_gt (roc_d f1) 0.0 f2 (neg f2))
            # If f1 has positive momentum, go long f2; else short f2
            for f1, f2 in pairs:
                results.append(
                    ConditionalOp(
                        "if_gt",
                        RollingOp("roc", d, Feature(f1)),
                        Constant(0.0),
                        Feature(f2),
                        UnaryOp("neg", Feature(f2)),
                    )
                )

            # Mean-reversion regime: (if_gt (rank_d f) 0.5 (neg f) f)
            # Rank > 0.5 = relatively expensive → short; else long
            for f in self.features:
                results.append(
                    ConditionalOp(
                        "if_gt",
                        RollingOp("rank", d, Feature(f)),
                        Constant(0.5),
                        UnaryOp("neg", Feature(f)),
                        Feature(f),
                    )
                )

            # --- lag templates ---

            # Lagged spread: (sub (lag_d f1) (lag_d f2))
            # Spread between two assets d days ago
            for f1, f2 in pairs:
                results.append(
                    BinaryOp(
                        "sub",
                        LagOp("lag", d, Feature(f1)),
                        LagOp("lag", d, Feature(f2)),
                    )
                )

            # Lagged momentum: (roc_d2 (lag_d f))
            # Momentum of past values — captures momentum regime shifts
            for d2 in self.windows:
                if d2 >= d:
                    continue
                for f in self.features:
                    results.append(
                        RollingOp(
                            "roc",
                            d2,
                            LagOp("lag", d, Feature(f)),
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


def generate_microstructure_templates() -> list[Expr]:
    """Seed templates for Layer 1 GP evolution with microstructure signals."""
    return [
        # VPIN spike detection
        ConditionalOp(
            op="if_gt",
            condition_left=Feature("vpin_btc"),
            condition_right=Constant(0.7),
            then_branch=Constant(1.0),
            else_branch=Constant(0.0),
        ),
        # Imbalance momentum
        BinaryOp(
            op="sub",
            left=Feature("book_imbalance_btc"),
            right=RollingOp(op="mean", window=5, child=Feature("book_imbalance_btc")),
        ),
        # Wide spread filter
        ConditionalOp(
            op="if_gt",
            condition_left=Feature("spread_bps_btc"),
            condition_right=Constant(10.0),
            then_branch=Constant(0.0),
            else_branch=Constant(1.0),
        ),
        # Flow weighted by inverse VPIN
        BinaryOp(
            op="mul",
            left=Feature("trade_flow_btc"),
            right=UnaryOp(op="neg", child=Feature("vpin_btc")),
        ),
        # Depth ratio divergence from mean
        BinaryOp(
            op="sub",
            left=Feature("book_depth_ratio_btc"),
            right=RollingOp(op="mean", window=20, child=Feature("book_depth_ratio_btc")),
        ),
        # Large trade count spike
        ConditionalOp(
            op="if_gt",
            condition_left=Feature("large_trade_count_btc"),
            condition_right=RollingOp(op="mean", window=10, child=Feature("large_trade_count_btc")),
            then_branch=Feature("trade_flow_btc"),
            else_branch=Constant(0.0),
        ),
    ]


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
