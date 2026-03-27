from __future__ import annotations

from .expr import (
    BinaryOp,
    ConditionalOp,
    Constant,
    Expr,
    Feature,
    LagOp,
    PairRollingOp,
    RollingOp,
    UnaryOp,
)


def collect_feature_names(expr: Expr) -> set[str]:
    """Return all feature names referenced by an expression tree."""
    names: set[str] = set()

    def walk(node: Expr) -> None:
        if isinstance(node, Feature):
            names.add(node.name)
            return
        if isinstance(node, Constant):
            return
        if isinstance(node, (UnaryOp, RollingOp, LagOp)):
            walk(node.child)
            return
        if isinstance(node, (BinaryOp, PairRollingOp)):
            walk(node.left)
            walk(node.right)
            return
        if isinstance(node, ConditionalOp):
            walk(node.condition_left)
            walk(node.condition_right)
            walk(node.then_branch)
            walk(node.else_branch)
            return

    walk(expr)
    return names
