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
from .parser import parse, to_string


_COMMUTATIVE_BINARY_OPS = {"add", "mul", "max", "min"}
_COMMUTATIVE_PAIR_OPS = {"corr", "cov"}


def canonicalize(expr: Expr) -> Expr:
    if isinstance(expr, (Feature, Constant)):
        return expr

    if isinstance(expr, UnaryOp):
        child = canonicalize(expr.child)
        return UnaryOp(expr.op, child)

    if isinstance(expr, BinaryOp):
        left = canonicalize(expr.left)
        right = canonicalize(expr.right)
        if expr.op == "sub" and to_string(left) == to_string(right):
            return Constant(0.0)
        if expr.op in _COMMUTATIVE_BINARY_OPS:
            left, right = sorted((left, right), key=to_string)
        return BinaryOp(expr.op, left, right)

    if isinstance(expr, RollingOp):
        return RollingOp(expr.op, expr.window, canonicalize(expr.child))

    if isinstance(expr, PairRollingOp):
        left = canonicalize(expr.left)
        right = canonicalize(expr.right)
        if expr.op in _COMMUTATIVE_PAIR_OPS:
            left, right = sorted((left, right), key=to_string)
        return PairRollingOp(expr.op, expr.window, left, right)

    if isinstance(expr, LagOp):
        return LagOp(expr.op, expr.window, canonicalize(expr.child))

    if isinstance(expr, ConditionalOp):
        cond_left = canonicalize(expr.condition_left)
        cond_right = canonicalize(expr.condition_right)
        then_branch = canonicalize(expr.then_branch)
        else_branch = canonicalize(expr.else_branch)
        if to_string(cond_left) == to_string(cond_right):
            return else_branch
        if to_string(then_branch) == to_string(else_branch):
            return then_branch
        return ConditionalOp(
            expr.op,
            cond_left,
            cond_right,
            then_branch,
            else_branch,
        )

    return expr


def canonical_string(expression: str) -> str:
    return to_string(canonicalize(parse(expression)))
