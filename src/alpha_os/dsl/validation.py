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

_SCALAR = "scalar"
_SERIES = "series"


def temporal_expression_issues(expr: Expr) -> list[str]:
    """Return structural issues that make time-series evaluation invalid."""
    return _expression_shape(expr)[1]


def is_temporally_valid(expr: Expr) -> bool:
    return not temporal_expression_issues(expr)


def _expression_shape(expr: Expr) -> tuple[str, list[str]]:
    if isinstance(expr, Feature):
        return _SERIES, []
    if isinstance(expr, Constant):
        return _SCALAR, []
    if isinstance(expr, UnaryOp):
        return _expression_shape(expr.child)
    if isinstance(expr, BinaryOp):
        lhs_kind, lhs_issues = _expression_shape(expr.left)
        rhs_kind, rhs_issues = _expression_shape(expr.right)
        result_kind = _SERIES if _SERIES in {lhs_kind, rhs_kind} else _SCALAR
        return result_kind, lhs_issues + rhs_issues
    if isinstance(expr, RollingOp):
        child_kind, child_issues = _expression_shape(expr.child)
        issues = list(child_issues)
        if child_kind != _SERIES:
            issues.append(
                f"{expr.op}_{expr.window} requires series input, got {child_kind}"
            )
        return _SERIES, issues
    if isinstance(expr, PairRollingOp):
        lhs_kind, lhs_issues = _expression_shape(expr.left)
        rhs_kind, rhs_issues = _expression_shape(expr.right)
        issues = list(lhs_issues) + list(rhs_issues)
        if lhs_kind != _SERIES:
            issues.append(
                f"{expr.op}_{expr.window} left input must be series, got {lhs_kind}"
            )
        if rhs_kind != _SERIES:
            issues.append(
                f"{expr.op}_{expr.window} right input must be series, got {rhs_kind}"
            )
        return _SERIES, issues
    if isinstance(expr, LagOp):
        child_kind, child_issues = _expression_shape(expr.child)
        issues = list(child_issues)
        if child_kind != _SERIES:
            issues.append(
                f"{expr.op}_{expr.window} requires series input, got {child_kind}"
            )
        return _SERIES, issues
    if isinstance(expr, ConditionalOp):
        cond_l_kind, cond_l_issues = _expression_shape(expr.condition_left)
        cond_r_kind, cond_r_issues = _expression_shape(expr.condition_right)
        then_kind, then_issues = _expression_shape(expr.then_branch)
        else_kind, else_issues = _expression_shape(expr.else_branch)
        result_kind = (
            _SERIES
            if _SERIES in {cond_l_kind, cond_r_kind, then_kind, else_kind}
            else _SCALAR
        )
        issues = cond_l_issues + cond_r_issues + then_issues + else_issues
        return result_kind, issues
    return _SERIES, []
