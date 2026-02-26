from __future__ import annotations

import re

from .tokens import UNARY_OPS, BINARY_OPS, ROLLING_OPS, PAIR_ROLLING_OPS, CONDITIONAL_OPS, LAG_OPS
from .expr import Expr, Feature, Constant, UnaryOp, BinaryOp, RollingOp, PairRollingOp, ConditionalOp, LagOp


_TOKEN_RE = re.compile(r"""(\(|\)|[^\s()]+)""")
_ROLLING_RE = re.compile(r"^([a-z_]+)_(\d+)$")


def parse(s: str) -> Expr:
    """Parse S-expression string to Expression tree.

    Examples:
      'nvda'                              -> Feature('nvda')
      '1.5'                               -> Constant(1.5)
      '(neg nvda)'                        -> UnaryOp('neg', Feature('nvda'))
      '(sub (roc_10 nvda) (roc_10 nasdaq))' -> BinaryOp(...)
      '(corr_60 nvda sp500)'              -> PairRollingOp(...)
    """
    tokens = _tokenize(s)
    expr, pos = _parse_expr(tokens, 0)
    if pos != len(tokens):
        raise SyntaxError(f"Unexpected token at position {pos}: {tokens[pos]}")
    return expr


def to_string(expr: Expr) -> str:
    """Convert Expression tree back to S-expression string."""
    return repr(expr)


def _tokenize(s: str) -> list[str]:
    tokens = _TOKEN_RE.findall(s.strip())
    if not tokens:
        raise SyntaxError("Empty expression")
    return tokens


def _parse_expr(tokens: list[str], pos: int) -> tuple[Expr, int]:
    if pos >= len(tokens):
        raise SyntaxError("Unexpected end of expression")

    token = tokens[pos]

    if token == "(":
        return _parse_compound(tokens, pos)

    if token == ")":
        raise SyntaxError(f"Unexpected ')' at position {pos}")

    # Atom: try numeric constant, then feature name
    return _parse_atom(token), pos + 1


def _parse_atom(token: str) -> Expr:
    try:
        return Constant(float(token))
    except ValueError:
        return Feature(token)


def _parse_compound(tokens: list[str], pos: int) -> tuple[Expr, int]:
    # pos points to '('
    pos += 1  # skip '('
    if pos >= len(tokens):
        raise SyntaxError("Unexpected end after '('")

    op_token = tokens[pos]
    pos += 1

    # Determine operator kind
    op_name, window = _resolve_op(op_token)

    if window is not None:
        if op_name in PAIR_ROLLING_OPS:
            left, pos = _parse_expr(tokens, pos)
            right, pos = _parse_expr(tokens, pos)
            pos = _expect(tokens, pos, ")")
            return PairRollingOp(op_name, window, left, right), pos
        if op_name in ROLLING_OPS:
            child, pos = _parse_expr(tokens, pos)
            pos = _expect(tokens, pos, ")")
            return RollingOp(op_name, window, child), pos
        if op_name in LAG_OPS:
            child, pos = _parse_expr(tokens, pos)
            pos = _expect(tokens, pos, ")")
            return LagOp(op_name, window, child), pos
        raise SyntaxError(f"Unknown rolling operator: {op_name}")

    if op_name in UNARY_OPS:
        child, pos = _parse_expr(tokens, pos)
        pos = _expect(tokens, pos, ")")
        return UnaryOp(op_name, child), pos

    if op_name in BINARY_OPS:
        left, pos = _parse_expr(tokens, pos)
        right, pos = _parse_expr(tokens, pos)
        pos = _expect(tokens, pos, ")")
        return BinaryOp(op_name, left, right), pos

    if op_name in CONDITIONAL_OPS:
        cond_left, pos = _parse_expr(tokens, pos)
        cond_right, pos = _parse_expr(tokens, pos)
        then_branch, pos = _parse_expr(tokens, pos)
        else_branch, pos = _parse_expr(tokens, pos)
        pos = _expect(tokens, pos, ")")
        return ConditionalOp(op_name, cond_left, cond_right, then_branch, else_branch), pos

    raise SyntaxError(f"Unknown operator: {op_name}")


def _resolve_op(token: str) -> tuple[str, int | None]:
    m = _ROLLING_RE.match(token)
    if m:
        return m.group(1), int(m.group(2))
    return token, None


def _expect(tokens: list[str], pos: int, expected: str) -> int:
    if pos >= len(tokens) or tokens[pos] != expected:
        found = tokens[pos] if pos < len(tokens) else "EOF"
        raise SyntaxError(f"Expected '{expected}' at position {pos}, got '{found}'")
    return pos + 1
