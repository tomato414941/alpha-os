"""Alpha expression DSL — S-expression based alpha factor language."""

from .tokens import (
    OpType,
    UNARY_OPS,
    BINARY_OPS,
    ROLLING_OPS,
    PAIR_ROLLING_OPS,
    CONDITIONAL_OPS,
    LAG_OPS,
    ALLOWED_WINDOWS,
)
from .expr import Expr, Feature, Constant, UnaryOp, BinaryOp, RollingOp, PairRollingOp, ConditionalOp, LagOp
from .parser import parse, to_string
from .generator import AlphaGenerator

__all__ = [
    "OpType",
    "UNARY_OPS",
    "BINARY_OPS",
    "ROLLING_OPS",
    "PAIR_ROLLING_OPS",
    "CONDITIONAL_OPS",
    "LAG_OPS",
    "ALLOWED_WINDOWS",
    "Expr",
    "Feature",
    "Constant",
    "UnaryOp",
    "BinaryOp",
    "RollingOp",
    "PairRollingOp",
    "ConditionalOp",
    "LagOp",
    "parse",
    "to_string",
    "AlphaGenerator",
]
