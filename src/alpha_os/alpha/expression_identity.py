from __future__ import annotations

from collections.abc import Iterable

from ..dsl import parse
from ..dsl.canonical import canonical_string
from ..dsl.features import collect_feature_names


def expression_semantic_key(expression: str) -> str:
    try:
        return canonical_string(expression)
    except Exception:
        return expression


def expression_feature_names(expression: str) -> set[str]:
    try:
        return collect_feature_names(parse(expression))
    except Exception:
        return set()


def count_expression_features(expressions: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for expression in expressions:
        for name in expression_feature_names(expression):
            counts[name] = counts.get(name, 0) + 1
    return counts
