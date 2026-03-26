from __future__ import annotations

from collections.abc import Iterable

from ..data.universe import infer_feature_family
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


def expression_feature_families(expression: str) -> tuple[str, ...]:
    return tuple(sorted(infer_feature_family(name) for name in expression_feature_names(expression)))


def representative_feature_family(families: Iterable[str]) -> str:
    unique = sorted({family for family in families if family})
    if not unique:
        return "other"
    specific = [family for family in unique if family not in {"other", "unknown"}]
    if specific:
        return specific[0]
    return unique[0]


def count_expression_features(expressions: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for expression in expressions:
        for name in expression_feature_names(expression):
            counts[name] = counts.get(name, 0) + 1
    return counts
