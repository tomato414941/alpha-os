from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from ..data.universe import infer_feature_family
from ..dsl import parse
from ..dsl.evaluator import EvaluationError, evaluate_expression, normalize_signal
from ..dsl.expr import (
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
from ..dsl.features import collect_feature_names
from ..dsl.generator import _collect_nodes


@dataclass(frozen=True)
class DiversitySummary:
    n_records: int
    n_analyzed: int
    n_skipped: int
    lookback: int
    mean_abs_signal_correlation: float
    mean_feature_overlap: float
    mean_structure_overlap: float
    mean_composite_similarity: float
    signal_diversity: float
    feature_diversity: float
    structure_diversity: float
    composite_diversity: float
    n_unique_features: int
    mean_abs_input_correlation: float
    input_diversity: float
    family_counts: dict[str, int]
    feature_usage_counts: dict[str, int]


@dataclass(frozen=True)
class AlphaDiversityRow:
    alpha_id: str
    feature_names: list[str]
    feature_families: list[str]
    node_count: int
    avg_abs_signal_correlation: float
    avg_feature_overlap: float
    avg_structure_overlap: float
    avg_composite_similarity: float


@dataclass(frozen=True)
class AlphaPairSimilarity:
    alpha_id_a: str
    alpha_id_b: str
    abs_signal_correlation: float
    feature_overlap: float
    structure_overlap: float
    composite_similarity: float


@dataclass(frozen=True)
class InputPairSimilarity:
    feature_a: str
    feature_b: str
    abs_input_correlation: float


@dataclass(frozen=True)
class DiversityReport:
    summary: DiversitySummary
    rows: list[AlphaDiversityRow]
    top_redundant_pairs: list[AlphaPairSimilarity]
    top_input_pairs: list[InputPairSimilarity]
    skipped_alpha_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "summary": asdict(self.summary),
            "rows": [asdict(row) for row in self.rows],
            "top_redundant_pairs": [asdict(pair) for pair in self.top_redundant_pairs],
            "top_input_pairs": [asdict(pair) for pair in self.top_input_pairs],
            "skipped_alpha_ids": self.skipped_alpha_ids,
        }


class SupportsExpression(Protocol):
    expression: str


def infer_feature_families(feature_names: set[str]) -> list[str]:
    return sorted({infer_feature_family(name) for name in feature_names})


def _record_id(record: object) -> str:
    hypothesis_id = getattr(record, "hypothesis_id", None)
    if isinstance(hypothesis_id, str) and hypothesis_id:
        return hypothesis_id
    alpha_id = getattr(record, "alpha_id", None)
    if isinstance(alpha_id, str) and alpha_id:
        return alpha_id
    raise AttributeError("diversity records must expose hypothesis_id or alpha_id")


def expression_structure_signature(expr: Expr) -> set[str]:
    signature: set[str] = set()

    def walk(node: Expr, depth: int) -> None:
        if isinstance(node, UnaryOp):
            signature.add(f"{depth}:U:{node.op}")
            walk(node.child, depth + 1)
            return
        if isinstance(node, BinaryOp):
            signature.add(f"{depth}:B:{node.op}")
            walk(node.left, depth + 1)
            walk(node.right, depth + 1)
            return
        if isinstance(node, RollingOp):
            signature.add(f"{depth}:R:{node.op}:{node.window}")
            walk(node.child, depth + 1)
            return
        if isinstance(node, PairRollingOp):
            signature.add(f"{depth}:PR:{node.op}:{node.window}")
            walk(node.left, depth + 1)
            walk(node.right, depth + 1)
            return
        if isinstance(node, LagOp):
            signature.add(f"{depth}:L:{node.op}:{node.window}")
            walk(node.child, depth + 1)
            return
        if isinstance(node, ConditionalOp):
            signature.add(f"{depth}:C:{node.op}")
            walk(node.condition_left, depth + 1)
            walk(node.condition_right, depth + 1)
            walk(node.then_branch, depth + 1)
            walk(node.else_branch, depth + 1)
            return
        if isinstance(node, Feature):
            signature.add(f"{depth}:F:{node.name}")
            return
        if isinstance(node, Constant):
            signature.add(f"{depth}:K")
            return
        signature.add(f"{depth}:X")

    walk(expr, 0)
    return signature


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _offdiag_mean(matrix: np.ndarray) -> float:
    n_rows = int(matrix.shape[0])
    if n_rows <= 1:
        return 0.0
    mask = ~np.eye(n_rows, dtype=bool)
    return float(matrix[mask].mean())


def _pairwise_jaccard(sets: list[set[str]]) -> np.ndarray:
    n_rows = len(sets)
    out = np.eye(n_rows, dtype=np.float64)
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            score = _jaccard_similarity(sets[i], sets[j])
            out[i, j] = score
            out[j, i] = score
    return out


def _signal_abs_correlation_matrix(signals: np.ndarray) -> np.ndarray:
    n_rows = signals.shape[0]
    if n_rows == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if n_rows == 1:
        return np.ones((1, 1), dtype=np.float64)
    corr = np.corrcoef(signals)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 1.0)
    return corr


def analyze_diversity(
    records: list[SupportsExpression],
    data: dict[str, np.ndarray],
    n_days: int,
    *,
    lookback: int = 252,
    top_pairs: int = 10,
) -> DiversityReport:
    analyzed_ids: list[str] = []
    feature_sets: list[set[str]] = []
    family_sets: list[list[str]] = []
    structure_sets: list[set[str]] = []
    node_counts: list[int] = []
    signal_rows: list[np.ndarray] = []
    skipped_ids: list[str] = []
    feature_usage_counts: dict[str, int] = {}

    for record in records:
        record_id = _record_id(record)
        try:
            expr = parse(record.expression)
            signal = normalize_signal(evaluate_expression(expr, data, n_days))
        except (EvaluationError, Exception):
            skipped_ids.append(record_id)
            continue

        names = collect_feature_names(expr)
        feature_sets.append(names)
        family_sets.append(infer_feature_families(names))
        structure_sets.append(expression_structure_signature(expr))
        node_counts.append(len(_collect_nodes(expr)))
        signal_rows.append(np.asarray(signal, dtype=np.float64)[-lookback:])
        analyzed_ids.append(record_id)
        for name in names:
            feature_usage_counts[name] = feature_usage_counts.get(name, 0) + 1

    n_analyzed = len(analyzed_ids)
    if n_analyzed == 0:
        summary = DiversitySummary(
            n_records=len(records),
            n_analyzed=0,
            n_skipped=len(skipped_ids),
            lookback=lookback,
            mean_abs_signal_correlation=0.0,
            mean_feature_overlap=0.0,
            mean_structure_overlap=0.0,
            mean_composite_similarity=0.0,
            signal_diversity=0.0,
            feature_diversity=0.0,
            structure_diversity=0.0,
            composite_diversity=0.0,
            n_unique_features=0,
            mean_abs_input_correlation=0.0,
            input_diversity=0.0,
            family_counts={},
            feature_usage_counts={},
        )
        return DiversityReport(summary, [], [], [], skipped_ids)

    signals = np.vstack(signal_rows)
    signal_corr = _signal_abs_correlation_matrix(signals)
    feature_overlap = _pairwise_jaccard(feature_sets)
    structure_overlap = _pairwise_jaccard(structure_sets)
    composite_similarity = (signal_corr + feature_overlap + structure_overlap) / 3.0

    unique_features = sorted({name for names in feature_sets for name in names})
    family_counts: dict[str, int] = {}
    for families in family_sets:
        for family in families:
            family_counts[family] = family_counts.get(family, 0) + 1

    input_pairs: list[InputPairSimilarity] = []
    mean_abs_input_correlation = 0.0
    if unique_features:
        feature_matrix = []
        ordered_features = []
        for feature_name in unique_features:
            values = data.get(feature_name)
            if values is None:
                continue
            ordered_features.append(feature_name)
            feature_matrix.append(np.asarray(values, dtype=np.float64)[-lookback:])
        if len(feature_matrix) > 1:
            input_corr = _signal_abs_correlation_matrix(np.vstack(feature_matrix))
            mean_abs_input_correlation = _offdiag_mean(input_corr)
            for i, feature_a in enumerate(ordered_features):
                for j in range(i + 1, len(ordered_features)):
                    input_pairs.append(
                        InputPairSimilarity(
                            feature_a=feature_a,
                            feature_b=ordered_features[j],
                            abs_input_correlation=float(input_corr[i, j]),
                        )
                    )
            input_pairs.sort(key=lambda pair: pair.abs_input_correlation, reverse=True)

    rows: list[AlphaDiversityRow] = []
    redundant_pairs: list[AlphaPairSimilarity] = []
    for idx, record_id in enumerate(analyzed_ids):
        if n_analyzed > 1:
            mean_signal = float(
                np.delete(signal_corr[idx], idx).mean()
            )
            mean_feature = float(
                np.delete(feature_overlap[idx], idx).mean()
            )
            mean_structure = float(
                np.delete(structure_overlap[idx], idx).mean()
            )
            mean_composite = float(
                np.delete(composite_similarity[idx], idx).mean()
            )
        else:
            mean_signal = 0.0
            mean_feature = 0.0
            mean_structure = 0.0
            mean_composite = 0.0
        rows.append(
            AlphaDiversityRow(
                alpha_id=record_id,
                feature_names=sorted(feature_sets[idx]),
                feature_families=family_sets[idx],
                node_count=node_counts[idx],
                avg_abs_signal_correlation=mean_signal,
                avg_feature_overlap=mean_feature,
                avg_structure_overlap=mean_structure,
                avg_composite_similarity=mean_composite,
            )
        )
        for other_idx in range(idx + 1, n_analyzed):
            redundant_pairs.append(
                AlphaPairSimilarity(
                    alpha_id_a=record_id,
                    alpha_id_b=analyzed_ids[other_idx],
                    abs_signal_correlation=float(signal_corr[idx, other_idx]),
                    feature_overlap=float(feature_overlap[idx, other_idx]),
                    structure_overlap=float(structure_overlap[idx, other_idx]),
                    composite_similarity=float(composite_similarity[idx, other_idx]),
                )
            )

    redundant_pairs.sort(key=lambda pair: pair.composite_similarity, reverse=True)
    summary = DiversitySummary(
        n_records=len(records),
        n_analyzed=n_analyzed,
        n_skipped=len(skipped_ids),
        lookback=lookback,
        mean_abs_signal_correlation=_offdiag_mean(signal_corr),
        mean_feature_overlap=_offdiag_mean(feature_overlap),
        mean_structure_overlap=_offdiag_mean(structure_overlap),
        mean_composite_similarity=_offdiag_mean(composite_similarity),
        signal_diversity=1.0 - _offdiag_mean(signal_corr),
        feature_diversity=1.0 - _offdiag_mean(feature_overlap),
        structure_diversity=1.0 - _offdiag_mean(structure_overlap),
        composite_diversity=1.0 - _offdiag_mean(composite_similarity),
        n_unique_features=len(unique_features),
        mean_abs_input_correlation=mean_abs_input_correlation,
        input_diversity=1.0 - mean_abs_input_correlation,
        family_counts=family_counts,
        feature_usage_counts=feature_usage_counts,
    )
    return DiversityReport(
        summary=summary,
        rows=rows,
        top_redundant_pairs=redundant_pairs[:top_pairs],
        top_input_pairs=input_pairs[:top_pairs],
        skipped_alpha_ids=skipped_ids,
    )
