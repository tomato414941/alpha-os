"""Helpers for analyzing alpha diversity across multiple axes."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..alpha.evaluator import EvaluationError, evaluate_expression, normalize_signal
from ..data.universe import infer_feature_family
from ..dsl import parse
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
from .managed_alphas import AlphaRecord


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


def infer_feature_families(feature_names: set[str]) -> list[str]:
    return sorted({infer_feature_family(name) for name in feature_names})


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
    n = int(matrix.shape[0])
    if n <= 1:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(matrix[mask].mean())


def _pairwise_jaccard(sets: list[set[str]]) -> np.ndarray:
    n = len(sets)
    out = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            score = _jaccard_similarity(sets[i], sets[j])
            out[i, j] = score
            out[j, i] = score
    return out


def _signal_abs_correlation_matrix(signals: np.ndarray) -> np.ndarray:
    n = signals.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if n == 1:
        return np.ones((1, 1), dtype=np.float64)
    corr = np.corrcoef(signals)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 1.0)
    return corr


def analyze_diversity(
    records: list[AlphaRecord],
    data: dict[str, np.ndarray],
    n_days: int,
    *,
    lookback: int = 252,
    top_pairs: int = 10,
) -> DiversityReport:
    analyzed_records: list[AlphaRecord] = []
    feature_sets: list[set[str]] = []
    family_sets: list[list[str]] = []
    structure_sets: list[set[str]] = []
    node_counts: list[int] = []
    signal_rows: list[np.ndarray] = []
    skipped_ids: list[str] = []
    feature_usage_counts: dict[str, int] = {}

    for record in records:
        try:
            expr = parse(record.expression)
            signal = normalize_signal(evaluate_expression(expr, data, n_days))
        except (EvaluationError, Exception):
            skipped_ids.append(record.alpha_id)
            continue

        names = collect_feature_names(expr)
        feature_sets.append(names)
        family_sets.append(infer_feature_families(names))
        structure_sets.append(expression_structure_signature(expr))
        node_counts.append(len(_collect_nodes(expr)))
        signal_rows.append(np.asarray(signal, dtype=np.float64)[-lookback:])
        analyzed_records.append(record)
        for name in names:
            feature_usage_counts[name] = feature_usage_counts.get(name, 0) + 1

    n_analyzed = len(analyzed_records)
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
        return DiversityReport(
            summary=summary,
            rows=[],
            top_redundant_pairs=[],
            top_input_pairs=[],
            skipped_alpha_ids=skipped_ids,
        )

    signals = np.vstack(signal_rows)
    signal_overlap = _signal_abs_correlation_matrix(signals)
    feature_overlap = _pairwise_jaccard(feature_sets)
    structure_overlap = _pairwise_jaccard(structure_sets)
    composite_similarity = (signal_overlap + feature_overlap + structure_overlap) / 3.0

    input_features = sorted(feature_usage_counts)
    input_series: list[np.ndarray] = []
    valid_input_features: list[str] = []
    for name in input_features:
        raw = data.get(name)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float64)[-lookback:]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size < 10 or np.std(arr) <= 1e-12:
            continue
        valid_input_features.append(name)
        input_series.append(arr)

    if input_series:
        input_overlap = _signal_abs_correlation_matrix(np.vstack(input_series))
    else:
        input_overlap = np.zeros((0, 0), dtype=np.float64)

    avg_signal = [0.0] * n_analyzed
    avg_feature = [0.0] * n_analyzed
    avg_structure = [0.0] * n_analyzed
    avg_composite = [0.0] * n_analyzed
    if n_analyzed > 1:
        for i in range(n_analyzed):
            mask = np.arange(n_analyzed) != i
            avg_signal[i] = float(signal_overlap[i, mask].mean())
            avg_feature[i] = float(feature_overlap[i, mask].mean())
            avg_structure[i] = float(structure_overlap[i, mask].mean())
            avg_composite[i] = float(composite_similarity[i, mask].mean())

    rows: list[AlphaDiversityRow] = []
    family_counts: dict[str, int] = {}
    for idx, record in enumerate(analyzed_records):
        for family in family_sets[idx]:
            family_counts[family] = family_counts.get(family, 0) + 1
        rows.append(
            AlphaDiversityRow(
                alpha_id=record.alpha_id,
                feature_names=sorted(feature_sets[idx]),
                feature_families=family_sets[idx],
                node_count=node_counts[idx],
                avg_abs_signal_correlation=avg_signal[idx],
                avg_feature_overlap=avg_feature[idx],
                avg_structure_overlap=avg_structure[idx],
                avg_composite_similarity=avg_composite[idx],
            )
        )

    pairs: list[AlphaPairSimilarity] = []
    for i in range(n_analyzed):
        for j in range(i + 1, n_analyzed):
            pairs.append(
                AlphaPairSimilarity(
                    alpha_id_a=analyzed_records[i].alpha_id,
                    alpha_id_b=analyzed_records[j].alpha_id,
                    abs_signal_correlation=float(signal_overlap[i, j]),
                    feature_overlap=float(feature_overlap[i, j]),
                    structure_overlap=float(structure_overlap[i, j]),
                    composite_similarity=float(composite_similarity[i, j]),
                )
            )
    pairs.sort(key=lambda pair: pair.composite_similarity, reverse=True)

    input_pairs: list[InputPairSimilarity] = []
    for i in range(len(valid_input_features)):
        for j in range(i + 1, len(valid_input_features)):
            input_pairs.append(
                InputPairSimilarity(
                    feature_a=valid_input_features[i],
                    feature_b=valid_input_features[j],
                    abs_input_correlation=float(input_overlap[i, j]),
                )
            )
    input_pairs.sort(key=lambda pair: pair.abs_input_correlation, reverse=True)

    mean_signal = _offdiag_mean(signal_overlap)
    mean_feature = _offdiag_mean(feature_overlap)
    mean_structure = _offdiag_mean(structure_overlap)
    mean_composite = _offdiag_mean(composite_similarity)
    mean_input = _offdiag_mean(input_overlap)
    summary = DiversitySummary(
        n_records=len(records),
        n_analyzed=n_analyzed,
        n_skipped=len(skipped_ids),
        lookback=lookback,
        mean_abs_signal_correlation=mean_signal,
        mean_feature_overlap=mean_feature,
        mean_structure_overlap=mean_structure,
        mean_composite_similarity=mean_composite,
        signal_diversity=1.0 - mean_signal,
        feature_diversity=1.0 - mean_feature,
        structure_diversity=1.0 - mean_structure,
        composite_diversity=1.0 - mean_composite,
        n_unique_features=len(valid_input_features),
        mean_abs_input_correlation=mean_input,
        input_diversity=1.0 - mean_input,
        family_counts=dict(sorted(family_counts.items())),
        feature_usage_counts=dict(
            sorted(feature_usage_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    )
    rows.sort(key=lambda row: row.avg_composite_similarity, reverse=True)
    return DiversityReport(
        summary=summary,
        rows=rows,
        top_redundant_pairs=pairs[:top_pairs],
        top_input_pairs=input_pairs[:top_pairs],
        skipped_alpha_ids=skipped_ids,
    )
