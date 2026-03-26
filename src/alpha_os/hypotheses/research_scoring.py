from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..backtest.cost_model import CostModel
from ..backtest.engine import BacktestEngine
from ..data.universe import required_raw_signals
from ..dsl import parse
from ..dsl.evaluator import EvaluationError, evaluate_expression
from ..validation.purged_cv import purged_walk_forward
from .identity import (
    expression_feature_families,
    expression_feature_names,
    representative_feature_family,
)
from .serious_templates import (
    serious_family_gap_scores,
    serious_seed_specs,
    serious_template_feature_gap_scores,
    serious_template_gap_scores,
)
from .store import HypothesisKind, HypothesisRecord


@dataclass(frozen=True)
class ResearchScoreUpdate:
    hypothesis_id: str
    oos_sharpe: float
    oos_log_growth: float
    n_folds: int
    research_quality_source: str = "batch_research_score"
    research_quality_status: str = "scored"
    research_scored_at: float = 0.0

    def metadata_update(self) -> dict[str, float | str]:
        return {
            "oos_sharpe": float(self.oos_sharpe),
            "oos_log_growth": float(self.oos_log_growth),
            "prior_quality_source": self.research_quality_source,
            "research_quality_source": self.research_quality_source,
            "research_quality_status": self.research_quality_status,
            "research_scored_at": float(self.research_scored_at or time.time()),
            "research_score_n_folds": int(self.n_folds),
        }


@dataclass(frozen=True)
class ResearchScoreFailure:
    hypothesis_id: str
    reason: str


@dataclass(frozen=True)
class ResearchScoringBatch:
    updates: list[ResearchScoreUpdate]
    failures: list[ResearchScoreFailure]


def is_exploratory_unscored(record: HypothesisRecord) -> bool:
    return (
        record.status == "active"
        and record.kind == HypothesisKind.DSL
        and record.source == "random_dsl"
        and str(record.metadata.get("research_quality_status", "unscored")) == "unscored"
        and bool(expression_feature_names(record.expression))
    )


def exploratory_scoring_candidates(
    records: list[HypothesisRecord],
    *,
    asset: str | None = None,
    limit: int | None = None,
) -> list[HypothesisRecord]:
    candidates = [record for record in records if is_exploratory_unscored(record)]
    if not candidates:
        return []
    serious_gaps = serious_family_gap_scores(asset or "BTC", records)
    template_gaps = serious_template_gap_scores(asset or "BTC", records)
    template_feature_gaps = serious_template_feature_gap_scores(asset or "BTC", records)
    template_specs = serious_seed_specs(asset or "BTC")

    family_counts: Counter[str] = Counter()
    family_totals: Counter[str] = Counter()
    family_successes: Counter[str] = Counter()
    for record in records:
        if record.source != "random_dsl":
            continue
        if float(record.stake) <= 0 and not bool(
            record.metadata.get("lifecycle_actionable_live", False)
        ):
            continue
        for family in set(expression_feature_families(record.expression)):
            family_counts[family] += 1
    for record in records:
        if record.source != "random_dsl":
            continue
        if str(record.metadata.get("research_quality_status", "")) != "scored":
            continue
        families = set(expression_feature_families(record.expression))
        if not families:
            continue
        success = (
            float(record.stake) > 0
            or bool(record.metadata.get("lifecycle_research_retained", False))
            or bool(record.metadata.get("lifecycle_actionable_live", False))
        )
        for family in families:
            family_totals[family] += 1
            if success:
                family_successes[family] += 1

    def _family_quality(family: str) -> float:
        total = family_totals.get(family, 0)
        if total <= 0:
            return 0.5
        return float(family_successes.get(family, 0) / total)

    def _record_families(record: HypothesisRecord) -> tuple[str, ...]:
        families = tuple(sorted(set(expression_feature_families(record.expression))))
        return families

    def _record_template_gap(record: HypothesisRecord) -> float:
        features = set(expression_feature_names(record.expression))
        families = set(_record_families(record))
        if not features or not families:
            return 0.0
        best = 0.0
        for spec in template_specs:
            gap = float(template_gaps.get(spec.template_id, 0.0))
            if gap <= 0.0 or spec.family not in families:
                continue
            template_features = set(expression_feature_names(spec.expression))
            if not template_features:
                continue
            overlap = len(features & template_features)
            if overlap <= 0:
                continue
            score = gap * (float(overlap) / float(len(template_features)))
            if score > best:
                best = score
        return best

    def _record_template_feature_gap(record: HypothesisRecord) -> float:
        features = expression_feature_names(record.expression)
        if not features:
            return 0.0
        return float(
            sum(template_feature_gaps.get(feature, 0.0) for feature in features)
            / len(features)
        )

    def _priority(record: HypothesisRecord) -> tuple[float, float, float, float, str]:
        families = _record_families(record)
        template_gap = _record_template_gap(record)
        template_feature_gap = _record_template_feature_gap(record)
        overlap = float(sum(family_counts.get(family, 0) for family in families))
        novelty = float(sum(1 for family in families if family_counts.get(family, 0) == 0))
        if families:
            serious_gap = float(
                sum(serious_gaps.get(family, 0.0) for family in families) / len(families)
            )
            family_quality = float(sum(_family_quality(family) for family in families) / len(families))
        else:
            serious_gap = 0.0
            family_quality = 0.0
        return (
            overlap,
            -template_gap,
            -template_feature_gap,
            -serious_gap,
            -family_quality,
            -novelty,
            record.hypothesis_id,
        )

    candidates.sort(key=_priority)

    if limit is None:
        return candidates
    limit = max(limit, 0)
    if limit <= 0:
        return []

    buckets: dict[str, list[HypothesisRecord]] = {}
    for record in candidates:
        families = _record_families(record)
        if families:
            representative_family = representative_feature_family(
                sorted(
                    families,
                    key=lambda family: (
                        family_counts.get(family, 0),
                        -max(
                            (
                                template_gaps.get(spec.template_id, 0.0)
                                for spec in template_specs
                                if spec.family == family
                            ),
                            default=0.0,
                        ),
                        -serious_gaps.get(family, 0.0),
                        -_family_quality(family),
                        family,
                    ),
                )
            )
        else:
            representative_family = "other"
        buckets.setdefault(representative_family, []).append(record)

    if len(buckets) <= 1:
        return candidates[:limit]

    family_order = sorted(
        buckets,
        key=lambda family: (
            family_counts.get(family, 0),
            -max(
                (
                    template_gaps.get(spec.template_id, 0.0)
                    for spec in template_specs
                    if spec.family == family
                ),
                default=0.0,
            ),
            -serious_gaps.get(family, 0.0),
            -_family_quality(family),
            family,
        ),
    )
    diversified: list[HypothesisRecord] = []
    while len(diversified) < limit:
        added = False
        for family in family_order:
            bucket = buckets[family]
            if not bucket:
                continue
            diversified.append(bucket.pop(0))
            added = True
            if len(diversified) >= limit:
                break
        if not added:
            break
    return diversified


def required_research_features(records: list[HypothesisRecord], price_feature: str) -> list[str]:
    feature_names: set[str] = {price_feature}
    for record in records:
        expression = record.expression
        if not expression:
            continue
        feature_names.update(expression_feature_names(expression))
    ordered = [price_feature]
    ordered.extend(
        name for name in required_raw_signals(feature_names)
        if name != price_feature
    )
    return ordered


def score_exploratory_hypotheses(
    records: list[HypothesisRecord],
    *,
    data: dict[str, np.ndarray],
    prices: np.ndarray,
    commission_pct: float,
    slippage_pct: float,
    allow_short: bool,
    n_cv_folds: int,
    embargo_days: int,
) -> ResearchScoringBatch:
    n_days = len(prices)
    engine = BacktestEngine(
        CostModel(commission_pct, slippage_pct),
        allow_short=allow_short,
    )
    updates: list[ResearchScoreUpdate] = []
    failures: list[ResearchScoreFailure] = []

    for record in records:
        expression = record.expression
        if not expression:
            failures.append(ResearchScoreFailure(record.hypothesis_id, "missing expression"))
            continue
        try:
            expr = parse(expression)
            signal = evaluate_expression(expr, data, n_days)
            cv = purged_walk_forward(
                signal,
                prices,
                engine,
                n_folds=n_cv_folds,
                embargo=embargo_days,
            )
            updates.append(
                ResearchScoreUpdate(
                    hypothesis_id=record.hypothesis_id,
                    oos_sharpe=float(cv.oos_sharpe),
                    oos_log_growth=float(cv.oos_expected_log_growth),
                    n_folds=int(cv.n_folds),
                )
            )
        except (EvaluationError, Exception) as exc:
            failures.append(ResearchScoreFailure(record.hypothesis_id, str(exc)[:200]))

    return ResearchScoringBatch(updates=updates, failures=failures)
