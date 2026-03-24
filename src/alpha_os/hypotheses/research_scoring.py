from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..backtest.cost_model import CostModel
from ..backtest.engine import BacktestEngine
from ..dsl import parse
from ..dsl.evaluator import EvaluationError, evaluate_expression
from ..validation.purged_cv import purged_walk_forward
from .identity import expression_feature_families, expression_feature_names
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
    )


def exploratory_scoring_candidates(
    records: list[HypothesisRecord],
    *,
    limit: int | None = None,
) -> list[HypothesisRecord]:
    candidates = [record for record in records if is_exploratory_unscored(record)]
    if not candidates:
        return []

    family_counts: Counter[str] = Counter()
    for record in records:
        if record.source != "random_dsl":
            continue
        if float(record.stake) <= 0 and not bool(
            record.metadata.get("lifecycle_actionable_live", False)
        ):
            continue
        for family in set(expression_feature_families(record.expression)):
            family_counts[family] += 1

    if family_counts:
        def _priority(record: HypothesisRecord) -> tuple[float, float, str]:
            families = set(expression_feature_families(record.expression))
            overlap = float(sum(family_counts.get(family, 0) for family in families))
            novelty = float(sum(1 for family in families if family_counts.get(family, 0) == 0))
            return overlap, -novelty, record.hypothesis_id

        candidates.sort(key=_priority)

    if limit is None:
        return candidates
    return candidates[: max(limit, 0)]


def required_research_features(records: list[HypothesisRecord], price_feature: str) -> list[str]:
    feature_names: set[str] = {price_feature}
    for record in records:
        expression = record.expression
        if not expression:
            continue
        feature_names.update(expression_feature_names(expression))
    ordered = [price_feature]
    ordered.extend(sorted(name for name in feature_names if name != price_feature))
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
