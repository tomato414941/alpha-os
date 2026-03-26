from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..data.universe import price_signal
from .research_scoring import (
    ResearchScoringBatch,
    exploratory_scoring_candidates,
    required_research_features,
    score_exploratory_hypotheses,
)


@dataclass(frozen=True)
class ExploratoryResearchScoringRun:
    asset: str
    dry_run: bool
    candidate_count: int
    batch: ResearchScoringBatch


def run_exploratory_research_scoring(
    *,
    store,
    config: Config,
    asset: str,
    limit: int | None,
    dry_run: bool,
    load_data,
) -> ExploratoryResearchScoringRun:
    candidates = exploratory_scoring_candidates(
        store.list_observation_active(asset=asset),
        limit=limit,
    )
    if not candidates:
        return ExploratoryResearchScoringRun(
            asset=asset.upper(),
            dry_run=dry_run,
            candidate_count=0,
            batch=ResearchScoringBatch(updates=[], failures=[]),
        )

    price_feature = price_signal(asset)
    features = required_research_features(candidates, price_feature)
    data, _ = load_data(
        features,
        config,
        eval_window=config.backtest.eval_window_days,
    )
    prices = np.asarray(data[price_feature], dtype=float)
    batch = score_exploratory_hypotheses(
        candidates,
        data=data,
        prices=prices,
        commission_pct=config.backtest.commission_pct,
        slippage_pct=config.backtest.slippage_pct,
        allow_short=config.trading.supports_short,
        n_cv_folds=config.validation.n_cv_folds,
        embargo_days=config.validation.embargo_days,
    )

    if not dry_run:
        for update in batch.updates:
            store.update_metadata(
                update.hypothesis_id,
                update.metadata_update(),
                merge=True,
            )

    return ExploratoryResearchScoringRun(
        asset=asset.upper(),
        dry_run=dry_run,
        candidate_count=len(candidates),
        batch=batch,
    )
