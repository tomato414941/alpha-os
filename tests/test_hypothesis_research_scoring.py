from __future__ import annotations

from argparse import Namespace

import numpy as np

from alpha_os.config import Config
from alpha_os.cli import _build_parser, cmd_score_exploratory_hypotheses
from alpha_os.data.universe import price_signal
from alpha_os.hypotheses.research_scoring import (
    exploratory_scoring_candidates,
    score_exploratory_hypotheses,
    required_research_features,
)
from alpha_os.hypotheses.store import HypothesisKind, HypothesisRecord, HypothesisStore


def _scoring_data(asset: str = "BTC", n_days: int = 260) -> tuple[dict[str, np.ndarray], int]:
    p = price_signal(asset)
    returns = np.array([0.001 if idx % 2 == 0 else 0.002 for idx in range(n_days - 1)])
    prices = 100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + returns)))
    data = {
        p: prices,
        "fear_greed": np.ones(n_days, dtype=float),
    }
    return data, n_days


def test_exploratory_scoring_candidates_filter_and_features(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="dsl_candidate",
            kind=HypothesisKind.DSL,
            definition={"expression": "fear_greed"},
            source="random_dsl",
            stake=0.0,
            metadata={"research_quality_status": "unscored"},
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="dsl_scored",
            kind=HypothesisKind.DSL,
            definition={"expression": "fear_greed"},
            source="random_dsl",
            stake=0.0,
            metadata={"research_quality_status": "scored"},
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="bootstrap_ml",
            kind=HypothesisKind.ML,
            definition={"model_ref": "m1"},
            source="bootstrap_ml",
            stake=1.0,
        )
    )

    candidates = exploratory_scoring_candidates(store.list_observation_active())

    assert [record.hypothesis_id for record in candidates] == ["dsl_candidate"]
    assert required_research_features(candidates, price_signal("BTC")) == [
        price_signal("BTC"),
        "fear_greed",
    ]
    store.close()


def test_exploratory_scoring_candidates_skip_featureless_random_dsl(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="dsl_constant",
            kind=HypothesisKind.DSL,
            definition={"expression": "1.23"},
            source="random_dsl",
            stake=0.0,
            metadata={"research_quality_status": "unscored"},
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="dsl_featured",
            kind=HypothesisKind.DSL,
            definition={"expression": "fear_greed"},
            source="random_dsl",
            stake=0.0,
            metadata={"research_quality_status": "unscored"},
        )
    )

    candidates = exploratory_scoring_candidates(store.list_observation_active())

    assert [record.hypothesis_id for record in candidates] == ["dsl_featured"]
    store.close()


def test_exploratory_scoring_candidates_prioritize_novel_families(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="incumbent",
            kind=HypothesisKind.DSL,
            definition={"expression": "fear_greed"},
            source="random_dsl",
            stake=1.0,
            metadata={
                "lifecycle_actionable_live": True,
                "research_quality_status": "scored",
            },
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="macro_candidate",
            kind=HypothesisKind.DSL,
            definition={"expression": "dxy"},
            source="random_dsl",
            stake=0.0,
            metadata={"research_quality_status": "unscored"},
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="novel_candidate",
            kind=HypothesisKind.DSL,
            definition={"expression": "btc_difficulty"},
            source="random_dsl",
            stake=0.0,
            metadata={"research_quality_status": "unscored"},
        )
    )

    candidates = exploratory_scoring_candidates(store.list_observation_active())

    assert [record.hypothesis_id for record in candidates] == [
        "novel_candidate",
        "macro_candidate",
    ]
    store.close()


def test_exploratory_scoring_candidates_prefer_families_with_better_scored_outcomes(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="onchain_scored_good",
            kind=HypothesisKind.DSL,
            definition={"expression": "btc_difficulty"},
            source="random_dsl",
            stake=0.0,
            metadata={
                "research_quality_status": "scored",
                "lifecycle_research_retained": True,
            },
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="event_scored_weak",
            kind=HypothesisKind.DSL,
            definition={"expression": "gdelt_sentiment_btc"},
            source="random_dsl",
            stake=0.0,
            metadata={
                "research_quality_status": "scored",
            },
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="event_candidate",
            kind=HypothesisKind.DSL,
            definition={"expression": "nasa_solar_flux"},
            source="random_dsl",
            stake=0.0,
            metadata={"research_quality_status": "unscored"},
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="onchain_candidate",
            kind=HypothesisKind.DSL,
            definition={"expression": "btc_hashrate"},
            source="random_dsl",
            stake=0.0,
            metadata={"research_quality_status": "unscored"},
        )
    )

    candidates = exploratory_scoring_candidates(store.list_observation_active())

    assert [record.hypothesis_id for record in candidates] == [
        "onchain_candidate",
        "event_candidate",
    ]
    store.close()


def test_exploratory_scoring_candidates_prefer_undercovered_serious_families(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    for idx in range(2):
        store.register(
            HypothesisRecord(
                hypothesis_id=f"serious_macro_{idx}",
                kind=HypothesisKind.DSL,
                definition={"expression": "fear_greed"},
                source="bootstrap_serious",
                stake=0.0,
                scope={"asset": "BTC"},
                metadata={
                    "serious_family": "macro",
                    "lifecycle_capital_backed": True,
                },
            )
        )
    store.register(
        HypothesisRecord(
            hypothesis_id="macro_candidate",
            kind=HypothesisKind.DSL,
            definition={"expression": "dxy"},
            source="random_dsl",
            stake=0.0,
            scope={"asset": "BTC"},
            metadata={"research_quality_status": "unscored"},
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="derivatives_candidate",
            kind=HypothesisKind.DSL,
            definition={"expression": "funding_rate_btc"},
            source="random_dsl",
            stake=0.0,
            scope={"asset": "BTC"},
            metadata={"research_quality_status": "unscored"},
        )
    )

    candidates = exploratory_scoring_candidates(
        store.list_observation_active(asset="BTC"),
        asset="BTC",
    )

    assert [record.hypothesis_id for record in candidates] == [
        "derivatives_candidate",
        "macro_candidate",
    ]
    store.close()


def test_exploratory_scoring_candidates_prefer_missing_serious_template_within_family(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="serious_macro_sentiment",
            kind=HypothesisKind.DSL,
            definition={"expression": "fear_greed"},
            source="bootstrap_serious",
            stake=0.0,
            scope={"asset": "BTC"},
            metadata={
                "serious_family": "macro",
                "serious_template": "macro_sentiment_acceleration",
                "lifecycle_capital_backed": True,
            },
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="macro_sentiment_candidate",
            kind=HypothesisKind.DSL,
            definition={"expression": "fear_greed"},
            source="random_dsl",
            stake=0.0,
            scope={"asset": "BTC"},
            metadata={"research_quality_status": "unscored"},
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="macro_dollar_candidate",
            kind=HypothesisKind.DSL,
            definition={"expression": "dxy"},
            source="random_dsl",
            stake=0.0,
            scope={"asset": "BTC"},
            metadata={"research_quality_status": "unscored"},
        )
    )

    candidates = exploratory_scoring_candidates(
        store.list_observation_active(asset="BTC"),
        asset="BTC",
    )

    assert [record.hypothesis_id for record in candidates] == [
        "macro_dollar_candidate",
        "macro_sentiment_candidate",
    ]
    store.close()


def test_exploratory_scoring_candidates_limit_preserves_family_diversity(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="price_scored_good",
            kind=HypothesisKind.DSL,
            definition={"expression": "fear_greed"},
            source="random_dsl",
            stake=0.0,
            metadata={
                "research_quality_status": "scored",
                "lifecycle_research_retained": True,
            },
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="onchain_scored_good",
            kind=HypothesisKind.DSL,
            definition={"expression": "btc_difficulty"},
            source="random_dsl",
            stake=0.0,
            metadata={
                "research_quality_status": "scored",
                "lifecycle_research_retained": True,
            },
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="price_candidate_a",
            kind=HypothesisKind.DSL,
            definition={"expression": "dxy"},
            source="random_dsl",
            stake=0.0,
            metadata={"research_quality_status": "unscored"},
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="price_candidate_b",
            kind=HypothesisKind.DSL,
            definition={"expression": "gold"},
            source="random_dsl",
            stake=0.0,
            metadata={"research_quality_status": "unscored"},
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="onchain_candidate",
            kind=HypothesisKind.DSL,
            definition={"expression": "btc_hashrate"},
            source="random_dsl",
            stake=0.0,
            metadata={"research_quality_status": "unscored"},
        )
    )

    candidates = exploratory_scoring_candidates(
        store.list_observation_active(),
        limit=2,
    )

    assert sorted(record.hypothesis_id for record in candidates) == [
        "onchain_candidate",
        "price_candidate_a",
    ]
    store.close()


def test_score_exploratory_hypotheses_parser():
    parser = _build_parser()
    args = parser.parse_args(["score-exploratory-hypotheses", "--asset", "BTC", "--dry-run"])

    assert args.command == "score-exploratory-hypotheses"
    assert args.asset == "BTC"
    assert args.dry_run is True
    assert args.limit is None


def test_required_research_features_and_scoring_support_derived_daily_features(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    record = HypothesisRecord(
        hypothesis_id="dsl_derived",
        kind=HypothesisKind.DSL,
        definition={"expression": "roc_5__funding_rate_btc"},
        source="random_dsl",
        stake=0.0,
        metadata={"research_quality_status": "unscored"},
    )
    store.register(record)

    candidates = exploratory_scoring_candidates(store.list_observation_active())
    features = required_research_features(candidates, price_signal("BTC"))
    assert features == [price_signal("BTC"), "funding_rate_btc"]

    n_days = 260
    returns = np.array([0.0015 if idx % 2 == 0 else 0.0005 for idx in range(n_days - 1)])
    prices = 100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + returns)))
    data = {
        price_signal("BTC"): prices,
        "funding_rate_btc": np.linspace(0.0, 1.0, n_days, dtype=float),
    }
    batch = score_exploratory_hypotheses(
        candidates,
        data=data,
        prices=prices,
        commission_pct=0.0,
        slippage_pct=0.0,
        allow_short=False,
        n_cv_folds=3,
        embargo_days=1,
    )

    assert [update.hypothesis_id for update in batch.updates] == ["dsl_derived"]
    assert batch.failures == []
    store.close()


def test_cmd_score_exploratory_hypotheses_dry_run_and_apply(monkeypatch, tmp_path, capsys):
    hdb = tmp_path / "hypotheses.db"
    store = HypothesisStore(hdb)
    store.register(
        HypothesisRecord(
            hypothesis_id="dsl_candidate",
            kind=HypothesisKind.DSL,
            definition={"expression": "fear_greed"},
            source="random_dsl",
            stake=0.0,
            metadata={"research_quality_status": "unscored"},
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="bootstrap_ml",
            kind=HypothesisKind.ML,
            definition={"model_ref": "m1"},
            source="bootstrap_ml",
            stake=1.0,
            metadata={"oos_sharpe": 0.8},
        )
    )
    store.close()

    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", hdb)
    monkeypatch.setattr("alpha_os.cli._load_config", lambda _path: Config())
    monkeypatch.setattr("alpha_os.cli._real_data", lambda features, config, eval_window=0: _scoring_data())

    cmd_score_exploratory_hypotheses(
        Namespace(asset="BTC", config=None, limit=None, dry_run=True)
    )
    out = capsys.readouterr().out
    assert "Research scoring [DRY RUN]: asset=BTC candidates=1 scored=1 failed=0" in out

    store = HypothesisStore(hdb)
    dry_run_record = store.get("dsl_candidate")
    assert dry_run_record is not None
    assert "oos_sharpe" not in dry_run_record.metadata
    assert dry_run_record.metadata["research_quality_status"] == "unscored"
    store.close()

    cmd_score_exploratory_hypotheses(
        Namespace(asset="BTC", config=None, limit=None, dry_run=False)
    )
    out = capsys.readouterr().out
    assert "Research scoring [APPLY]: asset=BTC candidates=1 scored=1 failed=0" in out
    assert "Research scoring summary: updated=1 active=2" in out

    store = HypothesisStore(hdb)
    updated = store.get("dsl_candidate")
    assert updated is not None
    assert updated.metadata["research_quality_status"] == "scored"
    assert updated.metadata["research_quality_source"] == "batch_research_score"
    assert updated.metadata["prior_quality_source"] == "batch_research_score"
    assert updated.metadata["research_score_n_folds"] > 0
    assert float(updated.metadata["oos_sharpe"]) >= 0.0
    assert float(updated.metadata["oos_log_growth"]) >= 0.0
    store.close()
