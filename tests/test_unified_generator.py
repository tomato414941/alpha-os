import pytest

from alpha_os.config import Config
from alpha_os.daemon.hypothesis_seeder import HypothesisSeederDaemon, RANDOM_DSL_METADATA
from alpha_os.dsl.expr import Constant, Feature, LagOp
from alpha_os.dsl import to_string
from alpha_os.hypotheses.bootstrap import bootstrap_hypotheses
from alpha_os.hypotheses import HypothesisStore
from alpha_os.hypotheses.store import HypothesisRecord


class _FakeAlphaGenerator:
    def __init__(self, features, feature_subset=None, seed=None):
        self.features = features

    def generate_random(self, n, max_depth=3):
        return self.features[:n]


def test_hypothesis_seeder_registers_random_dsl_and_bootstrap_hypotheses(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_signal_client_from_config",
        lambda config: None,
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_feature_list",
        lambda asset, client=None, prefer_cache=False: ["fear_greed", "dxy", "gold"],
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.AlphaGenerator",
        _FakeAlphaGenerator,
    )

    cfg = Config()
    cfg.alpha_generator.pop_size = 2
    cfg.alpha_generator.feature_subset_k = 3
    cfg.alpha_generator.min_feature_catalog_size = 1

    store = HypothesisStore(tmp_path / "hypotheses.db")
    daemon = HypothesisSeederDaemon(cfg, store=store, client=None)

    stats = daemon._run_round()
    rows = store.list_all()
    dsl_rows = [row for row in rows if row.kind == "dsl" and row.source == "random_dsl"]
    serious_rows = [row for row in rows if row.source == "bootstrap_serious"]
    technical_rows = [row for row in rows if row.kind == "technical"]
    ml_rows = [row for row in rows if row.kind == "ml"]

    assert stats.generated_dsl == 2
    assert stats.inserted_dsl == 2
    assert stats.inserted_bootstrap == 19
    assert len(rows) == 21
    assert len(dsl_rows) == 2
    assert len(serious_rows) == 9
    assert len(technical_rows) == 8
    assert len(ml_rows) == 2
    assert all(row.status == "active" for row in rows)
    assert all(row.stake == 0 for row in dsl_rows)
    assert all(row.metadata["research_quality_source"] == "exploratory_unscored" for row in dsl_rows)
    assert all(row.metadata["research_quality_status"] == "unscored" for row in dsl_rows)
    assert all(row.scope["asset"] == "BTC" for row in dsl_rows)
    assert all(row.stake == 0.0 for row in serious_rows)
    assert all(row.stake > 0 for row in technical_rows + ml_rows)

    daemon.close()


def test_hypothesis_seeder_can_seed_non_btc_observation_only_sleeve(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_signal_client_from_config",
        lambda config: None,
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_feature_list",
        lambda asset, client=None, prefer_cache=False, refresh_catalog=False: [
            "eth_price",
            "eth_volume",
            "eth_open_interest",
        ],
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.AlphaGenerator",
        _FakeAlphaGenerator,
    )

    cfg = Config()
    cfg.alpha_generator.pop_size = 2
    cfg.alpha_generator.feature_subset_k = 3
    cfg.alpha_generator.min_feature_catalog_size = 1

    store = HypothesisStore(tmp_path / "hypotheses.db")
    daemon = HypothesisSeederDaemon(
        cfg,
        primary_asset="ETH",
        include_bootstrap=False,
        store=store,
        client=None,
    )

    stats = daemon._run_round()
    rows = store.list_all()

    assert stats.generated_dsl == 2
    assert stats.inserted_dsl == 2
    assert stats.inserted_bootstrap == 0
    assert len(rows) == 2
    assert all(row.scope["asset"] == "ETH" for row in rows)
    assert all(row.stake == 0.0 for row in rows)

    daemon.close()


class _InvalidAlphaGenerator:
    def __init__(self, features, feature_subset=None, seed=None):
        self.features = features

    def generate_random(self, n, max_depth=3):
        return [
            Feature(self.features[0]),
            LagOp("lag", 30, Constant(1.17)),
        ][:n]


def test_hypothesis_seeder_skips_structurally_invalid_dsl_candidates(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_signal_client_from_config",
        lambda config: None,
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_feature_list",
        lambda asset, client=None, prefer_cache=False: ["fear_greed", "dxy", "gold"],
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.AlphaGenerator",
        _InvalidAlphaGenerator,
    )

    cfg = Config()
    cfg.alpha_generator.pop_size = 2
    cfg.alpha_generator.feature_subset_k = 3
    cfg.alpha_generator.min_feature_catalog_size = 1

    store = HypothesisStore(tmp_path / "hypotheses.db")
    daemon = HypothesisSeederDaemon(cfg, store=store, client=None)

    stats = daemon._run_round()
    rows = store.list_all()

    assert stats.generated_dsl == 2
    assert stats.inserted_dsl == 1
    assert stats.skipped_dsl == 1
    assert len([row for row in rows if row.kind == "dsl" and row.source == "random_dsl"]) == 1

    daemon.close()


def test_hypothesis_seeder_backfills_bootstrap_prior_quality_for_existing_records(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_signal_client_from_config",
        lambda config: None,
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_feature_list",
        lambda asset, client=None, prefer_cache=False: ["fear_greed", "dxy", "gold"],
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.AlphaGenerator",
        _FakeAlphaGenerator,
    )

    cfg = Config()
    cfg.alpha_generator.min_feature_catalog_size = 1
    store = HypothesisStore(tmp_path / "hypotheses.db")
    daemon = HypothesisSeederDaemon(cfg, store=store, client=None)

    original = bootstrap_hypotheses()[0]
    store.register(
        HypothesisRecord(
            hypothesis_id=original.hypothesis_id,
            kind=original.kind,
            name=original.name,
            definition=original.definition,
            status=original.status,
            stake=0.37,
            source=original.source,
            metadata={"seed_family": original.metadata["seed_family"]},
        )
    )

    inserted, skipped = daemon._register_bootstrap_hypotheses()
    updated = store.get(original.hypothesis_id)

    assert inserted == 18
    assert skipped == 1
    assert updated is not None
    assert updated.stake == 0.37
    assert updated.metadata["seed_family"] == original.metadata["seed_family"]
    assert updated.metadata["prior_quality_source"] == "bootstrap_seed"
    assert updated.metadata["oos_sharpe"] == original.metadata["oos_sharpe"]
    assert updated.metadata["oos_log_growth"] == original.metadata["oos_log_growth"]

    daemon.close()


def test_hypothesis_seeder_backfills_exploratory_metadata_for_existing_random_dsl(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_signal_client_from_config",
        lambda config: None,
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_feature_list",
        lambda asset, client=None, prefer_cache=False: ["fear_greed", "dxy", "gold"],
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.AlphaGenerator",
        _FakeAlphaGenerator,
    )

    cfg = Config()
    cfg.alpha_generator.min_feature_catalog_size = 1
    store = HypothesisStore(tmp_path / "hypotheses.db")
    daemon = HypothesisSeederDaemon(cfg, store=store, client=None)
    expr = Feature("fear_greed")
    expression = to_string(expr)
    hypothesis_id = daemon._dsl_hypothesis_id(expression)

    store.register(
        HypothesisRecord(
            hypothesis_id=hypothesis_id,
            kind="dsl",
            name=expression,
            definition={"expression": expression},
            status="active",
            stake=0.0,
            source="random_dsl",
            metadata={"generator": "hypothesis-seeder"},
        )
    )

    inserted, skipped = daemon._register_random_dsl([expr])
    updated = store.get(hypothesis_id)

    assert inserted == 0
    assert skipped == 1
    assert updated is not None
    assert updated.metadata["research_quality_source"] == "exploratory_unscored"
    assert updated.metadata["research_quality_status"] == "unscored"
    assert updated.metadata["registration_stage"] == "observation_only"

    daemon.close()


def test_hypothesis_seeder_skips_random_dsl_that_matches_live_diversity_key(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_signal_client_from_config",
        lambda config: None,
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_feature_list",
        lambda asset, client=None, prefer_cache=False, refresh_catalog=False: ["fear_greed", "dxy", "btc_usdt"],
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.AlphaGenerator",
        _FakeAlphaGenerator,
    )

    cfg = Config()
    cfg.alpha_generator.min_feature_catalog_size = 1
    store = HypothesisStore(tmp_path / "hypotheses.db")
    daemon = HypothesisSeederDaemon(cfg, store=store, client=None)

    existing_expression = to_string(Feature("fear_greed"))
    store.register(
        HypothesisRecord(
            hypothesis_id=daemon._dsl_hypothesis_id(existing_expression),
            kind="dsl",
            name=existing_expression,
            definition={"expression": existing_expression},
            status="active",
            stake=0.0,
            source="random_dsl",
            metadata={
                **RANDOM_DSL_METADATA,
                "lifecycle_actionable_live": True,
            },
        )
    )

    inserted, skipped = daemon._register_random_dsl(
        [Feature("dxy"), Feature("btc_usdt")]
    )
    rows = [row for row in store.list_all() if row.source == "random_dsl"]

    assert inserted == 1
    assert skipped == 1
    assert len(rows) == 2
    assert any(row.definition["expression"] == to_string(Feature("btc_usdt")) for row in rows)

    daemon.close()


def test_hypothesis_seeder_skips_featureless_random_dsl(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_signal_client_from_config",
        lambda config: None,
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_feature_list",
        lambda asset, client=None, prefer_cache=False, refresh_catalog=False: ["fear_greed"],
    )

    cfg = Config()
    cfg.alpha_generator.min_feature_catalog_size = 1
    store = HypothesisStore(tmp_path / "hypotheses.db")
    daemon = HypothesisSeederDaemon(cfg, store=store, client=None)

    inserted, skipped = daemon._register_random_dsl([Constant(1.23), Feature("fear_greed")])
    rows = [row for row in store.list_all() if row.source == "random_dsl"]

    assert inserted == 1
    assert skipped == 1
    assert len(rows) == 1
    assert rows[0].definition["expression"] == "fear_greed"

    daemon.close()


def test_hypothesis_seeder_retries_api_when_cached_catalog_is_too_small(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_signal_client_from_config",
        lambda config: None,
    )
    calls: list[tuple[bool, bool]] = []

    def _build_feature_list(asset, client=None, prefer_cache=False, refresh_catalog=False):
        calls.append((prefer_cache, refresh_catalog))
        if prefer_cache:
            return ["btc_ohlcv", "sig_60", "sig_86400"]
        return ["btc_ohlcv"] + [f"sig_{idx}" for idx in range(150)]

    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_feature_list",
        _build_feature_list,
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.AlphaGenerator",
        _FakeAlphaGenerator,
    )

    cfg = Config()
    cfg.alpha_generator.pop_size = 2
    cfg.alpha_generator.feature_subset_k = 3
    cfg.alpha_generator.min_feature_catalog_size = 100

    store = HypothesisStore(tmp_path / "hypotheses.db")
    daemon = HypothesisSeederDaemon(cfg, store=store, client=None)

    stats = daemon._run_round()

    assert calls == [(True, False), (False, True)]
    assert stats.generated_dsl == 2
    assert stats.inserted_dsl == 2

    daemon.close()


def test_hypothesis_seeder_guided_feature_subset_prefers_stronger_families(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_signal_client_from_config",
        lambda config: None,
    )

    cfg = Config()
    store = HypothesisStore(tmp_path / "hypotheses.db")
    daemon = HypothesisSeederDaemon(cfg, store=store, client=None)

    store.register(
        HypothesisRecord(
            hypothesis_id="onchain_good",
            kind="dsl",
            name="btc_hashrate",
            definition={"expression": "btc_hashrate"},
            status="active",
            stake=1.0,
            source="random_dsl",
            metadata={
                "research_quality_status": "scored",
                "lifecycle_research_retained": True,
            },
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="event_weak",
            kind="dsl",
            name="gdelt_doc_crypto",
            definition={"expression": "gdelt_doc_crypto"},
            status="active",
            stake=0.0,
            source="random_dsl",
            metadata={
                "research_quality_status": "scored",
            },
        )
    )

    subset = daemon._guided_feature_subset(
        ["btc_hashrate", "gdelt_doc_crypto"],
        k=1,
        seed=7,
    )

    assert subset == frozenset({"btc_hashrate"})

    daemon.close()


@pytest.mark.parametrize(
    ("hypothesis_id", "name", "definition"),
    [
        (
            "technical_volume_price_confirmation",
            "Volume Price Confirmation",
            {
                "indicator": "volume_price_confirmation",
                "params": {"price_window": 20, "volume_window": 20},
            },
        ),
        (
            "technical_roc_5_mean_reversion",
            "ROC 5 Mean Reversion",
            {
                "indicator": "roc_reversion",
                "params": {"window": 5},
            },
        ),
    ],
)
def test_hypothesis_seeder_retires_obsolete_bootstrap_hypothesis(
    tmp_path,
    monkeypatch,
    hypothesis_id,
    name,
    definition,
):
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_signal_client_from_config",
        lambda config: None,
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_feature_list",
        lambda asset, client=None, prefer_cache=False, refresh_catalog=False: ["fear_greed", "dxy", "gold"],
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.AlphaGenerator",
        _FakeAlphaGenerator,
    )

    cfg = Config()
    cfg.alpha_generator.min_feature_catalog_size = 1
    store = HypothesisStore(tmp_path / "hypotheses.db")
    daemon = HypothesisSeederDaemon(cfg, store=store, client=None)

    store.register(
        HypothesisRecord(
            hypothesis_id=hypothesis_id,
            kind="technical",
            name=name,
            definition=definition,
            stake=1.0,
            status="active",
            source="bootstrap_technical",
        )
    )

    daemon._register_bootstrap_hypotheses()
    retired = store.get(hypothesis_id)

    assert retired is not None
    assert retired.status == "archived"
    assert retired.stake == 0.0
    assert "retired_bootstrap_reason" in retired.metadata

    daemon.close()
