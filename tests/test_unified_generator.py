from alpha_os.config import Config
from alpha_os.daemon.hypothesis_seeder import HypothesisSeederDaemon
from alpha_os.dsl.expr import Constant, Feature, LagOp
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
        lambda asset, client=None: ["fear_greed", "dxy", "gold"],
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.AlphaGenerator",
        _FakeAlphaGenerator,
    )

    cfg = Config()
    cfg.alpha_generator.pop_size = 2
    cfg.alpha_generator.feature_subset_k = 3

    store = HypothesisStore(tmp_path / "hypotheses.db")
    daemon = HypothesisSeederDaemon(cfg, store=store, client=None)

    stats = daemon._run_round()
    rows = store.list_all()

    assert stats.generated_dsl == 2
    assert stats.inserted_dsl == 2
    assert stats.inserted_bootstrap == 11
    assert len(rows) == 13
    assert len([row for row in rows if row.kind == "dsl"]) == 2
    assert len([row for row in rows if row.kind == "technical"]) == 9
    assert len([row for row in rows if row.kind == "ml"]) == 2
    assert all(row.status == "active" for row in rows)
    assert all(row.stake > 0 for row in rows)

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
        lambda asset, client=None: ["fear_greed", "dxy", "gold"],
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.AlphaGenerator",
        _InvalidAlphaGenerator,
    )

    cfg = Config()
    cfg.alpha_generator.pop_size = 2
    cfg.alpha_generator.feature_subset_k = 3

    store = HypothesisStore(tmp_path / "hypotheses.db")
    daemon = HypothesisSeederDaemon(cfg, store=store, client=None)

    stats = daemon._run_round()
    rows = store.list_all()

    assert stats.generated_dsl == 2
    assert stats.inserted_dsl == 1
    assert stats.skipped_dsl == 1
    assert len([row for row in rows if row.kind == "dsl"]) == 1

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
        lambda asset, client=None: ["fear_greed", "dxy", "gold"],
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.AlphaGenerator",
        _FakeAlphaGenerator,
    )

    cfg = Config()
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

    assert inserted == 10
    assert skipped == 1
    assert updated is not None
    assert updated.stake == 0.37
    assert updated.metadata["seed_family"] == original.metadata["seed_family"]
    assert updated.metadata["prior_quality_source"] == "bootstrap_seed"
    assert updated.metadata["oos_sharpe"] == original.metadata["oos_sharpe"]
    assert updated.metadata["oos_log_growth"] == original.metadata["oos_log_growth"]

    daemon.close()


def test_hypothesis_seeder_retires_obsolete_bootstrap_hypothesis(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_signal_client_from_config",
        lambda config: None,
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.build_feature_list",
        lambda asset, client=None: ["fear_greed", "dxy", "gold"],
    )
    monkeypatch.setattr(
        "alpha_os.daemon.hypothesis_seeder.AlphaGenerator",
        _FakeAlphaGenerator,
    )

    cfg = Config()
    store = HypothesisStore(tmp_path / "hypotheses.db")
    daemon = HypothesisSeederDaemon(cfg, store=store, client=None)

    store.register(
        HypothesisRecord(
            hypothesis_id="technical_volume_price_confirmation",
            kind="technical",
            name="Volume Price Confirmation",
            definition={
                "indicator": "volume_price_confirmation",
                "params": {"price_window": 20, "volume_window": 20},
            },
            stake=1.0,
            status="active",
            source="bootstrap_technical",
        )
    )

    daemon._register_bootstrap_hypotheses()
    retired = store.get("technical_volume_price_confirmation")

    assert retired is not None
    assert retired.status == "archived"
    assert retired.stake == 0.0
    assert "retired_bootstrap_reason" in retired.metadata

    daemon.close()
