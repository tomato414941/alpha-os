from alpha_os.hypotheses.bootstrap import bootstrap_hypotheses
from alpha_os.hypotheses.store import HypothesisKind, HypothesisRecord, HypothesisStatus, HypothesisStore


def test_bootstrap_hypotheses_have_expected_kind_counts():
    hypotheses = bootstrap_hypotheses()

    dsl = [row for row in hypotheses if row.kind == HypothesisKind.DSL]
    technical = [row for row in hypotheses if row.kind == HypothesisKind.TECHNICAL]
    ml = [row for row in hypotheses if row.kind == HypothesisKind.ML]

    assert len(dsl) == 9
    assert len(technical) == 8
    assert len(ml) == 2
    assert len(hypotheses) == 19
    assert all(row.status == HypothesisStatus.ACTIVE for row in hypotheses)
    assert all(row.stake > 0 for row in technical + ml)
    assert all(row.stake == 0.0 for row in dsl)


def test_bootstrap_hypotheses_include_prior_quality_metadata():
    hypotheses = bootstrap_hypotheses()

    assert all(row.metadata["prior_quality_source"] == "bootstrap_seed" for row in hypotheses)
    assert all(float(row.metadata["oos_sharpe"]) > 0 for row in hypotheses)
    assert all(float(row.metadata["oos_log_growth"]) > 0 for row in hypotheses)
    assert all(row.scope == {"asset": "BTC", "universe": "core_universe_1000"} for row in hypotheses)


def test_bootstrap_hypotheses_include_serious_program_seeds():
    serious = [row for row in bootstrap_hypotheses() if row.source == "bootstrap_serious"]

    assert len(serious) == 9
    assert all(row.kind == HypothesisKind.DSL for row in serious)
    assert all(row.metadata["seed_family"] == "serious" for row in serious)
    assert all(row.metadata["serious_program"] == "btc_multi_family_v2" for row in serious)
    assert all(row.stake == 0.0 for row in serious)


def test_bootstrap_hypotheses_can_build_eth_serious_seeds():
    serious = bootstrap_hypotheses("ETH")

    assert len(serious) == 6
    assert all(row.source == "bootstrap_serious" for row in serious)
    assert all(row.scope["asset"] == "ETH" for row in serious)
    assert all(row.metadata["serious_program"] == "eth_multi_family_v2" for row in serious)
    assert {row.metadata["serious_family"] for row in serious} == {
        "derivatives",
        "macro",
        "price",
    }


def test_list_observation_active_includes_zero_stake_active_hypotheses(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="h_live",
            kind=HypothesisKind.DSL,
            definition={"expression": "x"},
            status=HypothesisStatus.ACTIVE,
            stake=1.0,
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="h_observe_only",
            kind=HypothesisKind.DSL,
            definition={"expression": "y"},
            status=HypothesisStatus.ACTIVE,
            stake=0.0,
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="h_paused",
            kind=HypothesisKind.DSL,
            definition={"expression": "z"},
            status=HypothesisStatus.PAUSED,
            stake=1.0,
        )
    )

    assert [row.hypothesis_id for row in store.list_active()] == ["h_live"]
    assert [row.hypothesis_id for row in store.list_observation_active()] == [
        "h_live",
        "h_observe_only",
    ]
    store.close()
