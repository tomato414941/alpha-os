from alpha_os.hypotheses.bootstrap import bootstrap_hypotheses
from alpha_os.hypotheses.store import HypothesisKind, HypothesisRecord, HypothesisStatus, HypothesisStore


def test_bootstrap_hypotheses_have_expected_kind_counts():
    hypotheses = bootstrap_hypotheses()

    technical = [row for row in hypotheses if row.kind == HypothesisKind.TECHNICAL]
    ml = [row for row in hypotheses if row.kind == HypothesisKind.ML]

    assert len(technical) == 8
    assert len(ml) == 2
    assert len(hypotheses) == 10
    assert all(row.status == HypothesisStatus.ACTIVE for row in hypotheses)
    assert all(row.stake > 0 for row in hypotheses)


def test_bootstrap_hypotheses_include_prior_quality_metadata():
    hypotheses = bootstrap_hypotheses()

    assert all(row.metadata["prior_quality_source"] == "bootstrap_seed" for row in hypotheses)
    assert all(float(row.metadata["oos_sharpe"]) > 0 for row in hypotheses)
    assert all(float(row.metadata["oos_log_growth"]) > 0 for row in hypotheses)
    assert all(row.scope == {"asset": "BTC", "universe": "core_universe_1000"} for row in hypotheses)


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
