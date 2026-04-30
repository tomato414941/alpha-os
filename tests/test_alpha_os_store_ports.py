from __future__ import annotations


def test_list_observations_for_subject_or_asset_prefers_subject_rows(tmp_path):
    from alpha_os.store import EvaluationStore

    store = EvaluationStore(tmp_path / "runtime.db")
    try:
        store.ensure_schema()
        store.finalize_observation(
            evaluation_id="asset:residual_return_3d:2026-03-24",
            subject_id="asset_subject",
            asset="BTC",
            target_id="residual_return_3d",
            observation_value=0.10,
        )
        store.finalize_observation(
            evaluation_id="subject:residual_return_3d:2026-03-25",
            subject_id="BTC_spot",
            asset="BTC",
            target_id="residual_return_3d",
            observation_value=0.20,
        )

        observations = store.list_observations_for_subject_or_asset(
            subject_id="BTC_spot",
            asset="BTC",
            target_id="residual_return_3d",
            limit=10,
        )

        assert [item.subject_id for item in observations] == ["BTC_spot"]
        assert [item.value for item in observations] == [0.20]
    finally:
        store.close()


def test_list_observations_for_subject_or_asset_falls_back_to_asset(tmp_path):
    from alpha_os.store import EvaluationStore

    store = EvaluationStore(tmp_path / "runtime.db")
    try:
        store.ensure_schema()
        for date, value in (
            ("2026-03-24", 0.10),
            ("2026-03-25", 0.20),
            ("2026-03-26", 0.30),
        ):
            store.finalize_observation(
                evaluation_id=f"BTC:residual_return_3d:{date}",
                subject_id="legacy_subject",
                asset="BTC",
                target_id="residual_return_3d",
                observation_value=value,
            )

        observations = store.list_observations_for_subject_or_asset(
            subject_id="BTC_spot",
            asset="BTC",
            target_id="residual_return_3d",
            limit=2,
        )

        assert [item.evaluation_id for item in observations] == [
            "BTC:residual_return_3d:2026-03-26",
            "BTC:residual_return_3d:2026-03-25",
        ]
        assert [item.value for item in observations] == [0.30, 0.20]
    finally:
        store.close()
