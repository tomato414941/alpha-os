from __future__ import annotations

from types import SimpleNamespace


def _insert_evaluation_snapshot(
    store,
    *,
    evaluation_id: str,
    signal_id: str,
    created_at: str,
) -> None:
    store.conn.execute(
        """
        INSERT INTO evaluation_snapshots (
            evaluation_id, subject_id, asset, target_id, signal_id,
            prediction_value, observation_value, signed_edge, absolute_error,
            input_source, input_range_start, input_range_end, signal_name, created_at
        )
        VALUES (?, 'subject-1', 'asset-1', 'target-1', ?, 0.1, 0.2, 0.02, 0.1,
                'test', '2026-01-01', '2026-01-02', ?, ?)
        """,
        (evaluation_id, signal_id, signal_id, created_at),
    )
    store.conn.commit()


def test_prune_screened_snapshots_removes_non_survivors_and_keeps_survivor_latest(
    tmp_path,
):
    from alpha_os.signal_discovery_application import prune_screened_snapshots
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()
    try:
        _insert_evaluation_snapshot(
            store,
            evaluation_id="survivor-old",
            signal_id="survivor",
            created_at="2026-01-01T00:00:00+00:00",
        )
        _insert_evaluation_snapshot(
            store,
            evaluation_id="survivor-new",
            signal_id="survivor",
            created_at="2026-01-02T00:00:00+00:00",
        )
        _insert_evaluation_snapshot(
            store,
            evaluation_id="rejected-1",
            signal_id="rejected",
            created_at="2026-01-02T00:00:00+00:00",
        )
        archived_count = store.archive_prepared_evaluation_snapshots(
            snapshot_set_id="run-1",
            signal_ids=["survivor"],
        )
        screening_state = SimpleNamespace(
            result=SimpleNamespace(
                survivors=(SimpleNamespace(signal_id="survivor"),),
            )
        )

        deleted_count = prune_screened_snapshots(
            store,
            selected_signal_ids=("survivor", "rejected"),
            screening_state=screening_state,
            snapshot_retention="latest_per_survivor",
        )

        remaining = store.list_evaluation_snapshots_for_signals(
            signal_ids=["survivor", "rejected"]
        )
        archived = store.list_prepared_evaluation_snapshots(
            snapshot_set_id="run-1"
        )
    finally:
        store.close()

    assert archived_count == 2
    assert deleted_count == 2
    assert [(item.evaluation_id, item.signal_id) for item in remaining] == [
        ("survivor-new", "survivor")
    ]
    assert len(archived) == 2
