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


def test_signal_discovery_run_builder_preserves_backfill_counts():
    from alpha_os.signal_discovery_persistence_builders import build_signal_discovery_run
    from alpha_os.subject_set_backfill_service import SubjectSetBackfillResult

    backfill_result = SubjectSetBackfillResult(
        latest_snapshot=None,
        created_count=0,
        existing_count=0,
        total_executables=10,
        pre_screen_selected_executables=8,
        probe_selected_executables=5,
        survivor_selected_executables=3,
        selected_executables=3,
        selected_signal_ids=("signal-1", "signal-2", "signal-3"),
        evaluation_inputs=30,
    )

    run = build_signal_discovery_run(
        signal_discovery_run_id="run-1",
        snapshot_set_id="snapshot-set-1",
        signal_discovery_id="discovery-1",
        subject_set_id="subject-set-1",
        target_id="target-1",
        start_date="2026-01-01",
        end_date="2026-01-31",
        screening_result_id="screening-1",
        compressed_belief_id="belief-1",
        workflow_runtime_s=1.25,
        backfill_result=backfill_result,
        pruned_snapshot_count=4,
        created_at="2026-02-01T00:00:00+00:00",
    )

    assert run.total_executables == 10
    assert run.pre_screen_selected == 8
    assert run.probe_selected == 5
    assert run.survivor_selected == 3
    assert run.persisted_signals == 3
    assert run.evaluation_inputs == 30
    assert run.pruned_snapshots == 4
