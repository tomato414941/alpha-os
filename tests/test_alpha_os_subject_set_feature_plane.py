from __future__ import annotations

import pandas as pd


def test_build_subject_set_feature_planes_injects_cross_sectional_signals(monkeypatch):
    from alpha_os.signal_registry import get_signal_definition
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
    )
    from alpha_os.subject_set_feature_plane import (
        SubjectPlaneKey,
        build_subject_set_feature_planes,
    )

    subject_set = SubjectSet(
        observation_specs=(
            ObservationSpec(
                observation_spec_id="aaa_close",
                observable_id="daily_close",
                adapter_kind="signal_noise_asset_observable",
            ),
            ObservationSpec(
                observation_spec_id="bbb_close",
                observable_id="daily_close",
                adapter_kind="signal_noise_asset_observable",
            ),
        ),
        bindings=(
            SubjectObservationBinding(
                subject_id="AAA_equity",
                subject_kind="equity",
                asset="AAA",
                observation_spec_id="aaa_close",
            ),
            SubjectObservationBinding(
                subject_id="BBB_equity",
                subject_kind="equity",
                asset="BBB",
                observation_spec_id="bbb_close",
            ),
        ),
    )

    dates = pd.date_range("2026-03-01", periods=30, freq="D", tz="UTC")
    aaa_frame = pd.DataFrame(
        {
            "timestamp": [timestamp.isoformat() for timestamp in dates],
            "close": [100.0 + idx * 1.5 for idx in range(len(dates))],
        }
    )
    bbb_frame = pd.DataFrame(
        {
            "timestamp": [timestamp.isoformat() for timestamp in dates],
            "close": [100.0 - idx * 0.7 for idx in range(len(dates))],
        }
    )

    def fake_load_observation_frame(observation_spec, *, asset, base_url, client=None):
        assert base_url == "https://example.test"
        if asset == "AAA":
            return aaa_frame
        if asset == "BBB":
            return bbb_frame
        raise AssertionError(asset)

    monkeypatch.setattr(
        "alpha_os.data_repositories.load_observation_frame",
        fake_load_observation_frame,
    )

    planes = build_subject_set_feature_planes(
        subject_set=subject_set,
        executable_definitions=[
            get_signal_definition("relative_strength_rank_20d"),
            get_signal_definition("peer_mean_reversion_20d"),
        ],
        base_url="https://example.test",
    )

    aaa_plane = planes[SubjectPlaneKey(asset="AAA", observation_spec_id="aaa_close")]
    bbb_plane = planes[SubjectPlaneKey(asset="BBB", observation_spec_id="bbb_close")]
    aaa_rank = aaa_plane.signal_series(kind="relative_strength_rank", lookback=20)
    bbb_rank = bbb_plane.signal_series(kind="relative_strength_rank", lookback=20)
    aaa_peer_reversion = aaa_plane.signal_series(kind="peer_mean_reversion", lookback=20)

    assert aaa_rank.loc["2026-03-25"] > 0.0
    assert bbb_rank.loc["2026-03-25"] < 0.0
    assert aaa_peer_reversion.loc["2026-03-25"] == -aaa_rank.loc["2026-03-25"]


def test_build_subject_set_feature_planes_reuses_cached_frames(tmp_path, monkeypatch):
    from alpha_os.data_repositories import (
        FeaturePlaneRepository,
        ObservationFrameRepository,
    )
    from alpha_os.signal_registry import get_signal_definition
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
    )
    from alpha_os.store import EvaluationStore
    from alpha_os.subject_set_feature_plane import build_subject_set_feature_planes

    store = EvaluationStore(str(tmp_path / "runtime.db"))
    store.ensure_schema()
    repository = FeaturePlaneRepository(
        observation_repository=ObservationFrameRepository(store=store)
    )
    subject_set = SubjectSet(
        observation_specs=(
            ObservationSpec(
                observation_spec_id="aaa_close",
                observable_id="daily_close",
                adapter_kind="signal_noise_asset_observable",
            ),
        ),
        bindings=(
            SubjectObservationBinding(
                subject_id="AAA_equity",
                subject_kind="equity",
                asset="AAA",
                observation_spec_id="aaa_close",
            ),
        ),
    )
    dates = pd.date_range("2026-03-01", periods=40, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": [timestamp.isoformat() for timestamp in dates],
            "close": [100.0 + idx for idx in range(len(dates))],
        }
    )
    call_count = {"value": 0}

    def fake_load_observation_frame(observation_spec, *, asset, base_url, client=None):
        call_count["value"] += 1
        assert asset == "AAA"
        return frame

    monkeypatch.setattr(
        "alpha_os.data_repositories.load_observation_frame",
        fake_load_observation_frame,
    )

    build_subject_set_feature_planes(
        subject_set=subject_set,
        executable_definitions=[get_signal_definition("momentum_3d")],
        base_url="https://example.test",
        feature_plane_repository=repository,
    )
    persisted_repository = FeaturePlaneRepository(
        observation_repository=ObservationFrameRepository(store=store)
    )
    build_subject_set_feature_planes(
        subject_set=subject_set,
        executable_definitions=[get_signal_definition("momentum_3d")],
        base_url="https://example.test",
        feature_plane_repository=persisted_repository,
    )

    assert call_count["value"] == 1
