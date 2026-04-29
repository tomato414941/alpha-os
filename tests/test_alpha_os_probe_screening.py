from __future__ import annotations

import warnings

import pandas as pd


def _plane():
    from alpha_os.evaluation_generation import prepare_feature_plane_from_frame

    return prepare_feature_plane_from_frame(
        frame=pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-20T00:00:00Z",
                    "2026-03-21T00:00:00Z",
                    "2026-03-22T00:00:00Z",
                    "2026-03-23T00:00:00Z",
                    "2026-03-24T00:00:00Z",
                    "2026-03-25T00:00:00Z",
                    "2026-03-26T00:00:00Z",
                    "2026-03-27T00:00:00Z",
                ],
                "value": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0, 107.0, 106.0],
            }
        )
    )


def _definitions():
    from alpha_os.signal_registry import get_signal_definition

    return [
        get_signal_definition("reversal_1d"),
        get_signal_definition("reversal_3d"),
        get_signal_definition("average_gap_3d"),
    ]


def test_probe_screen_keeps_one_candidate_per_family():
    from alpha_os.probe_screening import ProbeScreenPolicy, probe_screen_on_feature_plane

    result = probe_screen_on_feature_plane(
        plane=_plane(),
        start_date="2026-03-21",
        end_date="2026-03-24",
        definitions=_definitions(),
        policy=ProbeScreenPolicy(
            max_dates=4,
            min_sample_count=2,
            max_family_survivors_per_subject=1,
        ),
        family_ids_by_signal_id={
            "reversal_1d": "reversal_family",
            "reversal_3d": "reversal_family",
            "average_gap_3d": "average_gap_family",
        },
    )

    selected_ids = {item.signal_id for item in result.selected_definitions}
    assert "average_gap_3d" in selected_ids
    assert len([item for item in selected_ids if item.startswith("reversal_")]) == 1
    dropped = {
        item.signal_id: item.reasons
        for item in result.candidates
        if not item.keep
    }
    assert any(reasons == ("probe_family_cap",) for reasons in dropped.values())


def test_probe_screen_requires_dates_in_range():
    import pytest

    from alpha_os.probe_screening import ProbeScreenPolicy, probe_screen_on_feature_plane

    with pytest.raises(ValueError, match="no dates found in range"):
        probe_screen_on_feature_plane(
            plane=_plane(),
            start_date="2026-04-01",
            end_date="2026-04-02",
            definitions=_definitions(),
            policy=ProbeScreenPolicy(max_dates=2),
        )


def test_probe_screen_handles_constant_signal_without_runtime_warning():
    from alpha_os.evaluation_generation import prepare_feature_plane_from_frame
    from alpha_os.probe_screening import ProbeScreenPolicy, probe_screen_on_feature_plane
    from alpha_os.signal_registry import get_signal_definition

    plane = prepare_feature_plane_from_frame(
        frame=pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-20T00:00:00Z",
                    "2026-03-21T00:00:00Z",
                    "2026-03-22T00:00:00Z",
                    "2026-03-23T00:00:00Z",
                    "2026-03-24T00:00:00Z",
                    "2026-03-25T00:00:00Z",
                ],
                "value": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            }
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = probe_screen_on_feature_plane(
            plane=plane,
            start_date="2026-03-21",
            end_date="2026-03-25",
            definitions=[get_signal_definition("reversal_1d")],
            policy=ProbeScreenPolicy(
                max_dates=5,
                min_sample_count=2,
            ),
        )

    assert len(result.candidates) == 1
    assert result.candidates[0].corr == 0.0
