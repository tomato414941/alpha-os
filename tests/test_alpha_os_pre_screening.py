from __future__ import annotations

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


def test_cheap_pre_screen_keeps_top_k_per_kind():
    from alpha_os.pre_screening import CheapPreScreenPolicy, cheap_pre_screen_on_feature_plane

    result = cheap_pre_screen_on_feature_plane(
        plane=_plane(),
        start_date="2026-03-24",
        end_date="2026-03-24",
        definitions=_definitions(),
        policy=CheapPreScreenPolicy(
            min_abs_corr=0.0,
            top_k_per_kind=1,
        ),
    )

    selected_ids = {item.signal_id for item in result.selected_definitions}
    assert "average_gap_3d" in selected_ids
    assert len([item for item in selected_ids if item.startswith("reversal_")]) == 1
    assert len(result.candidates) == 3


def test_cheap_pre_screen_requires_dates_in_range():
    import pytest

    from alpha_os.pre_screening import CheapPreScreenPolicy, cheap_pre_screen_on_feature_plane

    with pytest.raises(ValueError, match="no dates found in range"):
        cheap_pre_screen_on_feature_plane(
            plane=_plane(),
            start_date="2026-04-01",
            end_date="2026-04-02",
            definitions=_definitions(),
            policy=CheapPreScreenPolicy(),
        )
