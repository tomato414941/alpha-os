from __future__ import annotations

import pandas as pd

from .feature_plane import PriceFeaturePlane


def _daily_close_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        raise ValueError("signal-noise returned no rows")
    if "timestamp" not in frame.columns:
        raise ValueError("signal-noise frame is missing timestamp")

    if "research_close" in frame.columns:
        value_column = "research_close"
    elif "close" in frame.columns:
        value_column = "close"
    elif "value" in frame.columns:
        value_column = "value"
    else:
        raise ValueError("signal-noise frame is missing close/value column")

    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["timestamp", value_column]).sort_values("timestamp")
    if normalized.empty:
        raise ValueError("signal-noise frame has no valid timestamp/value rows")

    normalized["date"] = normalized["timestamp"].dt.strftime("%Y-%m-%d")
    by_day = normalized.groupby("date", sort=True)[value_column].last()
    return by_day.astype(float)


def _numeric_frame_series_by_column(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if frame.empty or "timestamp" not in frame.columns or column not in frame.columns:
        return None
    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["timestamp", column]).sort_values("timestamp")
    if normalized.empty:
        return None
    normalized["date"] = normalized["timestamp"].dt.strftime("%Y-%m-%d")
    by_day = normalized.groupby("date", sort=True)[column].last()
    return by_day.astype(float)


def _daily_volume_series(frame: pd.DataFrame) -> pd.Series | None:
    return _numeric_frame_series_by_column(frame, "volume")


def _metadata_frame_series_by_column(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if frame.empty or "timestamp" not in frame.columns or column not in frame.columns:
        return None
    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["timestamp", column]).sort_values("timestamp")
    if normalized.empty:
        return None
    normalized["date"] = normalized["timestamp"].dt.strftime("%Y-%m-%d")
    by_day = normalized.groupby("date", sort=True)[column].last()
    return by_day.astype(str)


def _first_present_series(*series_candidates: pd.Series | None) -> pd.Series | None:
    for series in series_candidates:
        if series is not None:
            return series
    return None


def prepare_feature_plane_from_frame(
    *,
    frame: pd.DataFrame,
) -> PriceFeaturePlane:
    daily_close = _daily_close_series(frame)
    daily_volume = _daily_volume_series(frame)
    extra_observables = {
        observable_id: series
        for observable_id, series in (
            ("front_price", _numeric_frame_series_by_column(frame, "front_price")),
            ("next_price", _numeric_frame_series_by_column(frame, "next_price")),
            ("funding_rate", _numeric_frame_series_by_column(frame, "funding_rate")),
            ("open_interest", _numeric_frame_series_by_column(frame, "open_interest")),
            ("borrow_fee", _numeric_frame_series_by_column(frame, "borrow_fee")),
            ("roll_cost_bps", _numeric_frame_series_by_column(frame, "roll_cost_bps")),
            ("financing_cost_bps", _numeric_frame_series_by_column(frame, "financing_cost_bps")),
            ("valuation_ratio", _numeric_frame_series_by_column(frame, "valuation_ratio")),
            ("earnings_revision", _numeric_frame_series_by_column(frame, "earnings_revision")),
            ("basis", _numeric_frame_series_by_column(frame, "basis")),
            (
                "tradable_price",
                _first_present_series(
                    _numeric_frame_series_by_column(frame, "tradable_price"),
                    _numeric_frame_series_by_column(frame, "front_price"),
                ),
            ),
        )
        if series is not None
    }
    metadata_observables = {
        observable_id: series
        for observable_id, series in (
            ("contract_id", _metadata_frame_series_by_column(frame, "contract_id")),
            ("next_contract_id", _metadata_frame_series_by_column(frame, "next_contract_id")),
            ("expiry", _metadata_frame_series_by_column(frame, "expiry")),
            ("contract_family", _metadata_frame_series_by_column(frame, "contract_family")),
            ("quote_ccy", _metadata_frame_series_by_column(frame, "quote_ccy")),
            ("collateral_ccy", _metadata_frame_series_by_column(frame, "collateral_ccy")),
        )
        if series is not None
    }
    return PriceFeaturePlane.from_daily_close(
        daily_close,
        daily_volume=daily_volume,
        extra_observables=extra_observables,
        metadata_observables=metadata_observables,
    )
