from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import DEFAULT_ASSET, DEFAULT_PRICE_SIGNAL, DEFAULT_TARGET
from .inputs import CycleInput
from .signal_client import build_signal_client


def _daily_close_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        raise ValueError("signal-noise returned no rows")
    if "timestamp" not in frame.columns:
        raise ValueError("signal-noise frame is missing timestamp")

    if "close" in frame.columns:
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


def build_cycle_input_from_frame(
    *,
    frame: pd.DataFrame,
    date: str,
    hypothesis_id: str,
    asset: str = DEFAULT_ASSET,
    target: str = DEFAULT_TARGET,
) -> CycleInput:
    daily_close = _daily_close_series(frame)
    dates = list(daily_close.index)
    if date not in daily_close.index:
        raise ValueError(f"date not found in signal history: {date}")

    idx = dates.index(date)
    if idx <= 0:
        raise ValueError(f"date {date} needs a previous close to build prediction")
    if idx >= len(dates) - 1:
        raise ValueError(f"date {date} needs a next close to build observation")

    prev_close = float(daily_close.iloc[idx - 1])
    current_close = float(daily_close.iloc[idx])
    next_close = float(daily_close.iloc[idx + 1])
    if prev_close == 0.0 or current_close == 0.0:
        raise ValueError("close price cannot be zero")

    prediction = (current_close / prev_close) - 1.0
    observation = (next_close / current_close) - 1.0
    return CycleInput(
        date=date,
        hypothesis_id=hypothesis_id,
        prediction=prediction,
        observation=observation,
        asset=asset,
        target=target,
    )


def build_cycle_inputs_from_frame(
    *,
    frame: pd.DataFrame,
    start_date: str,
    end_date: str,
    hypothesis_id: str,
    asset: str = DEFAULT_ASSET,
    target: str = DEFAULT_TARGET,
) -> list[CycleInput]:
    daily_close = _daily_close_series(frame)
    dates = list(daily_close.index)
    selected_dates = [date for date in dates if start_date <= date <= end_date]
    if not selected_dates:
        raise ValueError(f"no dates found in range: {start_date}..{end_date}")

    return [
        build_cycle_input_from_frame(
            frame=frame,
            date=date,
            hypothesis_id=hypothesis_id,
            asset=asset,
            target=target,
        )
        for date in selected_dates
    ]


def build_cycle_input_from_signal_noise(
    *,
    date: str,
    hypothesis_id: str,
    base_url: str,
    signal_name: str = DEFAULT_PRICE_SIGNAL,
) -> CycleInput:
    client = build_signal_client(base_url=base_url)
    batch = client.get_batch([signal_name], columns=["close"], resolution="1d")
    frame = batch.get(signal_name)
    if frame is None:
        raise ValueError(f"signal-noise batch did not return signal: {signal_name}")
    return build_cycle_input_from_frame(
        frame=frame,
        date=date,
        hypothesis_id=hypothesis_id,
    )


def build_cycle_inputs_from_signal_noise(
    *,
    start_date: str,
    end_date: str,
    hypothesis_id: str,
    base_url: str,
    signal_name: str = DEFAULT_PRICE_SIGNAL,
) -> list[CycleInput]:
    client = build_signal_client(base_url=base_url)
    batch = client.get_batch([signal_name], columns=["close"], resolution="1d")
    frame = batch.get(signal_name)
    if frame is None:
        raise ValueError(f"signal-noise batch did not return signal: {signal_name}")
    return build_cycle_inputs_from_frame(
        frame=frame,
        start_date=start_date,
        end_date=end_date,
        hypothesis_id=hypothesis_id,
    )


def write_cycle_input(path: str | Path, cycle_input: CycleInput) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(cycle_input)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path


def write_cycle_inputs(path: str | Path, cycle_inputs: list[CycleInput]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in cycle_inputs]
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path
