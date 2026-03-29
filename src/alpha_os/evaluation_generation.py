from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import DEFAULT_ASSET, DEFAULT_PRICE_SIGNAL, DEFAULT_TARGET
from .hypothesis_registry import HypothesisDefinition, get_hypothesis_definition
from .evaluation_inputs import EvaluationInput
from .signal_client import build_signal_client


def _load_price_frame_from_signal_noise(*, base_url: str, signal_name: str) -> pd.DataFrame:
    client = build_signal_client(base_url=base_url)
    return client.get_data(signal_name, resolution="1d")


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


def _resolve_hypothesis_definition(
    *,
    hypothesis_id: str,
    signal_name: str | None,
    definition: HypothesisDefinition | None = None,
) -> HypothesisDefinition:
    if definition is None:
        definition = get_hypothesis_definition(hypothesis_id)
    if signal_name is not None and signal_name != definition.signal_name:
        raise ValueError(
            f"signal_name override does not match hypothesis definition: "
            f"{signal_name} != {definition.signal_name}"
        )
    return definition


def _prediction_from_history(
    *,
    daily_close: pd.Series,
    daily_returns: pd.Series,
    dates: list[str],
    date: str,
    definition: HypothesisDefinition,
) -> float:
    idx = dates.index(date)
    if idx < definition.lookback:
        raise ValueError(
            f"date {date} needs {definition.lookback} prior daily returns "
            f"for {definition.hypothesis_id}"
        )

    window = daily_returns.iloc[idx - definition.lookback + 1 : idx + 1]
    if window.isna().any():
        raise ValueError(
            f"daily return window is incomplete for {definition.hypothesis_id} on {date}"
        )

    base_signal = float(window.mean())
    if definition.kind == "momentum":
        return base_signal
    if definition.kind == "reversal":
        return -base_signal
    close_window = daily_close.iloc[idx - definition.lookback + 1 : idx + 1]
    if close_window.isna().any():
        raise ValueError(
            f"daily close window is incomplete for {definition.hypothesis_id} on {date}"
        )
    current_close = float(close_window.iloc[-1])
    if definition.kind == "average_gap":
        average_close = float(close_window.mean())
        if average_close == 0.0:
            raise ValueError("average close cannot be zero")
        return (current_close / average_close) - 1.0
    if definition.kind == "range_position":
        window_min = float(close_window.min())
        window_max = float(close_window.max())
        if window_max == window_min:
            return 0.0
        return ((current_close - window_min) / (window_max - window_min)) * 2.0 - 1.0
    raise ValueError(f"unsupported hypothesis kind: {definition.kind}")


def generate_evaluation_input_from_frame(
    *,
    frame: pd.DataFrame,
    date: str,
    hypothesis_id: str,
    signal_name: str | None = None,
    definition: HypothesisDefinition | None = None,
    asset: str = DEFAULT_ASSET,
    target: str = DEFAULT_TARGET,
) -> EvaluationInput:
    definition = _resolve_hypothesis_definition(
        hypothesis_id=hypothesis_id,
        signal_name=signal_name,
        definition=definition,
    )
    daily_close = _daily_close_series(frame)
    daily_returns = daily_close.pct_change()
    dates = list(daily_close.index)
    if date not in daily_close.index:
        raise ValueError(f"date not found in signal history: {date}")

    idx = dates.index(date)
    observation_idx = idx + definition.horizon_days
    if observation_idx >= len(dates):
        raise ValueError(
            f"date {date} needs a future close {definition.horizon_days} days ahead "
            f"to build observation"
        )

    current_close = float(daily_close.iloc[idx])
    next_close = float(daily_close.iloc[observation_idx])
    if current_close == 0.0:
        raise ValueError("close price cannot be zero")

    prediction = _prediction_from_history(
        daily_close=daily_close,
        daily_returns=daily_returns,
        dates=dates,
        date=date,
        definition=definition,
    )
    observation = (next_close / current_close) - 1.0
    return EvaluationInput(
        date=date,
        hypothesis_id=hypothesis_id,
        prediction=prediction,
        observation=observation,
        asset=definition.asset if asset == DEFAULT_ASSET else asset,
        target=definition.target_id if target == DEFAULT_TARGET else target,
    )


def generate_evaluation_inputs_from_frame(
    *,
    frame: pd.DataFrame,
    start_date: str,
    end_date: str,
    hypothesis_id: str,
    signal_name: str | None = None,
    definition: HypothesisDefinition | None = None,
    asset: str = DEFAULT_ASSET,
    target: str = DEFAULT_TARGET,
) -> list[EvaluationInput]:
    daily_close = _daily_close_series(frame)
    dates = list(daily_close.index)
    selected_dates = [date for date in dates if start_date <= date <= end_date]
    if not selected_dates:
        raise ValueError(f"no dates found in range: {start_date}..{end_date}")

    return [
        generate_evaluation_input_from_frame(
            frame=frame,
            date=date,
            hypothesis_id=hypothesis_id,
            signal_name=signal_name,
            definition=definition,
            asset=asset,
            target=target,
        )
        for date in selected_dates
    ]


def generate_evaluation_input_from_signal_noise(
    *,
    date: str,
    hypothesis_id: str,
    base_url: str,
    signal_name: str | None = DEFAULT_PRICE_SIGNAL,
    definition: HypothesisDefinition | None = None,
) -> EvaluationInput:
    definition = _resolve_hypothesis_definition(
        hypothesis_id=hypothesis_id,
        signal_name=signal_name,
        definition=definition,
    )
    frame = _load_price_frame_from_signal_noise(
        base_url=base_url,
        signal_name=definition.signal_name,
    )
    return generate_evaluation_input_from_frame(
        frame=frame,
        date=date,
        hypothesis_id=hypothesis_id,
        signal_name=definition.signal_name,
        definition=definition,
    )


def generate_evaluation_inputs_from_signal_noise(
    *,
    start_date: str,
    end_date: str,
    hypothesis_id: str,
    base_url: str,
    signal_name: str | None = DEFAULT_PRICE_SIGNAL,
    definition: HypothesisDefinition | None = None,
) -> list[EvaluationInput]:
    definition = _resolve_hypothesis_definition(
        hypothesis_id=hypothesis_id,
        signal_name=signal_name,
        definition=definition,
    )
    frame = _load_price_frame_from_signal_noise(
        base_url=base_url,
        signal_name=definition.signal_name,
    )
    return generate_evaluation_inputs_from_frame(
        frame=frame,
        start_date=start_date,
        end_date=end_date,
        hypothesis_id=hypothesis_id,
        signal_name=definition.signal_name,
        definition=definition,
    )


def write_evaluation_input(path: str | Path, evaluation_input: EvaluationInput) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(evaluation_input)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path


def write_evaluation_inputs(path: str | Path, evaluation_inputs: list[EvaluationInput]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in evaluation_inputs]
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path
