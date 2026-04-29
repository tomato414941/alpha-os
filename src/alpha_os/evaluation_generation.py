from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import DEFAULT_TARGET
from .feature_plane import PriceFeaturePlane
from .signal_compiler import (
    CompiledSignalFamily,
    compile_signal_families,
)
from .signal_registry import (
    SignalDefinition,
    asset_observable_observation_spec,
    get_signal_definition,
    subject_id_for_signal,
)
from .evaluation_inputs import SubjectEvaluationInput
from .observation_adapters import load_observation_frame
from .instrument_lifecycle import resolve_roll_resolution
from .portfolio_decision import ObservationSpec


def _optional_feature_plane_scalar(
    plane: PriceFeaturePlane,
    *,
    date: str,
    observable_id: str,
) -> float | None:
    try:
        series = plane.observable_series(observable_id=observable_id)
    except ValueError:
        return None
    value = series.iloc[plane.date_to_index[date]]
    if pd.isna(value):
        return None
    return float(value)


def _optional_metadata_value(
    plane: PriceFeaturePlane,
    *,
    date: str,
    observable_id: str,
) -> str | None:
    series = plane.metadata_series(observable_id=observable_id)
    if series is None or date not in series.index:
        return None
    value = series.loc[date]
    if pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def _research_price_observable_id(observation_spec: ObservationSpec | None) -> str:
    if observation_spec is None or observation_spec.research_price_observable_id is None:
        return "daily_close"
    return str(observation_spec.research_price_observable_id)


def _tradable_price_observable_id(observation_spec: ObservationSpec | None) -> str:
    if observation_spec is None or observation_spec.tradable_price_observable_id is None:
        return "tradable_price"
    return str(observation_spec.tradable_price_observable_id)


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


def _evaluation_input_lifecycle_from_plane(
    plane: PriceFeaturePlane,
    *,
    date: str,
    contract_family: str | None = None,
    quote_ccy: str | None = None,
    collateral_ccy: str | None = None,
    roll_rule: str | None = None,
    contract_multiplier: float | None = None,
) -> dict[str, object]:
    lifecycle = resolve_roll_resolution(
        plane,
        date=date,
        roll_rule=roll_rule,
        contract_family=(
            contract_family
            if contract_family is not None
            else _optional_metadata_value(
                plane,
                date=date,
                observable_id="contract_family",
            )
        ),
        quote_ccy=(
            quote_ccy
            if quote_ccy is not None
            else _optional_metadata_value(
                plane,
                date=date,
                observable_id="quote_ccy",
            )
        ),
        collateral_ccy=(
            collateral_ccy
            if collateral_ccy is not None
            else _optional_metadata_value(
                plane,
                date=date,
                observable_id="collateral_ccy",
            )
        ),
    )
    funding_rate = _optional_feature_plane_scalar(
        plane,
        date=date,
        observable_id="funding_rate",
    )
    borrow_fee = _optional_feature_plane_scalar(
        plane,
        date=date,
        observable_id="borrow_fee",
    )
    roll_cost_bps = _optional_feature_plane_scalar(
        plane,
        date=date,
        observable_id="roll_cost_bps",
    )
    financing_cost = _optional_feature_plane_scalar(
        plane,
        date=date,
        observable_id="financing_cost_bps",
    )
    return {
        "funding_cost_bps": (
            None if funding_rate is None else float(funding_rate * 10000.0)
        ),
        "borrow_fee_bps": (
            None if borrow_fee is None else float(borrow_fee * 10000.0)
        ),
        "roll_cost_bps": (
            None if roll_cost_bps is None else float(roll_cost_bps)
        ),
        "financing_cost_bps": (
            None if financing_cost is None else float(financing_cost * 10000.0)
        ),
        "contract_multiplier": (
            None if contract_multiplier is None else float(contract_multiplier)
        ),
        "contract_id": lifecycle.contract_id,
        "contract_family": lifecycle.contract_family,
        "quote_ccy": lifecycle.quote_ccy,
        "collateral_ccy": lifecycle.collateral_ccy,
        "roll_event": lifecycle.to_document() if lifecycle.rolled else None,
    }

def _load_price_frame_from_signal_noise(
    *,
    base_url: str,
    asset: str | None = None,
    observation_spec: ObservationSpec | None = None,
    signal_name: str | None = None,
    frame_repository: object | None = None,
) -> pd.DataFrame:
    if observation_spec is not None and asset is not None:
        if frame_repository is not None:
            return frame_repository.load_signal_noise_frame(
                observation_spec=observation_spec,
                asset=asset,
                base_url=base_url,
            )
        return load_observation_frame(
            observation_spec,
            asset=asset,
            base_url=base_url,
        )
    raise ValueError("observation_spec and asset are required for signal-noise observation loads")


def _resolve_observation_spec(
    *,
    definition: SignalDefinition,
    observation_spec: ObservationSpec | None = None,
) -> ObservationSpec:
    if observation_spec is not None:
        return observation_spec
    if definition.observation_spec is None:
        return asset_observable_observation_spec(
            observation_spec_id=f"{definition.signal_id}__default"
        )
    return definition.observation_spec


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


def _resolve_signal_definition(
    *,
    signal_id: str,
    definition: SignalDefinition | None = None,
) -> SignalDefinition:
    if definition is None:
        definition = get_signal_definition(signal_id)
    return definition


def _prediction_from_history(
    *,
    daily_close: pd.Series,
    daily_returns: pd.Series,
    dates: list[str],
    date: str,
    definition: SignalDefinition,
) -> float:
    idx = dates.index(date)
    if idx < definition.lookback:
        raise ValueError(
            f"date {date} needs {definition.lookback} prior daily returns "
            f"for {definition.signal_id}"
        )

    window = daily_returns.iloc[idx - definition.lookback + 1 : idx + 1]
    if window.isna().any():
        raise ValueError(
            f"daily return window is incomplete for {definition.signal_id} on {date}"
        )

    base_signal = float(window.mean())
    if definition.kind == "momentum":
        return base_signal
    if definition.kind == "reversal":
        return -base_signal
    if definition.kind in {
        "vol_compression_breakout",
        "vol_expansion_reversal",
        "momentum_low_vol",
        "reversal_after_shock",
        "trend_volume_confirmation",
    }:
        plane = PriceFeaturePlane.from_daily_close(
            daily_close,
            daily_volume=None,
        )
        return _prediction_from_feature_plane(
            plane=plane,
            date=date,
            definition=definition,
        )
    close_window = daily_close.iloc[idx - definition.lookback + 1 : idx + 1]
    if close_window.isna().any():
        raise ValueError(
            f"daily close window is incomplete for {definition.signal_id} on {date}"
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
    raise ValueError(f"unsupported signal kind: {definition.kind}")


def _prediction_from_feature_plane(
    *,
    plane: PriceFeaturePlane,
    date: str,
    definition: SignalDefinition,
) -> float:
    if date not in plane.date_to_index:
        raise ValueError(f"date not found in signal history: {date}")
    signal = plane.signal_series(kind=definition.kind, lookback=definition.lookback)
    value = signal.iloc[plane.date_to_index[date]]
    if pd.isna(value):
        raise ValueError(
            f"signal window is incomplete for {definition.signal_id} on {date}"
        )
    return float(value)


def generate_evaluation_input_from_frame(
    *,
    frame: pd.DataFrame,
    date: str,
    signal_id: str,
    signal_name: str | None = None,
    definition: SignalDefinition | None = None,
    target_id: str = DEFAULT_TARGET,
    subject_id: str | None = None,
    observation_spec: ObservationSpec | None = None,
    contract_multiplier: float | None = None,
    contract_family: str | None = None,
    quote_ccy: str | None = None,
    collateral_ccy: str | None = None,
    roll_rule: str | None = None,
) -> SubjectEvaluationInput:
    definition = _resolve_signal_definition(
        signal_id=signal_id,
        definition=definition,
    )
    plane = prepare_feature_plane_from_frame(frame=frame)
    return generate_evaluation_input_from_feature_plane(
        plane=plane,
        date=date,
        signal_id=signal_id,
        definition=definition,
        target_id=target_id,
        subject_id=subject_id,
        observation_spec=observation_spec,
        contract_multiplier=contract_multiplier,
        contract_family=contract_family,
        quote_ccy=quote_ccy,
        collateral_ccy=collateral_ccy,
        roll_rule=roll_rule,
    )


def generate_evaluation_input_from_feature_plane(
    *,
    plane: PriceFeaturePlane,
    date: str,
    signal_id: str,
    definition: SignalDefinition | None = None,
    target_id: str = DEFAULT_TARGET,
    subject_id: str | None = None,
    observation_spec: ObservationSpec | None = None,
    contract_multiplier: float | None = None,
    contract_family: str | None = None,
    quote_ccy: str | None = None,
    collateral_ccy: str | None = None,
    roll_rule: str | None = None,
) -> SubjectEvaluationInput:
    definition = _resolve_signal_definition(
        signal_id=signal_id,
        definition=definition,
    )
    if date not in plane.date_to_index:
        raise ValueError(f"date not found in signal history: {date}")

    prediction = _prediction_from_feature_plane(
        plane=plane,
        date=date,
        definition=definition,
    )
    observation_series = plane.observation_series(
        horizon_days=definition.horizon_days,
        observable_id=_tradable_price_observable_id(observation_spec),
    )
    observation = observation_series.iloc[plane.date_to_index[date]]
    if pd.isna(observation):
        raise ValueError(
            f"date {date} needs a future close {definition.horizon_days} days ahead "
            f"to build observation"
        )
    lifecycle_artifacts = _evaluation_input_lifecycle_from_plane(
        plane,
        date=date,
        contract_family=contract_family,
        quote_ccy=quote_ccy,
        collateral_ccy=collateral_ccy,
        roll_rule=roll_rule,
        contract_multiplier=contract_multiplier,
    )
    return SubjectEvaluationInput(
        date=date,
        signal_id=signal_id,
        prediction=prediction,
        observation=float(observation),
        subject_id=subject_id_for_signal(
            signal_id=signal_id,
            asset=definition.asset if subject_id is None else str(subject_id),
        )
        if subject_id is None
        else str(subject_id),
        target_id=definition.target_id if target_id == DEFAULT_TARGET else target_id,
        funding_cost_bps=lifecycle_artifacts["funding_cost_bps"],
        borrow_fee_bps=lifecycle_artifacts["borrow_fee_bps"],
        roll_cost_bps=lifecycle_artifacts["roll_cost_bps"],
        financing_cost_bps=lifecycle_artifacts["financing_cost_bps"],
        contract_multiplier=lifecycle_artifacts["contract_multiplier"],
        contract_id=lifecycle_artifacts["contract_id"],
        contract_family=lifecycle_artifacts["contract_family"],
        quote_ccy=lifecycle_artifacts["quote_ccy"],
        collateral_ccy=lifecycle_artifacts["collateral_ccy"],
        roll_event=lifecycle_artifacts["roll_event"],
    )


def generate_evaluation_inputs_from_frame(
    *,
    frame: pd.DataFrame,
    start_date: str,
    end_date: str,
    signal_id: str,
    signal_name: str | None = None,
    definition: SignalDefinition | None = None,
    target_id: str = DEFAULT_TARGET,
    subject_id: str | None = None,
    observation_spec: ObservationSpec | None = None,
    contract_multiplier: float | None = None,
    contract_family: str | None = None,
    quote_ccy: str | None = None,
    collateral_ccy: str | None = None,
    roll_rule: str | None = None,
) -> list[SubjectEvaluationInput]:
    plane = prepare_feature_plane_from_frame(frame=frame)
    return generate_evaluation_inputs_from_feature_plane(
        plane=plane,
        start_date=start_date,
        end_date=end_date,
        signal_id=signal_id,
        definition=definition,
        target_id=target_id,
        subject_id=subject_id,
        observation_spec=observation_spec,
        contract_multiplier=contract_multiplier,
        contract_family=contract_family,
        quote_ccy=quote_ccy,
        collateral_ccy=collateral_ccy,
        roll_rule=roll_rule,
    )


def generate_evaluation_inputs_from_feature_plane(
    *,
    plane: PriceFeaturePlane,
    start_date: str,
    end_date: str,
    signal_id: str,
    definition: SignalDefinition | None = None,
    target_id: str = DEFAULT_TARGET,
    subject_id: str | None = None,
    observation_spec: ObservationSpec | None = None,
    contract_multiplier: float | None = None,
    contract_family: str | None = None,
    quote_ccy: str | None = None,
    collateral_ccy: str | None = None,
    roll_rule: str | None = None,
) -> list[SubjectEvaluationInput]:
    selected_dates = [
        date for date in plane.dates if start_date <= date <= end_date
    ]
    if not selected_dates:
        raise ValueError(f"no dates found in range: {start_date}..{end_date}")

    return [
        generate_evaluation_input_from_feature_plane(
            plane=plane,
            date=date,
            signal_id=signal_id,
            definition=definition,
            target_id=target_id,
            subject_id=subject_id,
            observation_spec=observation_spec,
            contract_multiplier=contract_multiplier,
            contract_family=contract_family,
            quote_ccy=quote_ccy,
            collateral_ccy=collateral_ccy,
            roll_rule=roll_rule,
        )
        for date in selected_dates
    ]


def generate_evaluation_inputs_batch_from_feature_plane(
    *,
    plane: PriceFeaturePlane,
    start_date: str,
    end_date: str,
    definitions: list[SignalDefinition],
    observation_spec: ObservationSpec | None = None,
    contract_multiplier: float | None = None,
    contract_family: str | None = None,
    quote_ccy: str | None = None,
    collateral_ccy: str | None = None,
    roll_rule: str | None = None,
) -> list[SubjectEvaluationInput]:
    compiled_families = compile_signal_families(definitions)
    return generate_evaluation_inputs_from_compiled_families(
        plane=plane,
        start_date=start_date,
        end_date=end_date,
        compiled_families=compiled_families,
        observation_spec=observation_spec,
        contract_multiplier=contract_multiplier,
        contract_family=contract_family,
        quote_ccy=quote_ccy,
        collateral_ccy=collateral_ccy,
        roll_rule=roll_rule,
    )


def generate_evaluation_inputs_from_compiled_families(
    *,
    plane: PriceFeaturePlane,
    start_date: str,
    end_date: str,
    compiled_families: tuple[CompiledSignalFamily, ...],
    observation_spec: ObservationSpec | None = None,
    contract_multiplier: float | None = None,
    contract_family: str | None = None,
    quote_ccy: str | None = None,
    collateral_ccy: str | None = None,
    roll_rule: str | None = None,
) -> list[SubjectEvaluationInput]:
    selected_dates = [
        date for date in plane.dates if start_date <= date <= end_date
    ]
    if not selected_dates:
        raise ValueError(f"no dates found in range: {start_date}..{end_date}")

    evaluation_inputs: list[SubjectEvaluationInput] = []
    for family in compiled_families:
        signal_frame = plane.signal_frame(
            kind=family.kind,
            lookbacks=family.lookbacks,
        ).loc[selected_dates]
        observation_slice = plane.observation_series(
            horizon_days=family.horizon_days,
            observable_id=_tradable_price_observable_id(observation_spec),
        ).loc[selected_dates]
        invalid_observation_dates = observation_slice[observation_slice.isna()].index.tolist()
        if invalid_observation_dates:
            first_invalid = str(invalid_observation_dates[0])
            raise ValueError(
                f"date {first_invalid} needs a future close {family.horizon_days} days ahead "
                f"to build observation"
            )

        for lookback, definitions_at_lookback in family.definitions_by_lookback.items():
            signal_slice = signal_frame[lookback]
            invalid_signal_dates = signal_slice[signal_slice.isna()].index.tolist()
            if invalid_signal_dates:
                first_invalid = str(invalid_signal_dates[0])
                raise ValueError(
                    "signal window is incomplete for "
                    f"{definitions_at_lookback[0].signal_id} on {first_invalid}"
                )
            prediction_values = signal_slice.tolist()
            observation_values = observation_slice.tolist()
            for definition in definitions_at_lookback:
                subject_id = subject_id_for_signal(
                    signal_id=definition.signal_id,
                    asset=definition.asset,
                )
                for date, prediction_value, observation_value in zip(
                    selected_dates,
                    prediction_values,
                    observation_values,
                    strict=True,
                ):
                    lifecycle_artifacts = _evaluation_input_lifecycle_from_plane(
                        plane,
                        date=str(date),
                        contract_family=contract_family,
                        quote_ccy=quote_ccy,
                        collateral_ccy=collateral_ccy,
                        roll_rule=roll_rule,
                        contract_multiplier=contract_multiplier,
                    )
                    evaluation_inputs.append(
                        SubjectEvaluationInput(
                            date=str(date),
                            signal_id=definition.signal_id,
                            prediction=float(prediction_value),
                            observation=float(observation_value),
                            subject_id=subject_id,
                            target_id=definition.target_id,
                            funding_cost_bps=lifecycle_artifacts["funding_cost_bps"],
                            borrow_fee_bps=lifecycle_artifacts["borrow_fee_bps"],
                            roll_cost_bps=lifecycle_artifacts["roll_cost_bps"],
                            financing_cost_bps=lifecycle_artifacts["financing_cost_bps"],
                            contract_multiplier=lifecycle_artifacts["contract_multiplier"],
                            contract_id=lifecycle_artifacts["contract_id"],
                            contract_family=lifecycle_artifacts["contract_family"],
                            quote_ccy=lifecycle_artifacts["quote_ccy"],
                            collateral_ccy=lifecycle_artifacts["collateral_ccy"],
                            roll_event=lifecycle_artifacts["roll_event"],
                        )
                    )
    return evaluation_inputs
def generate_evaluation_input_from_signal_noise(
    *,
    date: str,
    signal_id: str,
    base_url: str,
    definition: SignalDefinition | None = None,
    observation_spec: ObservationSpec | None = None,
) -> SubjectEvaluationInput:
    definition = _resolve_signal_definition(
        signal_id=signal_id,
        definition=definition,
    )
    resolved_observation_spec = _resolve_observation_spec(
        definition=definition,
        observation_spec=observation_spec,
    )
    frame = _load_price_frame_from_signal_noise(
        base_url=base_url,
        asset=definition.asset,
        observation_spec=resolved_observation_spec,
    )
    return generate_evaluation_input_from_frame(
        frame=frame,
        date=date,
        signal_id=signal_id,
        definition=definition,
    )


def generate_evaluation_inputs_from_signal_noise(
    *,
    start_date: str,
    end_date: str,
    signal_id: str,
    base_url: str,
    definition: SignalDefinition | None = None,
    observation_spec: ObservationSpec | None = None,
) -> list[SubjectEvaluationInput]:
    definition = _resolve_signal_definition(
        signal_id=signal_id,
        definition=definition,
    )
    resolved_observation_spec = _resolve_observation_spec(
        definition=definition,
        observation_spec=observation_spec,
    )
    frame = _load_price_frame_from_signal_noise(
        base_url=base_url,
        asset=definition.asset,
        observation_spec=resolved_observation_spec,
    )
    return generate_evaluation_inputs_from_frame(
        frame=frame,
        start_date=start_date,
        end_date=end_date,
        signal_id=signal_id,
        definition=definition,
    )


def write_evaluation_input(path: str | Path, evaluation_input: SubjectEvaluationInput) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(evaluation_input)
    payload["signal_id"] = evaluation_input.signal_id
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path


def write_evaluation_inputs(path: str | Path, evaluation_inputs: list[SubjectEvaluationInput]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for item in evaluation_inputs:
        document = asdict(item)
        document["signal_id"] = item.signal_id
        payload.append(document)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path
