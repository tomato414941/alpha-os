from __future__ import annotations

# TODO: Split this module once the hypotheses-first runtime contract settles;
# it currently combines sync, evaluation, calibration, and prediction writes.

import hashlib
import logging
import math
from datetime import date as date_cls

import numpy as np
import requests

from ..config import Config, SIGNAL_CACHE_DB
from ..data.eval_universe import load_cached_eval_universe
from ..data.signal_client import build_signal_client_from_config, signal_noise_api_key
from ..data.store import DataStore
from ..data.universe import init_universe, price_signal
from ..dsl import parse, temporal_expression_issues
from ..dsl.evaluator import evaluate_expression
from ..predictions.store import Prediction, PredictionStore, SignalMeta
from .identity import expression_feature_names
from .store import HypothesisKind, HypothesisRecord, HypothesisStore

logger = logging.getLogger(__name__)


def produce_active_hypothesis_predictions(
    config: Config,
    *,
    today: str | None = None,
    assets: list[str] | None = None,
    hypothesis_store: HypothesisStore | None = None,
    prediction_store: PredictionStore | None = None,
) -> int:
    today = today or date_cls.today().isoformat()
    hypothesis_store = hypothesis_store or HypothesisStore()
    prediction_store = prediction_store or PredictionStore()

    active = hypothesis_store.list_observation_active()
    if not active:
        prediction_store.close()
        hypothesis_store.close()
        return 0

    client = build_signal_client_from_config(config.api)
    universe_assets = assets
    if universe_assets is None:
        if _quick_healthcheck(config.api.base_url):
            try:
                init_universe(client)
            except Exception as exc:
                logger.warning("Failed to refresh universe before prediction production: %s", exc)
        universe_assets = load_cached_eval_universe()
    if not universe_assets:
        hypothesis_store.close()
        prediction_store.close()
        return 0

    needed_features = collect_required_features(active, universe_assets)
    store = DataStore(SIGNAL_CACHE_DB, client)
    if _should_sync_required_features(store, active, needed_features):
        if _quick_healthcheck(config.api.base_url):
            try:
                store.sync(sorted(needed_features))
            except Exception as exc:
                logger.warning("API sync failed during hypothesis prediction production: %s", exc)
        else:
            logger.warning("Skipping hypothesis prediction sync: signal-noise health probe failed")
    matrix = store.get_matrix(sorted(needed_features))
    store.close()
    if matrix.empty:
        prediction_store.close()
        hypothesis_store.close()
        return 0
    data = {col: matrix[col].fillna(0).values for col in matrix.columns}

    written = write_hypothesis_predictions(
        active,
        data=data,
        assets=universe_assets,
        prediction_store=prediction_store,
        date=today,
    )

    prediction_store.close()
    hypothesis_store.close()
    return written


def write_hypothesis_predictions(
    hypotheses: list[HypothesisRecord],
    *,
    data: dict[str, np.ndarray],
    assets: list[str],
    prediction_store: PredictionStore,
    date: str,
) -> int:
    predictions: list[Prediction] = []
    for record in hypotheses:
        try:
            definition = _signal_definition(record)
            prediction_store.register_signal(
                SignalMeta(
                    signal_id=record.hypothesis_id,
                    source=record.source or record.kind,
                    definition=definition,
                    horizon=_prediction_horizon_days(record.horizon),
                    metadata={
                        "kind": record.kind,
                        "target_kind": record.target_kind,
                        "scope": record.scope,
                    },
                )
            )

            for asset in assets:
                value = compute_hypothesis_prediction(record, data=data, asset=asset)
                predictions.append(
                    Prediction(
                        signal_id=record.hypothesis_id,
                        date=date,
                        asset=asset,
                        value=value,
                        horizon=_prediction_horizon_days(record.horizon),
                    )
                )
        except Exception as exc:
            logger.warning(
                "Skipping hypothesis %s during prediction production: %s",
                record.hypothesis_id,
                exc,
            )
            continue

    return prediction_store.write(predictions) if predictions else 0


def collect_required_features(
    hypotheses: list[HypothesisRecord],
    assets: list[str],
) -> set[str]:
    needed = {_resolve_asset_series_name(asset) for asset in assets}
    for record in hypotheses:
        if record.kind == HypothesisKind.DSL:
            expression = record.definition.get("expression", "")
            needed.update(expression_feature_names(expression))
        elif record.kind == HypothesisKind.ML:
            needed.update(record.definition.get("features", []))
    return needed


def compute_hypothesis_prediction(
    record: HypothesisRecord,
    *,
    data: dict[str, np.ndarray],
    asset: str,
) -> float:
    if record.kind == HypothesisKind.DSL:
        return _compute_dsl_prediction(record, data)
    if record.kind == HypothesisKind.TECHNICAL:
        return _compute_technical_prediction(record, data, asset)
    if record.kind == HypothesisKind.ML:
        return _compute_ml_prediction(record, data, asset)
    return 0.0


def _compute_dsl_prediction(record: HypothesisRecord, data: dict[str, np.ndarray]) -> float:
    expression = record.definition.get("expression")
    if not expression or not data:
        return 0.0
    expr = parse(expression)
    issues = temporal_expression_issues(expr)
    if issues:
        raise ValueError(issues[0])
    sample = next(iter(data.values()))
    signal = evaluate_expression(expr, data, len(sample))
    if len(signal) == 0:
        return 0.0
    return _series_tail_score(signal)


def _compute_technical_prediction(
    record: HypothesisRecord,
    data: dict[str, np.ndarray],
    asset: str,
) -> float:
    series = data.get(_resolve_asset_series_name(asset))
    if series is None:
        return 0.0

    indicator = record.definition.get("indicator")
    params = record.definition.get("params", {})

    if indicator == "rsi_reversion":
        return -_center_rank(_rank(_roc_array(series, params.get("window", 14)), 20))
    if indicator == "zscore_reversion":
        return -_bounded_prediction(_zscore(series, params.get("window", 60)), scale=2.0)
    if indicator == "roc_momentum":
        return _bounded_prediction(_roc(series, params.get("window", 20)), scale=0.1)
    if indicator == "roc_reversion":
        return -_bounded_prediction(_roc(series, params.get("window", 5)), scale=0.1)
    if indicator == "macd_trend":
        return _bounded_prediction(
            _macd_signal(
                series,
                fast=params.get("fast", 12),
                slow=params.get("slow", 26),
                signal=params.get("signal", 9),
            ),
            scale=0.01,
        )
    if indicator == "bollinger_reversion":
        return -_bounded_prediction(
            _zscore(series, params.get("window", 20), std_scale=params.get("std", 2.0)),
            scale=2.0,
        )
    if indicator == "breakout":
        return _bounded_prediction(_breakout(series, params.get("window", 60)) * 2.0)
    if indicator == "low_volatility":
        return -_bounded_prediction(
            _rolling_std(_roc_array(series, 1), params.get("window", 20)),
            scale=0.05,
        )
    if indicator == "moving_average_cross":
        return _bounded_prediction(
            _moving_average_cross(
                series,
                fast=params.get("fast", 20),
                slow=params.get("slow", 60),
            ),
            scale=0.02,
        )
    if indicator == "volume_price_confirmation":
        return _bounded_prediction(_roc(series, params.get("price_window", 20)), scale=0.1)
    return 0.0


def _compute_ml_prediction(
    record: HypothesisRecord,
    data: dict[str, np.ndarray],
    asset: str,
) -> float:
    feature_names = list(record.definition.get("features", []))
    asset_feature = _resolve_asset_series_name(asset)
    if asset_feature not in feature_names:
        feature_names = [asset_feature] + feature_names

    weighted = 0.0
    total_weight = 0.0
    for feature_name in feature_names:
        series = data.get(feature_name)
        if series is None:
            continue
        feature_value = _zscore(series, 60)
        weight = _stable_weight(record.hypothesis_id, feature_name)
        weighted += weight * feature_value
        total_weight += abs(weight)
    if total_weight <= 0:
        return 0.0
    return _bounded_prediction(weighted / total_weight, scale=2.0)


def _resolve_asset_series_name(asset: str) -> str:
    try:
        return price_signal(asset)
    except KeyError:
        return asset if "_" in asset else asset.lower()


def _minimum_required_observations(hypotheses: list[HypothesisRecord]) -> int:
    required = 2
    for record in hypotheses:
        if record.kind == HypothesisKind.DSL:
            required = max(required, 60)
            continue
        params = record.definition.get("params", {})
        for value in params.values():
            if isinstance(value, int) and value > 0:
                required = max(required, value + 2)
        if record.kind == HypothesisKind.ML:
            required = max(required, 60)
    return required


def _should_sync_required_features(
    store: DataStore,
    hypotheses: list[HypothesisRecord],
    needed_features: set[str],
) -> bool:
    if not hasattr(store, "signal_row_counts"):
        return True
    row_counts = store.signal_row_counts(sorted(needed_features))
    minimum_rows = _minimum_required_observations(hypotheses)
    return any(row_counts.get(feature, 0) < minimum_rows for feature in needed_features)


def _quick_healthcheck(base_url: str, timeout: float = 3.0) -> bool:
    headers = {}
    api_key = signal_noise_api_key()
    if api_key:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-API-Key": api_key,
        }
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/health",
            timeout=timeout,
            headers=headers,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("status") in {"ok", "degraded"}
    except Exception as exc:
        logger.warning("signal-noise quick healthcheck failed: %s", exc)
        return False


def _signal_definition(record: HypothesisRecord) -> str:
    if record.kind == HypothesisKind.DSL:
        return record.definition.get("expression", "")
    if record.kind == HypothesisKind.TECHNICAL:
        return record.definition.get("indicator", "")
    if record.kind == HypothesisKind.ML:
        return record.definition.get("model_ref", record.definition.get("model_type", ""))
    return record.name


def _prediction_horizon_days(horizon: str) -> int:
    if horizon == "20D2L":
        return 20
    if horizon == "60D2L":
        return 60
    return 1


def _safe_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return value


def _zscore(arr: np.ndarray, window: int = 60, std_scale: float = 1.0) -> float:
    if len(arr) < window:
        return 0.0
    recent = arr[-window:]
    mean = float(np.nanmean(recent))
    std = float(np.nanstd(recent))
    if std < 1e-12:
        return 0.0
    return _safe_float((arr[-1] - mean) / (std * std_scale))


def _bounded_prediction(value: float, *, scale: float = 1.0) -> float:
    if not math.isfinite(value):
        return 0.0
    return _safe_float(np.tanh(value / max(scale, 1e-12)))


def _center_rank(value: float) -> float:
    return _safe_float((value - 0.5) * 2.0)


def _series_tail_score(signal: np.ndarray, window: int = 60) -> float:
    if len(signal) == 0:
        return 0.0
    recent = np.asarray(signal[-min(window, len(signal)):], dtype=np.float64)
    recent = recent[np.isfinite(recent)]
    if len(recent) == 0:
        return 0.0
    last = float(recent[-1])
    mean = float(np.nanmean(recent))
    std = float(np.nanstd(recent))
    if std < 1e-12:
        return _safe_float(np.sign(last - mean))
    return _bounded_prediction((last - mean) / std, scale=2.0)


def _rank(arr: np.ndarray, window: int = 20) -> float:
    if len(arr) < window:
        return 0.5
    recent = arr[-window:]
    val = arr[-1]
    return _safe_float(np.mean(recent <= val))


def _roc(arr: np.ndarray, period: int = 20) -> float:
    if len(arr) < period + 1:
        return 0.0
    prev = arr[-period - 1]
    if abs(prev) < 1e-12:
        return 0.0
    return _safe_float((arr[-1] - prev) / abs(prev))


def _roc_array(arr: np.ndarray, period: int = 5) -> np.ndarray:
    if len(arr) < period + 1:
        return np.array([0.0])
    result = np.zeros(len(arr))
    result[period:] = (arr[period:] - arr[:-period]) / (np.abs(arr[:-period]) + 1e-12)
    return result


def _ema(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) == 0:
        return np.array([])
    alpha = 2.0 / (window + 1.0)
    out = np.zeros(len(arr))
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _macd_signal(arr: np.ndarray, *, fast: int, slow: int, signal: int) -> float:
    if len(arr) < slow + signal:
        return 0.0
    fast_ema = _ema(arr, fast)
    slow_ema = _ema(arr, slow)
    macd = fast_ema - slow_ema
    signal_line = _ema(macd, signal)
    price = abs(arr[-1]) + 1e-12
    return _safe_float((macd[-1] - signal_line[-1]) / price)


def _breakout(arr: np.ndarray, window: int) -> float:
    if len(arr) < window:
        return 0.0
    recent = arr[-window:]
    upper = float(np.nanmax(recent))
    lower = float(np.nanmin(recent))
    if abs(upper - lower) < 1e-12:
        return 0.0
    return _safe_float((arr[-1] - lower) / (upper - lower) - 0.5)


def _rolling_std(arr: np.ndarray, window: int = 20) -> float:
    if len(arr) < window:
        return 0.0
    return _safe_float(np.nanstd(arr[-window:]))


def _moving_average_cross(arr: np.ndarray, *, fast: int, slow: int) -> float:
    if len(arr) < slow:
        return 0.0
    fast_ma = float(np.nanmean(arr[-fast:]))
    slow_ma = float(np.nanmean(arr[-slow:]))
    if abs(slow_ma) < 1e-12:
        return 0.0
    return _safe_float((fast_ma - slow_ma) / abs(slow_ma))


def _stable_weight(hypothesis_id: str, feature_name: str) -> float:
    digest = hashlib.md5(
        f"{hypothesis_id}:{feature_name}".encode(),
        usedforsecurity=False,
    ).digest()
    raw = int.from_bytes(digest[:4], "big") / 2**32
    return raw * 2.0 - 1.0
