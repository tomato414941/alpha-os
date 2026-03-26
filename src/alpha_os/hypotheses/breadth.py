from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np

from ..dsl import parse, temporal_expression_issues
from ..dsl.evaluator import evaluate_expression, normalize_signal
from ..data.store import DataStore
from ..data.universe import price_signal
from .lifecycle import AllocationRebalanceEntry
from .producer import collect_required_features
from .store import HypothesisKind, HypothesisRecord


@dataclass(frozen=True)
class BreadthPair:
    left_id: str
    right_id: str
    correlation: float
    abs_correlation: float


@dataclass(frozen=True)
class BreadthReport:
    n_records: int
    n_analyzed: int
    n_skipped: int
    lookback: int
    mean_abs_correlation: float
    effective_breadth: float
    top_pairs: list[BreadthPair]
    skipped_ids: list[str]


def analyze_capital_breadth(
    records: list[HypothesisRecord],
    *,
    data: dict[str, np.ndarray],
    asset: str,
    lookback: int = 252,
    top_pairs: int = 5,
) -> BreadthReport:
    analyzed_ids: list[str] = []
    skipped_ids: list[str] = []
    rows: list[np.ndarray] = []

    for record in records:
        series = hypothesis_signal_series(record, data=data, asset=asset)
        if series is None:
            skipped_ids.append(record.hypothesis_id)
            continue
        arr = np.asarray(series, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = arr[-lookback:]
        if arr.size < 10 or float(np.nanstd(arr)) <= 1e-12:
            skipped_ids.append(record.hypothesis_id)
            continue
        analyzed_ids.append(record.hypothesis_id)
        rows.append(arr)

    if not rows:
        return BreadthReport(
            n_records=len(records),
            n_analyzed=0,
            n_skipped=len(skipped_ids),
            lookback=lookback,
            mean_abs_correlation=0.0,
            effective_breadth=0.0,
            top_pairs=[],
            skipped_ids=skipped_ids,
        )

    signals = np.vstack(rows)
    corr = np.corrcoef(signals)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 1.0)

    if abs_corr.shape[0] <= 1:
        mean_abs_correlation = 0.0
    else:
        mask = ~np.eye(abs_corr.shape[0], dtype=bool)
        mean_abs_correlation = float(abs_corr[mask].mean())

    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.clip(np.real(eigvals), 0.0, None)
    denom = float(np.square(eigvals).sum())
    effective_breadth = float(np.square(eigvals.sum()) / denom) if denom > 1e-12 else 0.0

    pairs: list[BreadthPair] = []
    for i in range(len(analyzed_ids)):
        for j in range(i + 1, len(analyzed_ids)):
            pairs.append(
                BreadthPair(
                    left_id=analyzed_ids[i],
                    right_id=analyzed_ids[j],
                    correlation=float(corr[i, j]),
                    abs_correlation=float(abs_corr[i, j]),
                )
            )
    pairs.sort(key=lambda pair: pair.abs_correlation, reverse=True)

    return BreadthReport(
        n_records=len(records),
        n_analyzed=len(analyzed_ids),
        n_skipped=len(skipped_ids),
        lookback=lookback,
        mean_abs_correlation=mean_abs_correlation,
        effective_breadth=effective_breadth,
        top_pairs=pairs[:top_pairs],
        skipped_ids=skipped_ids,
    )


def load_bootstrap_capital_backed_records(
    store,
    *,
    floor: float = 0.0,
    asset: str | None = None,
) -> list[HypothesisRecord]:
    records = []
    for record in store.list_live(asset=asset):
        try:
            bootstrap_value = float(record.metadata.get("lifecycle_bootstrap_trust", 0.0))
        except (TypeError, ValueError):
            bootstrap_value = 0.0
        if bootstrap_value > floor:
            records.append(record)
    return records


def load_capital_backed_records(
    store,
    *,
    floor: float = 0.0,
    asset: str | None = None,
) -> list[HypothesisRecord]:
    return store.list_capital_backed(floor=floor, asset=asset)


def load_breadth_matrix(
    store: DataStore,
    records: list[HypothesisRecord],
    *,
    asset: str,
    lookback: int,
) -> dict[str, np.ndarray]:
    if not records:
        return {}
    features = sorted(collect_required_features(records, [asset]))
    matrix = store.get_matrix(features)
    if matrix.empty:
        return {}
    matrix = matrix.tail(lookback + 252)
    return {column: matrix[column].fillna(0.0).to_numpy(dtype=np.float64) for column in matrix.columns}


def apply_bootstrap_redundancy_cap(
    plan: list[AllocationRebalanceEntry],
    records: list[HypothesisRecord],
    *,
    data: dict[str, np.ndarray],
    asset: str,
    corr_max: float,
    floor: float = 0.0,
) -> list[AllocationRebalanceEntry]:
    return _apply_redundancy_cap(
        plan,
        records,
        data=data,
        asset=asset,
        corr_max=corr_max,
        floor=floor,
        eligibility=lambda entry: entry.research_backed,
    )


def apply_capital_redundancy_cap(
    plan: list[AllocationRebalanceEntry],
    records: list[HypothesisRecord],
    *,
    data: dict[str, np.ndarray],
    asset: str,
    corr_max: float,
    floor: float = 0.0,
) -> list[AllocationRebalanceEntry]:
    return _apply_redundancy_cap(
        plan,
        records,
        data=data,
        asset=asset,
        corr_max=corr_max,
        floor=floor,
        eligibility=lambda entry: entry.capital_eligible,
        should_compare=lambda entry, selected_entry: (
            _should_compare_weak_research_family(entry, selected_entry)
            if _both_weak_batch_research(entry, selected_entry)
            else (
                _should_compare_weak_research_family(entry, selected_entry)
                if _both_actionable_live(entry, selected_entry)
                else True
            )
        ),
    )


def apply_weak_research_redundancy_cap(
    plan: list[AllocationRebalanceEntry],
    records: list[HypothesisRecord],
    *,
    data: dict[str, np.ndarray],
    asset: str,
    corr_max: float,
    floor: float = 0.0,
) -> list[AllocationRebalanceEntry]:
    return _apply_redundancy_cap(
        plan,
        records,
        data=data,
        asset=asset,
        corr_max=corr_max,
        floor=floor,
        eligibility=lambda entry: (
            entry.research_retained
            and entry.research_quality_source == "batch_research_score"
            and not entry.live_proven
        ),
        should_compare=lambda entry, selected_entry: (
            _should_compare_weak_research_family(entry, selected_entry)
        ),
    )


def _both_weak_batch_research(
    entry: AllocationRebalanceEntry,
    selected_entry: AllocationRebalanceEntry,
) -> bool:
    return (
        entry.research_retained
        and selected_entry.research_retained
        and entry.research_quality_source == "batch_research_score"
        and selected_entry.research_quality_source == "batch_research_score"
        and not entry.live_proven
        and not selected_entry.live_proven
    )


def _both_actionable_live(
    entry: AllocationRebalanceEntry,
    selected_entry: AllocationRebalanceEntry,
) -> bool:
    return entry.actionable_live and selected_entry.actionable_live


def _should_compare_weak_research_family(
    entry: AllocationRebalanceEntry,
    selected_entry: AllocationRebalanceEntry,
) -> bool:
    return (
        (entry.representative_family or "other")
        == (selected_entry.representative_family or "other")
    )


def apply_live_proven_return_redundancy_cap(
    plan: list[AllocationRebalanceEntry],
    *,
    live_returns_for,
    corr_max: float,
    floor: float = 0.0,
    min_observations: int = 10,
) -> list[AllocationRebalanceEntry]:
    series_by_id: dict[str, np.ndarray] = {}
    for entry in plan:
        if not entry.live_proven or entry.research_backed or entry.proposed_stake <= floor:
            continue
        returns = np.asarray(live_returns_for(entry.hypothesis_id), dtype=np.float64)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        if returns.size < min_observations or float(np.nanstd(returns)) <= 1e-12:
            continue
        series_by_id[entry.hypothesis_id] = returns

    return _apply_series_redundancy_cap(
        plan,
        series_by_id=series_by_id,
        corr_max=corr_max,
        floor=floor,
        rank_key=lambda entry: (
            entry.proposed_stake,
            entry.live_quality,
            entry.blended_quality,
            entry.hypothesis_id,
        ),
    )


def _apply_redundancy_cap(
    plan: list[AllocationRebalanceEntry],
    records: list[HypothesisRecord],
    *,
    data: dict[str, np.ndarray],
    asset: str,
    corr_max: float,
    floor: float,
    eligibility,
    should_compare=None,
) -> list[AllocationRebalanceEntry]:
    record_by_id = {record.hypothesis_id: record for record in records}
    series_by_id: dict[str, np.ndarray] = {}
    for entry in plan:
        if not eligibility(entry) or entry.proposed_stake <= floor:
            continue
        record = record_by_id.get(entry.hypothesis_id)
        if record is None:
            continue
        series = hypothesis_signal_series(record, data=data, asset=asset)
        if series is None:
            continue
        arr = np.asarray(series, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size < 10 or float(np.nanstd(arr)) <= 1e-12:
            continue
        series_by_id[entry.hypothesis_id] = arr

    return _apply_series_redundancy_cap(
        plan,
        series_by_id=series_by_id,
        corr_max=corr_max,
        floor=floor,
        rank_key=lambda entry: (
            entry.proposed_stake,
            entry.bootstrap_trust_value,
            entry.blended_quality,
            entry.hypothesis_id,
        ),
        should_compare=should_compare,
    )


def _apply_series_redundancy_cap(
    plan: list[AllocationRebalanceEntry],
    *,
    series_by_id: dict[str, np.ndarray],
    corr_max: float,
    floor: float,
    rank_key,
    should_compare=None,
) -> list[AllocationRebalanceEntry]:
    if len(series_by_id) <= 1:
        return plan

    selected: list[str] = []
    blocked: dict[str, tuple[str, float]] = {}
    entry_by_id = {entry.hypothesis_id: entry for entry in plan}
    ranked = sorted(
        (entry for entry in plan if entry.hypothesis_id in series_by_id),
        key=rank_key,
        reverse=True,
    )

    for entry in ranked:
        candidate = series_by_id[entry.hypothesis_id]
        blocker: tuple[str, float] | None = None
        for selected_id in selected:
            selected_entry = entry_by_id[selected_id]
            if should_compare is not None and not should_compare(entry, selected_entry):
                continue
            corr = _aligned_corr(candidate, series_by_id[selected_id])
            if np.isfinite(corr) and corr > corr_max:
                blocker = (selected_id, corr)
                break
        if blocker is None:
            selected.append(entry.hypothesis_id)
            continue
        blocked[entry.hypothesis_id] = blocker

    capped: list[AllocationRebalanceEntry] = []
    for entry in plan:
        blocker = blocked.get(entry.hypothesis_id)
        if blocker is None:
            capped.append(entry)
            continue
        capped.append(
            replace(
                entry,
                proposed_stake=float(floor),
                redundancy_capped_by=blocker[0],
                redundancy_correlation=float(blocker[1]),
            )
        )
    return capped


def _aligned_corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    min_size = min(lhs.size, rhs.size)
    if min_size < 2:
        return float("nan")
    if lhs.size != min_size:
        lhs = lhs[-min_size:]
    if rhs.size != min_size:
        rhs = rhs[-min_size:]
    if float(np.nanstd(lhs)) <= 1e-12 or float(np.nanstd(rhs)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(lhs, rhs)[0, 1])


def hypothesis_signal_series(
    record: HypothesisRecord,
    *,
    data: dict[str, np.ndarray],
    asset: str,
) -> np.ndarray | None:
    if record.kind == HypothesisKind.DSL:
        return _dsl_signal_series(record, data)
    if record.kind == HypothesisKind.TECHNICAL:
        return _technical_signal_series(record, data, asset)
    if record.kind == HypothesisKind.ML:
        return _ml_signal_series(record, data, asset)
    return None


def _dsl_signal_series(
    record: HypothesisRecord,
    data: dict[str, np.ndarray],
) -> np.ndarray | None:
    expression = record.definition.get("expression", "")
    if not expression or not data:
        return None
    try:
        expr = parse(expression)
    except SyntaxError:
        return None
    issues = temporal_expression_issues(expr)
    if issues:
        return None
    sample = next(iter(data.values()))
    try:
        signal = evaluate_expression(expr, data, len(sample))
    except Exception:
        return None
    if len(signal) == 0:
        return None
    return normalize_signal(signal)


def _resolve_asset_series_name(asset: str) -> str:
    try:
        return price_signal(asset)
    except KeyError:
        return asset if "_" in asset else asset.lower()


def _technical_signal_series(
    record: HypothesisRecord,
    data: dict[str, np.ndarray],
    asset: str,
) -> np.ndarray | None:
    series = data.get(_resolve_asset_series_name(asset))
    if series is None:
        return None

    indicator = record.definition.get("indicator")
    params = record.definition.get("params", {})

    if indicator == "rsi_reversion":
        return -_center_rank_array(_rank_array(_roc_array(series, params.get("window", 14)), 20))
    if indicator == "zscore_reversion":
        return -_bounded_prediction_array(_zscore_array(series, params.get("window", 60)), scale=2.0)
    if indicator == "roc_momentum":
        return _bounded_prediction_array(_roc_array(series, params.get("window", 20)), scale=0.1)
    if indicator == "roc_reversion":
        return -_bounded_prediction_array(_roc_array(series, params.get("window", 5)), scale=0.1)
    if indicator == "macd_trend":
        return _bounded_prediction_array(
            _macd_signal_array(
                series,
                fast=params.get("fast", 12),
                slow=params.get("slow", 26),
                signal=params.get("signal", 9),
            ),
            scale=0.01,
        )
    if indicator == "bollinger_reversion":
        return -_bounded_prediction_array(
            _zscore_array(series, params.get("window", 20), std_scale=params.get("std", 2.0)),
            scale=2.0,
        )
    if indicator == "breakout":
        return _bounded_prediction_array(_breakout_array(series, params.get("window", 60)) * 2.0)
    if indicator == "low_volatility":
        return -_bounded_prediction_array(
            _rolling_std_array(_roc_array(series, 1), params.get("window", 20)),
            scale=0.05,
        )
    if indicator == "moving_average_cross":
        return _bounded_prediction_array(
            _moving_average_cross_array(
                series,
                fast=params.get("fast", 20),
                slow=params.get("slow", 60),
            ),
            scale=0.02,
        )
    if indicator == "volume_price_confirmation":
        return _bounded_prediction_array(
            _roc_array(series, params.get("price_window", 20)),
            scale=0.1,
        )
    return None


def _ml_signal_series(
    record: HypothesisRecord,
    data: dict[str, np.ndarray],
    asset: str,
) -> np.ndarray | None:
    feature_names = list(record.definition.get("features", []))
    asset_feature = _resolve_asset_series_name(asset)
    if asset_feature not in feature_names:
        feature_names = [asset_feature] + feature_names

    weighted: np.ndarray | None = None
    total_weight = 0.0
    for feature_name in feature_names:
        series = data.get(feature_name)
        if series is None:
            continue
        feature_value = _zscore_array(series, 60)
        weight = _stable_weight(record.hypothesis_id, feature_name)
        if weighted is None:
            weighted = np.zeros_like(feature_value, dtype=np.float64)
        weighted += weight * feature_value
        total_weight += abs(weight)
    if weighted is None or total_weight <= 0:
        return None
    return _bounded_prediction_array(weighted / total_weight, scale=2.0)


def _stable_weight(hypothesis_id: str, feature_name: str) -> float:
    digest = hashlib.md5(
        f"{hypothesis_id}:{feature_name}".encode(),
        usedforsecurity=False,
    ).digest()
    raw = int.from_bytes(digest[:4], "big") / 2**32
    return raw * 2.0 - 1.0


def _bounded_prediction_array(values: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    safe = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.tanh(safe / max(scale, 1e-12))


def _center_rank_array(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - 0.5) * 2.0


def _zscore_array(arr: np.ndarray, window: int = 60, std_scale: float = 1.0) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    out = np.zeros(len(values), dtype=np.float64)
    if len(values) < window:
        return out
    for idx in range(window - 1, len(values)):
        recent = values[idx - window + 1:idx + 1]
        mean = float(np.nanmean(recent))
        std = float(np.nanstd(recent))
        if std >= 1e-12:
            out[idx] = (values[idx] - mean) / (std * std_scale)
    return out


def _rank_array(arr: np.ndarray, window: int = 20) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    out = np.full(len(values), 0.5, dtype=np.float64)
    if len(values) < window:
        return out
    for idx in range(window - 1, len(values)):
        recent = values[idx - window + 1:idx + 1]
        out[idx] = float(np.mean(recent <= values[idx]))
    return out


def _roc_array(arr: np.ndarray, period: int = 5) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    out = np.zeros(len(values), dtype=np.float64)
    if len(values) < period + 1:
        return out
    denom = np.abs(values[:-period]) + 1e-12
    out[period:] = (values[period:] - values[:-period]) / denom
    return out


def _ema(arr: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    if len(values) == 0:
        return np.zeros(0, dtype=np.float64)
    alpha = 2.0 / (window + 1.0)
    out = np.zeros(len(values), dtype=np.float64)
    out[0] = values[0]
    for idx in range(1, len(values)):
        out[idx] = alpha * values[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def _macd_signal_array(arr: np.ndarray, *, fast: int, slow: int, signal: int) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    out = np.zeros(len(values), dtype=np.float64)
    if len(values) < slow + signal:
        return out
    fast_ema = _ema(values, fast)
    slow_ema = _ema(values, slow)
    macd = fast_ema - slow_ema
    signal_line = _ema(macd, signal)
    price = np.abs(values) + 1e-12
    out[slow + signal - 1:] = (macd[slow + signal - 1:] - signal_line[slow + signal - 1:]) / price[slow + signal - 1:]
    return out


def _breakout_array(arr: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    out = np.zeros(len(values), dtype=np.float64)
    if len(values) < window:
        return out
    for idx in range(window - 1, len(values)):
        recent = values[idx - window + 1:idx + 1]
        upper = float(np.nanmax(recent))
        lower = float(np.nanmin(recent))
        if abs(upper - lower) >= 1e-12:
            out[idx] = (values[idx] - lower) / (upper - lower) - 0.5
    return out


def _rolling_std_array(arr: np.ndarray, window: int = 20) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    out = np.zeros(len(values), dtype=np.float64)
    if len(values) < window:
        return out
    for idx in range(window - 1, len(values)):
        out[idx] = float(np.nanstd(values[idx - window + 1:idx + 1]))
    return out


def _moving_average_cross_array(arr: np.ndarray, *, fast: int, slow: int) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    out = np.zeros(len(values), dtype=np.float64)
    if len(values) < slow:
        return out
    for idx in range(slow - 1, len(values)):
        fast_ma = float(np.nanmean(values[max(0, idx - fast + 1):idx + 1]))
        slow_ma = float(np.nanmean(values[idx - slow + 1:idx + 1]))
        if abs(slow_ma) >= 1e-12:
            out[idx] = (fast_ma - slow_ma) / abs(slow_ma)
    return out
