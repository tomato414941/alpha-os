"""Execution optimizer: microstructure-aware order timing."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from signal_noise.client import SignalClient

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    imbalance_threshold: float = 0.3
    vpin_threshold: float = 0.85
    spread_threshold_bps: float = 5.0
    max_slices: int = 5
    signal_lookback_minutes: int = 15
    max_signal_age_seconds: int = 300


@dataclass
class ExecutionAdvice:
    reason: str = ""
    slice_count: int = 1


class ExecutionOptimizer:
    """Execution optimizer: convert microstructure state into execution advice."""

    def __init__(
        self,
        client: SignalClient,
        config: ExecutionConfig | None = None,
    ):
        self._client = client
        self._config = config or ExecutionConfig()

    def _parse_timestamp(self, raw: object) -> datetime | None:
        if not isinstance(raw, str) or not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            return None

    def _extract_latest_value(self, payload: dict | None) -> float | None:
        if not isinstance(payload, dict):
            return None

        value = payload.get("value")
        if value is None:
            return None

        timestamp = self._parse_timestamp(payload.get("timestamp"))
        if timestamp is None:
            return float(value)

        age = datetime.now(UTC) - timestamp
        if age.total_seconds() <= self._config.max_signal_age_seconds:
            return float(value)

        logger.info(
            "Ignoring stale latest %s: age=%.0fs > %ss",
            payload.get("name", "signal"),
            age.total_seconds(),
            self._config.max_signal_age_seconds,
        )
        return None

    def _get_recent_signal(self, name: str) -> float | None:
        since = (datetime.now(UTC) - timedelta(minutes=self._config.signal_lookback_minutes)).isoformat()
        try:
            frame = self._client.get_data(name, since=since)
        except Exception:
            return None

        if frame is None or frame.empty:
            return None

        latest_row = frame.iloc[-1]
        value = latest_row.get("value")
        if value is None:
            return None

        timestamp = latest_row.get("timestamp")
        if timestamp is not None:
            ts = timestamp.to_pydatetime().astimezone(UTC)
            age = datetime.now(UTC) - ts
            if age.total_seconds() > self._config.max_signal_age_seconds:
                return None

        return float(value)

    def get_signal(self, name: str) -> float | None:
        recent = self._get_recent_signal(name)
        if recent is not None:
            return recent

        try:
            latest = self._client.get_latest(name)
            if latest is None:
                return None
            return self._extract_latest_value(latest)
        except Exception:
            return None

    def execution_advice(self, side: str) -> ExecutionAdvice:
        """Return microstructure-aware execution advice without hard blocking."""
        imbalance = self.get_signal("book_imbalance_btc")
        vpin = self.get_signal("vpin_btc")
        spread = self.get_signal("spread_bps_btc")

        if any(v is None for v in [imbalance, vpin, spread]):
            return ExecutionAdvice()

        cfg = self._config
        reasons: list[str] = []
        slice_count = 1

        if vpin > cfg.vpin_threshold:
            reasons.append(f"high VPIN {vpin:.3f} > {cfg.vpin_threshold:.3f}")
            slice_count = max(slice_count, cfg.max_slices)

        if spread > cfg.spread_threshold_bps:
            reasons.append(
                f"wide spread {spread:.1f} > {cfg.spread_threshold_bps:.1f} bps"
            )
            slice_count = max(slice_count, max(3, cfg.max_slices - 1))

        if side == "buy" and imbalance < -cfg.imbalance_threshold:
            reasons.append(f"ask-heavy imbalance {imbalance:.3f} for buy")
            slice_count = max(slice_count, max(3, cfg.max_slices - 1))
        elif side == "sell" and imbalance > cfg.imbalance_threshold:
            reasons.append(f"bid-heavy imbalance {imbalance:.3f} for sell")
            slice_count = max(slice_count, max(3, cfg.max_slices - 1))

        return ExecutionAdvice(reason="; ".join(reasons), slice_count=slice_count)

    def split_order(self, total_qty: float, side: str) -> list[float]:
        """Split order quantity according to current execution advice."""
        advice = self.execution_advice(side)
        slice_qty = total_qty / advice.slice_count
        return [slice_qty] * advice.slice_count
