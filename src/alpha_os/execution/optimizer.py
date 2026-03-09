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
    max_deferral_attempts: int = 2
    deferral_sleep_seconds: float = 30.0


class ExecutionOptimizer:
    """Layer 1: decide WHEN to execute based on microstructure signals."""

    def __init__(
        self,
        client: SignalClient,
        config: ExecutionConfig | None = None,
    ):
        self._client = client
        self._config = config or ExecutionConfig()

    @property
    def max_deferral_attempts(self) -> int:
        return max(1, int(self._config.max_deferral_attempts))

    @property
    def deferral_sleep_seconds(self) -> float:
        return max(0.0, float(self._config.deferral_sleep_seconds))

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

    def optimal_execution_window(self, side: str) -> bool:
        """Check if microstructure conditions favor execution now."""
        imbalance = self.get_signal("book_imbalance_btc")
        vpin = self.get_signal("vpin_btc")
        spread = self.get_signal("spread_bps_btc")

        # If signals unavailable, don't block execution
        if any(v is None for v in [imbalance, vpin, spread]):
            return True

        cfg = self._config

        if vpin > cfg.vpin_threshold:
            logger.info("Execution delayed: VPIN %.3f > %.3f", vpin, cfg.vpin_threshold)
            return False

        if spread > cfg.spread_threshold_bps:
            logger.info("Execution delayed: spread %.1f > %.1f bps", spread, cfg.spread_threshold_bps)
            return False

        # Unfavorable imbalance for our side
        if side == "buy" and imbalance < -cfg.imbalance_threshold:
            logger.info("Execution delayed: ask-heavy imbalance %.3f for buy", imbalance)
            return False
        if side == "sell" and imbalance > cfg.imbalance_threshold:
            logger.info("Execution delayed: bid-heavy imbalance %.3f for sell", imbalance)
            return False

        return True

    def split_order(self, total_qty: float) -> list[float]:
        """TWAP-style order splitting based on current conditions."""
        cfg = self._config
        vpin = self.get_signal("vpin_btc")
        spread = self.get_signal("spread_bps_btc")

        if vpin is not None and vpin > cfg.vpin_threshold:
            n_slices = cfg.max_slices
        elif spread is not None and spread > cfg.spread_threshold_bps:
            n_slices = max(3, cfg.max_slices - 1)
        else:
            n_slices = 1

        slice_qty = total_qty / n_slices
        return [slice_qty] * n_slices
