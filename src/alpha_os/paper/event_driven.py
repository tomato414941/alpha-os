"""Event-driven trading: evaluates on market events, not fixed schedule."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from signal_noise.client import SignalClient

from .trader import Trader, PaperCycleResult

logger = logging.getLogger(__name__)


@dataclass
class EventTriggerConfig:
    # Minimum seconds between evaluations (debounce)
    min_interval: float = 900.0  # 15 minutes
    # Maximum seconds between evaluations (timer fallback)
    max_interval: float = 14400.0  # 4 hours
    # Signal patterns to subscribe to
    subscribe_pattern: str = "funding_rate_*,liq_*"
    # Anomaly events trigger immediate evaluation
    anomaly_trigger: bool = True


class EventDrivenTrader:
    """Event-triggered trading: evaluates on market events, not fixed schedule.

    Wraps an existing Trader instance. Does not replace it.

    Trigger conditions:
    - Timer: max_interval elapsed since last evaluation (fallback)
    - Event: anomaly or circuit_break event received
    - Value change: any update on subscribed signals

    Debounce: at least min_interval between evaluations.
    """

    def __init__(
        self,
        trader: Trader,
        client: SignalClient,
        config: EventTriggerConfig | None = None,
        *,
        pre_cycle_hook: object | None = None,
    ):
        self.trader = trader
        self.client = client
        self.config = config or EventTriggerConfig()
        self._last_eval_time: float = 0.0
        self._running = False
        self._pre_cycle_hook = pre_cycle_hook

    async def run(self) -> None:
        """Main event loop. Blocks until cancelled."""
        self._running = True

        # Run initial cycle immediately
        self._run_cycle("initial")

        timer_task = asyncio.create_task(self._timer_loop())
        event_task = asyncio.create_task(self._event_loop())

        try:
            await asyncio.gather(timer_task, event_task)
        except asyncio.CancelledError:
            timer_task.cancel()
            event_task.cancel()
            self._running = False

    async def _timer_loop(self) -> None:
        """Fallback timer: evaluate at max_interval regardless of events."""
        while self._running:
            await asyncio.sleep(self.config.max_interval)
            if self._should_evaluate("timer"):
                self._run_cycle("timer")

    async def _event_loop(self) -> None:
        """Subscribe to signal events and trigger evaluation."""
        async for event in self.client.subscribe(
            self.config.subscribe_pattern,
        ):
            if not self._running:
                break

            trigger = False
            reason = ""

            if event["event_type"] == "anomaly":
                if self.config.anomaly_trigger:
                    trigger = True
                    reason = f"anomaly: {event['name']}"

            elif event["event_type"] == "circuit_break":
                trigger = True
                reason = f"circuit_break: {event['name']}"

            elif event["event_type"] == "update":
                trigger = True
                reason = f"update: {event['name']}"

            if trigger and self._should_evaluate(reason):
                self._run_cycle(reason)

    def _should_evaluate(self, reason: str) -> bool:
        """Check debounce: at least min_interval since last evaluation."""
        now = time.time()
        elapsed = now - self._last_eval_time
        if elapsed < self.config.min_interval:
            logger.debug(
                "Debounced '%s': %.0fs since last eval (min=%.0fs)",
                reason, elapsed, self.config.min_interval,
            )
            return False
        logger.info("Trigger: %s (%.0fs since last eval)", reason, elapsed)
        return True

    def _run_cycle(self, reason: str) -> PaperCycleResult | None:
        """Execute one trading cycle via the underlying Trader."""
        self._last_eval_time = time.time()
        try:
            if self._pre_cycle_hook is not None:
                self._pre_cycle_hook()  # type: ignore[operator]
            result = self.trader.run_cycle()
            logger.info(
                "Cycle [%s]: signal=%.4f, PV=$%.2f, PnL=$%.2f",
                reason, result.combined_signal,
                result.portfolio_value, result.daily_pnl,
            )
            return result
        except Exception:
            logger.exception("Trading cycle failed [%s]", reason)
            return None
