"""Pipeline scheduler — periodic autonomous loop execution."""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    interval_seconds: int = 86400  # daily
    max_runs: int = 0  # 0 = unlimited
    retry_delay: int = 300  # 5 min on failure


class PipelineScheduler:
    """Run pipeline on a schedule with graceful shutdown support."""

    def __init__(
        self,
        run_fn: Callable[[], None],
        config: SchedulerConfig | None = None,
    ):
        self.run_fn = run_fn
        self.config = config or SchedulerConfig()
        self._running = False
        self._run_count = 0

    def start(self) -> None:
        """Start the scheduling loop. Blocks until stopped."""
        self._running = True
        self._setup_signal_handlers()
        cfg = self.config

        logger.info(
            f"Scheduler started: interval={cfg.interval_seconds}s, "
            f"max_runs={'unlimited' if cfg.max_runs == 0 else cfg.max_runs}"
        )

        while self._running:
            if cfg.max_runs > 0 and self._run_count >= cfg.max_runs:
                logger.info(f"Reached max_runs={cfg.max_runs}, stopping")
                self._running = False
                break

            try:
                logger.info(f"Run #{self._run_count + 1} starting...")
                t0 = time.time()
                self.run_fn()
                elapsed = time.time() - t0
                self._run_count += 1
                logger.info(f"Run #{self._run_count} completed in {elapsed:.1f}s")
            except Exception as e:
                logger.error(f"Run failed: {e}")
                logger.info(f"Retrying in {cfg.retry_delay}s...")
                self._sleep(cfg.retry_delay)
                continue

            if self._running:
                logger.info(f"Next run in {cfg.interval_seconds}s")
                self._sleep(cfg.interval_seconds)

        logger.info(f"Scheduler stopped after {self._run_count} runs")

    def stop(self) -> None:
        """Signal the scheduler to stop after current run."""
        self._running = False
        logger.info("Stop requested")

    @property
    def run_count(self) -> int:
        return self._run_count

    @property
    def is_running(self) -> bool:
        return self._running

    def _sleep(self, seconds: int) -> None:
        """Interruptible sleep."""
        end = time.time() + seconds
        while self._running and time.time() < end:
            time.sleep(min(1.0, end - time.time()))

    def _setup_signal_handlers(self) -> None:
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except (OSError, ValueError):
            pass  # not main thread or unsupported platform

    def _handle_signal(self, signum, frame) -> None:
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
