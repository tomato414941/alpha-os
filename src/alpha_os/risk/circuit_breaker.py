"""Circuit breaker: kill switch, daily loss limit, consecutive losses, drawdown halt.

Checked before every trade cycle. Once halted, stays halted until the next
daily reset (or manual removal of the halt). State is persisted to JSON
so it survives process restarts.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from ..config import DATA_DIR

logger = logging.getLogger(__name__)

CB_STATE_PATH = DATA_DIR / "metrics" / "circuit_breaker.json"


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text to file atomically via tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(tmp_fd, content.encode())
        os.close(tmp_fd)
        os.replace(tmp_path, str(path))
    except BaseException:
        os.close(tmp_fd)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


@dataclass
class CircuitBreakerConfig:
    daily_loss_limit_pct: float = 3.0
    max_consecutive_losses: int = 5
    max_drawdown_pct: float = 10.0
    kill_file: str = "data/KILL_SWITCH"


@dataclass
class CircuitBreaker:
    """Centralized risk circuit breaker. Checked before every trade cycle."""

    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    _state_path: Path = field(default_factory=lambda: CB_STATE_PATH)
    _daily_start_equity: float = 0.0
    _daily_pnl: float = 0.0
    _consecutive_losses: int = 0
    _peak_equity: float = 0.0
    _halted: bool = False
    _halt_reason: str = ""
    _current_date: str = ""

    def reset_daily(self, equity: float) -> None:
        """Reset daily counters if the date has changed."""
        today = date.today().isoformat()
        if today != self._current_date:
            self._current_date = today
            self._daily_start_equity = equity
            self._daily_pnl = 0.0
            self._halted = False
            self._halt_reason = ""
            logger.info("Circuit breaker daily reset: equity=$%.2f", equity)
            self.save()

    def record_trade(self, pnl: float) -> None:
        """Record a trade's P&L for daily loss and consecutive loss tracking."""
        self._daily_pnl += pnl
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        self.save()

    def is_safe_to_trade(self, current_equity: float) -> tuple[bool, str]:
        """Check all safety limits. Returns (safe, reason)."""
        # 1. Kill switch file
        if Path(self.config.kill_file).exists():
            return False, "Kill switch file detected"

        # 2. Already halted
        if self._halted:
            return False, f"Halted: {self._halt_reason}"

        # 3. Daily loss limit
        if self._daily_start_equity > 0:
            daily_loss_pct = -self._daily_pnl / self._daily_start_equity * 100
            if daily_loss_pct > self.config.daily_loss_limit_pct:
                self._halted = True
                self._halt_reason = (
                    f"Daily loss {daily_loss_pct:.1f}% > "
                    f"limit {self.config.daily_loss_limit_pct}%"
                )
                logger.error("CIRCUIT BREAKER: %s", self._halt_reason)
                self.save()
                return False, self._halt_reason

        # 4. Consecutive losses
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            self._halted = True
            self._halt_reason = f"Consecutive losses: {self._consecutive_losses}"
            logger.error("CIRCUIT BREAKER: %s", self._halt_reason)
            self.save()
            return False, self._halt_reason

        # 5. Max drawdown from peak
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        if self._peak_equity > 0:
            dd_pct = (self._peak_equity - current_equity) / self._peak_equity * 100
            if dd_pct > self.config.max_drawdown_pct:
                self._halted = True
                self._halt_reason = (
                    f"Drawdown {dd_pct:.1f}% > limit {self.config.max_drawdown_pct}%"
                )
                logger.error("CIRCUIT BREAKER: %s", self._halt_reason)
                self.save()
                return False, self._halt_reason

        return True, "ok"

    def save(self, path: Path | None = None) -> None:
        """Persist state to JSON."""
        path = path or self._state_path
        data = {
            "daily_start_equity": self._daily_start_equity,
            "daily_pnl": self._daily_pnl,
            "consecutive_losses": self._consecutive_losses,
            "peak_equity": self._peak_equity,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "current_date": self._current_date,
        }
        _atomic_write_text(path, json.dumps(data, indent=2))

    @classmethod
    def load(
        cls,
        config: CircuitBreakerConfig | None = None,
        path: Path | None = None,
    ) -> CircuitBreaker:
        """Load state from JSON, or return fresh instance if file missing."""
        config = config or CircuitBreakerConfig()
        path = path or CB_STATE_PATH
        cb = cls(config=config, _state_path=path)
        if not path.exists():
            return cb
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load circuit breaker state: %s", e)
            return cb
        cb._daily_start_equity = data.get("daily_start_equity", 0.0)
        cb._daily_pnl = data.get("daily_pnl", 0.0)
        cb._consecutive_losses = data.get("consecutive_losses", 0)
        cb._peak_equity = data.get("peak_equity", 0.0)
        cb._halted = data.get("halted", False)
        cb._halt_reason = data.get("halt_reason", "")
        cb._current_date = data.get("current_date", "")
        logger.info(
            "Circuit breaker loaded: pnl=$%.2f, peak=$%.2f, halted=%s",
            cb._daily_pnl, cb._peak_equity, cb._halted,
        )
        return cb

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason
