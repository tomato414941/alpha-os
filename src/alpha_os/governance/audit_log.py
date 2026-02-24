"""JSONL audit log — append-only event recording for governance."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from ..config import DATA_DIR


@dataclass
class AuditEvent:
    timestamp: float
    event_type: str
    alpha_id: str
    details: dict


class AuditLog:
    """Append-only JSONL audit log for alpha lifecycle events."""

    def __init__(self, log_path: Path | None = None):
        self._path = log_path or DATA_DIR / "audit.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        event_type: str,
        alpha_id: str = "",
        details: dict | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            timestamp=time.time(),
            event_type=event_type,
            alpha_id=alpha_id,
            details=details or {},
        )
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(event)) + "\n")
        return event

    def log_state_change(
        self, alpha_id: str, old_state: str, new_state: str, reason: str = ""
    ) -> AuditEvent:
        return self.log(
            "state_change",
            alpha_id=alpha_id,
            details={"old_state": old_state, "new_state": new_state, "reason": reason},
        )

    def log_adoption(self, alpha_id: str, expression: str, metrics: dict) -> AuditEvent:
        return self.log(
            "adoption",
            alpha_id=alpha_id,
            details={"expression": expression, **metrics},
        )

    def log_retirement(self, alpha_id: str, reason: str) -> AuditEvent:
        return self.log(
            "retirement",
            alpha_id=alpha_id,
            details={"reason": reason},
        )

    def log_pipeline_run(self, details: dict) -> AuditEvent:
        return self.log("pipeline_run", details=details)

    def log_trade(
        self, alpha_id: str, symbol: str, side: str, qty: float, price: float
    ) -> AuditEvent:
        return self.log(
            "trade",
            alpha_id=alpha_id,
            details={"symbol": symbol, "side": side, "qty": qty, "price": price},
        )

    def read_all(self) -> list[AuditEvent]:
        if not self._path.exists():
            return []
        events = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    events.append(AuditEvent(**d))
        return events

    def read_by_type(self, event_type: str) -> list[AuditEvent]:
        return [e for e in self.read_all() if e.event_type == event_type]

    def read_by_alpha(self, alpha_id: str) -> list[AuditEvent]:
        return [e for e in self.read_all() if e.alpha_id == alpha_id]
