"""Testnet validation — daily health checks and consecutive success tracking.

Phase 4 success criteria: 10 consecutive days without errors or unexpected state.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from ..config import DATA_DIR

logger = logging.getLogger(__name__)

VALIDATION_STATE_PATH = DATA_DIR / "metrics" / "testnet_validation.json"
VALIDATION_REPORT_PATH = DATA_DIR / "metrics" / "testnet_reports.jsonl"


@dataclass
class DailyReport:
    """Structured health report for one testnet cycle."""

    date: str
    timestamp: float
    # Cycle outcome
    cycle_completed: bool
    error_message: str = ""
    # Portfolio
    portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    daily_return: float = 0.0
    # Fills
    n_fills: int = 0
    mean_slippage_bps: float = 0.0
    mean_latency_ms: float = 0.0
    # Reconciliation
    reconciliation_match: bool = False
    qty_diff: float = 0.0
    cash_diff: float = 0.0
    # Circuit breaker
    circuit_breaker_halted: bool = False
    circuit_breaker_reason: str = ""
    # Alphas
    n_alphas_active: int = 0
    n_alphas_evaluated: int = 0
    # Order failures
    n_order_failures: int = 0
    # Validation result
    has_errors: bool = False
    error_details: list[str] = field(default_factory=list)


@dataclass
class ValidationState:
    """Persistent state for Phase 4 validation tracking."""

    consecutive_success_days: int = 0
    total_days_run: int = 0
    last_success_date: str = ""
    last_run_date: str = ""
    first_run_date: str = ""
    target_days: int = 10
    passed: bool = False


class TestnetValidator:
    """Run post-cycle health checks and track Phase 4 progress."""

    def __init__(
        self,
        state_path: Path | None = None,
        report_path: Path | None = None,
        target_days: int = 10,
        max_slippage_bps: float = 50.0,
    ) -> None:
        self._state_path = state_path or VALIDATION_STATE_PATH
        self._report_path = report_path or VALIDATION_REPORT_PATH
        self._max_slippage_bps = max_slippage_bps
        self._state = self._load_state()
        self._state.target_days = target_days

    def _load_state(self) -> ValidationState:
        if not self._state_path.exists():
            return ValidationState()
        try:
            data = json.loads(self._state_path.read_text())
            return ValidationState(**{
                k: v for k, v in data.items()
                if k in ValidationState.__dataclass_fields__
            })
        except (json.JSONDecodeError, TypeError):
            return ValidationState()

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(asdict(self._state), indent=2))

    def _append_report(self, report: DailyReport) -> None:
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._report_path, "a") as f:
            f.write(json.dumps(asdict(report)) + "\n")

    def validate_cycle(
        self,
        cycle_result,
        reconciliation: dict,
        circuit_breaker,
        fills: list,
        *,
        order_failures: int = 0,
        today_override: str | None = None,
    ) -> DailyReport:
        """Run all health checks and produce a daily report."""
        today = today_override or date.today().isoformat()
        errors: list[str] = []

        # Extract fill metrics
        slippages = [f.slippage_bps for f in fills]
        latencies = [f.latency_ms for f in fills]
        mean_slip = sum(slippages) / len(slippages) if slippages else 0.0
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0

        # Check 1: Cycle completed (non-zero portfolio)
        cycle_ok = cycle_result.portfolio_value > 0
        if not cycle_ok:
            errors.append("Portfolio value is zero or negative")

        # Check 2: Reconciliation
        recon_match = reconciliation.get("match", False)
        if not recon_match and reconciliation.get("status") != "no_data":
            qty_d = reconciliation.get("qty_diff", 0)
            cash_d = reconciliation.get("cash_diff", 0)
            errors.append(
                f"Reconciliation mismatch: qty_diff={qty_d:.6f}, cash_diff={cash_d:.2f}"
            )

        # Check 3: Slippage within bounds
        if mean_slip > self._max_slippage_bps:
            errors.append(f"Extreme slippage: mean={mean_slip:.1f} bps")

        # Check 4: Order failures
        if order_failures > 0:
            errors.append(f"Order failures: {order_failures} orders skipped")

        report = DailyReport(
            date=today,
            timestamp=time.time(),
            cycle_completed=cycle_ok,
            portfolio_value=cycle_result.portfolio_value,
            daily_pnl=cycle_result.daily_pnl,
            daily_return=cycle_result.daily_return,
            n_fills=len(fills),
            mean_slippage_bps=mean_slip,
            mean_latency_ms=mean_lat,
            reconciliation_match=recon_match,
            qty_diff=reconciliation.get("qty_diff", 0.0),
            cash_diff=reconciliation.get("cash_diff", 0.0),
            circuit_breaker_halted=circuit_breaker.halted,
            circuit_breaker_reason=circuit_breaker.halt_reason,
            n_alphas_active=cycle_result.n_alphas_active,
            n_alphas_evaluated=cycle_result.n_alphas_evaluated,
            n_order_failures=order_failures,
            has_errors=len(errors) > 0,
            error_details=errors,
        )

        self._update_state(today, report)
        self._append_report(report)
        self._save_state()

        return report

    def _update_state(self, today: str, report: DailyReport) -> None:
        s = self._state
        if not s.first_run_date:
            s.first_run_date = today

        # Only count once per calendar day
        if today == s.last_run_date:
            return

        s.total_days_run += 1
        s.last_run_date = today

        if report.has_errors:
            s.consecutive_success_days = 0
        else:
            s.consecutive_success_days += 1
            s.last_success_date = today

        if s.consecutive_success_days >= s.target_days:
            s.passed = True

    @property
    def state(self) -> ValidationState:
        return self._state

    def print_status(self) -> None:
        s = self._state
        status = "PASSED" if s.passed else "IN PROGRESS"
        print(f"\nPhase 4 Testnet Validation: {status}")
        print(f"  Consecutive success days: {s.consecutive_success_days}/{s.target_days}")
        print(f"  Total days run: {s.total_days_run}")
        print(f"  First run: {s.first_run_date or 'N/A'}")
        print(f"  Last run:  {s.last_run_date or 'N/A'}")
        if s.passed:
            print(f"  PASSED on {s.last_success_date}")
