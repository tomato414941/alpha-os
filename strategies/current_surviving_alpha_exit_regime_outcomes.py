from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"


@dataclass(frozen=True)
class SurvivingAlphaExitRegimeOutcome:
    ticket_id: str
    opened_at: str
    checked_at: str
    checkpoint_status: str
    outcome: str
    candidate_id: str
    asset: str
    decision: str
    side: str
    entry_mark: str
    exit_horizon_minutes: int
    stop_bps: float
    first_candle_at: str
    last_candle_at: str
    candle_count: int
    max_favorable_bps: float
    max_adverse_bps: float
    close_return_bps: float
    stop_status: str
    evidence: str
    next_step: str


def build_surviving_alpha_exit_regime_outcomes(
    *,
    tickets_path: Path = ROOT / "current_surviving_alpha_exit_regime_tickets.csv",
    now: datetime | None = None,
    url: str = HYPERLIQUID_INFO_URL,
) -> tuple[SurvivingAlphaExitRegimeOutcome, ...]:
    checked_at = now or datetime.now(UTC)
    return tuple(_outcome_for_ticket(row=row, checked_at=checked_at, url=url) for row in _read_rows(tickets_path))


def write_surviving_alpha_exit_regime_outcomes_csv(
    rows: tuple[SurvivingAlphaExitRegimeOutcome, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "ticket_id",
                "opened_at",
                "checked_at",
                "checkpoint_status",
                "outcome",
                "candidate_id",
                "asset",
                "decision",
                "side",
                "entry_mark",
                "exit_horizon_minutes",
                "stop_bps",
                "first_candle_at",
                "last_candle_at",
                "candle_count",
                "max_favorable_bps",
                "max_adverse_bps",
                "close_return_bps",
                "stop_status",
                "evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.ticket_id,
                    row.opened_at,
                    row.checked_at,
                    row.checkpoint_status,
                    row.outcome,
                    row.candidate_id,
                    row.asset,
                    row.decision,
                    row.side,
                    row.entry_mark,
                    row.exit_horizon_minutes,
                    f"{row.stop_bps:.2f}",
                    row.first_candle_at,
                    row.last_candle_at,
                    row.candle_count,
                    f"{row.max_favorable_bps:.8f}",
                    f"{row.max_adverse_bps:.8f}",
                    f"{row.close_return_bps:.8f}",
                    row.stop_status,
                    row.evidence,
                    row.next_step,
                )
            )
    return output_path


def write_surviving_alpha_exit_regime_outcomes_md(
    rows: tuple[SurvivingAlphaExitRegimeOutcome, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Surviving Alpha Exit Regime Outcomes\n\n")
        handle.write(
            "These outcomes check fresh exit-regime paper tickets against public 1m candle path. "
            "They are paper observations, not live fill reports.\n\n"
        )
        handle.write(
            "| ticket | status | outcome | asset | side | exit min | stop | close | adverse | candles | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.ticket_id} | "
                f"{row.checkpoint_status} | "
                f"{row.outcome} | "
                f"{row.asset} | "
                f"{row.side} | "
                f"{row.exit_horizon_minutes} | "
                f"{row.stop_status} | "
                f"{row.close_return_bps:.4f} | "
                f"{row.max_adverse_bps:.4f} | "
                f"{row.candle_count} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _outcome_for_ticket(
    *,
    row: dict[str, str],
    checked_at: datetime,
    url: str,
) -> SurvivingAlphaExitRegimeOutcome:
    opened_at = _parse_time(row.get("opened_at", ""))
    horizon = int(float(row.get("exit_horizon_minutes") or 0))
    entry = _float(row.get("entry_mark"))
    stop_bps = _float(row.get("stop_bps"))
    if opened_at is None:
        return _missing_row(row=row, checked_at=checked_at, reason="opened_at is missing")
    if entry <= 0.0:
        return _missing_row(row=row, checked_at=checked_at, reason="entry_mark is missing")
    if horizon <= 0:
        return _missing_row(row=row, checked_at=checked_at, reason="exit horizon is missing")

    horizon_end = opened_at + timedelta(minutes=horizon)
    mature = checked_at >= horizon_end
    end_at = horizon_end if mature else checked_at
    candles = tuple(
        candle
        for candle in _fetch_candles(asset=row.get("asset", ""), start=opened_at, end=end_at, url=url)
        if _datetime_from_ms(candle.get("t")) >= opened_at and _datetime_from_ms(candle.get("T")) <= end_at
    )
    if not candles:
        if not mature:
            return _pending_row(row=row, checked_at=checked_at, reason="waiting for first public 1m candle")
        return _missing_row(row=row, checked_at=checked_at, reason="public 1m candle path is missing")

    path = _path_metrics(candles=candles, entry=entry, decision=row.get("decision", ""))
    stop_status = _stop_status(path.max_adverse_bps, threshold_bps=stop_bps)
    checkpoint_status = "ready" if mature else "pending"
    outcome = _outcome(checkpoint_status=checkpoint_status, close_bps=path.close_return_bps, stop_status=stop_status)
    return SurvivingAlphaExitRegimeOutcome(
        ticket_id=row.get("ticket_id", ""),
        opened_at=row.get("opened_at", ""),
        checked_at=checked_at.isoformat(timespec="seconds"),
        checkpoint_status=checkpoint_status,
        outcome=outcome,
        candidate_id=row.get("candidate_id", ""),
        asset=row.get("asset", ""),
        decision=row.get("decision", ""),
        side=row.get("side", ""),
        entry_mark=row.get("entry_mark", ""),
        exit_horizon_minutes=horizon,
        stop_bps=stop_bps,
        first_candle_at=_time_from_ms(candles[0]["t"]),
        last_candle_at=_time_from_ms(candles[-1]["T"]),
        candle_count=len(candles),
        max_favorable_bps=path.max_favorable_bps,
        max_adverse_bps=path.max_adverse_bps,
        close_return_bps=path.close_return_bps,
        stop_status=stop_status,
        evidence=(
            f"candles={len(candles)}; first={_time_from_ms(candles[0]['t'])}; "
            f"last={_time_from_ms(candles[-1]['T'])}; entry={row.get('entry_mark', '')}"
        ),
        next_step=_next_step(row=row, outcome=outcome, stop_status=stop_status),
    )


@dataclass(frozen=True)
class PathMetrics:
    max_favorable_bps: float
    max_adverse_bps: float
    close_return_bps: float


def _path_metrics(*, candles: tuple[dict[str, str], ...], entry: float, decision: str) -> PathMetrics:
    highs = tuple(_float(candle.get("h")) for candle in candles)
    lows = tuple(_float(candle.get("l")) for candle in candles)
    close = _float(candles[-1].get("c"))
    if decision == "paper_short":
        favorable = (entry / min(lows) - 1.0) * 10_000.0 if lows and min(lows) > 0.0 else 0.0
        adverse = (entry / max(highs) - 1.0) * 10_000.0 if highs and max(highs) > 0.0 else 0.0
        close_return = (entry / close - 1.0) * 10_000.0 if close > 0.0 else 0.0
    else:
        favorable = (max(highs) / entry - 1.0) * 10_000.0 if highs else 0.0
        adverse = (min(lows) / entry - 1.0) * 10_000.0 if lows else 0.0
        close_return = (close / entry - 1.0) * 10_000.0 if close > 0.0 else 0.0
    return PathMetrics(max(favorable, 0.0), min(adverse, 0.0), close_return)


def _outcome(*, checkpoint_status: str, close_bps: float, stop_status: str) -> str:
    if checkpoint_status != "ready":
        return "pending"
    if stop_status == "stop_triggered":
        return "paper_exit_regime_stop_loss"
    if close_bps > 0.0:
        return "paper_exit_regime_win"
    return "paper_exit_regime_loss"


def _next_step(*, row: dict[str, str], outcome: str, stop_status: str) -> str:
    asset = row.get("asset", "")
    side = row.get("side", "")
    horizon = row.get("exit_horizon_minutes", "")
    if outcome == "pending":
        return f"wait for {horizon}m exit checkpoint for {asset} {side}"
    if outcome == "paper_exit_regime_win":
        return f"repeat {asset} {side} exit regime once more before promotion"
    if stop_status == "stop_triggered":
        return f"do not promote {asset} {side}; 100bps stop was hit"
    return f"do not promote {asset} {side}; {horizon}m exit failed"


def _missing_row(
    *,
    row: dict[str, str],
    checked_at: datetime,
    reason: str,
) -> SurvivingAlphaExitRegimeOutcome:
    return SurvivingAlphaExitRegimeOutcome(
        ticket_id=row.get("ticket_id", ""),
        opened_at=row.get("opened_at", ""),
        checked_at=checked_at.isoformat(timespec="seconds"),
        checkpoint_status="missing",
        outcome="missing_path",
        candidate_id=row.get("candidate_id", ""),
        asset=row.get("asset", ""),
        decision=row.get("decision", ""),
        side=row.get("side", ""),
        entry_mark=row.get("entry_mark", ""),
        exit_horizon_minutes=int(float(row.get("exit_horizon_minutes") or 0)),
        stop_bps=_float(row.get("stop_bps")),
        first_candle_at="",
        last_candle_at="",
        candle_count=0,
        max_favorable_bps=0.0,
        max_adverse_bps=0.0,
        close_return_bps=0.0,
        stop_status="not_checked",
        evidence=reason,
        next_step=f"fix missing exit-regime ticket data: {reason}",
    )


def _pending_row(
    *,
    row: dict[str, str],
    checked_at: datetime,
    reason: str,
) -> SurvivingAlphaExitRegimeOutcome:
    return SurvivingAlphaExitRegimeOutcome(
        ticket_id=row.get("ticket_id", ""),
        opened_at=row.get("opened_at", ""),
        checked_at=checked_at.isoformat(timespec="seconds"),
        checkpoint_status="pending",
        outcome="pending",
        candidate_id=row.get("candidate_id", ""),
        asset=row.get("asset", ""),
        decision=row.get("decision", ""),
        side=row.get("side", ""),
        entry_mark=row.get("entry_mark", ""),
        exit_horizon_minutes=int(float(row.get("exit_horizon_minutes") or 0)),
        stop_bps=_float(row.get("stop_bps")),
        first_candle_at="",
        last_candle_at="",
        candle_count=0,
        max_favorable_bps=0.0,
        max_adverse_bps=0.0,
        close_return_bps=0.0,
        stop_status="not_checked",
        evidence=reason,
        next_step=f"wait for {row.get('exit_horizon_minutes', '')}m exit checkpoint for {row.get('asset', '')}",
    )


def _fetch_candles(*, asset: str, start: datetime, end: datetime, url: str) -> tuple[dict[str, str], ...]:
    response = requests.post(
        url,
        json={
            "type": "candleSnapshot",
            "req": {
                "coin": asset,
                "interval": "1m",
                "startTime": int(start.timestamp() * 1000),
                "endTime": int(end.timestamp() * 1000),
            },
        },
        timeout=30,
    )
    response.raise_for_status()
    return tuple(response.json())


def _stop_status(max_adverse_bps: float, *, threshold_bps: float) -> str:
    return "stop_triggered" if max_adverse_bps <= -threshold_bps else "stop_survived"


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _time_from_ms(value: object) -> str:
    return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC).isoformat(timespec="seconds")


def _datetime_from_ms(value: object) -> datetime:
    return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickets-path",
        type=Path,
        default=ROOT / "current_surviving_alpha_exit_regime_tickets.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_surviving_alpha_exit_regime_outcomes.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_surviving_alpha_exit_regime_outcomes.md",
    )
    args = parser.parse_args()

    rows = build_surviving_alpha_exit_regime_outcomes(tickets_path=args.tickets_path)
    write_surviving_alpha_exit_regime_outcomes_csv(rows, output_path=args.output_path)
    write_surviving_alpha_exit_regime_outcomes_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.checkpoint_status, row.outcome, f"{row.close_return_bps:.2f}")


if __name__ == "__main__":
    main()
