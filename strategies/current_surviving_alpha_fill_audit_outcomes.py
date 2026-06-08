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
class SurvivingAlphaFillAuditOutcome:
    ticket_id: str
    opened_at: str
    checked_at: str
    horizon: str
    checkpoint_status: str
    outcome: str
    asset: str
    decision: str
    side: str
    entry_mark: str
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


def build_surviving_alpha_fill_audit_outcomes(
    *,
    tickets_path: Path = ROOT / "current_surviving_alpha_fill_audit_tickets.csv",
    now: datetime | None = None,
    url: str = HYPERLIQUID_INFO_URL,
) -> tuple[SurvivingAlphaFillAuditOutcome, ...]:
    checked_at = now or datetime.now(UTC)
    rows = []
    for ticket in _read_rows(tickets_path):
        for horizon in _horizons(ticket.get("audit_horizons", "")):
            rows.append(_outcome_for_ticket(ticket=ticket, horizon=horizon, checked_at=checked_at, url=url))
    return tuple(rows)


def write_surviving_alpha_fill_audit_outcomes_csv(
    rows: tuple[SurvivingAlphaFillAuditOutcome, ...],
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
                "horizon",
                "checkpoint_status",
                "outcome",
                "asset",
                "decision",
                "side",
                "entry_mark",
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
                    row.horizon,
                    row.checkpoint_status,
                    row.outcome,
                    row.asset,
                    row.decision,
                    row.side,
                    row.entry_mark,
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


def write_surviving_alpha_fill_audit_outcomes_md(
    rows: tuple[SurvivingAlphaFillAuditOutcome, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Surviving Alpha Fill Audit Outcomes\n\n")
        handle.write(
            "These outcomes check fresh fill-audit paper tickets against public 1m candle path. "
            "They do not prove live fill quality.\n\n"
        )
        handle.write("| ticket | horizon | status | outcome | asset | side | close | adverse | stop | candles | next step |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.ticket_id} | "
                f"{row.horizon} | "
                f"{row.checkpoint_status} | "
                f"{row.outcome} | "
                f"{row.asset} | "
                f"{row.side} | "
                f"{row.close_return_bps:.4f} | "
                f"{row.max_adverse_bps:.4f} | "
                f"{row.stop_status} | "
                f"{row.candle_count} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _outcome_for_ticket(
    *,
    ticket: dict[str, str],
    horizon: tuple[str, int],
    checked_at: datetime,
    url: str,
) -> SurvivingAlphaFillAuditOutcome:
    horizon_name, horizon_minutes = horizon
    opened_at = _parse_time(ticket.get("opened_at", ""))
    entry = _float(ticket.get("entry_mark"))
    stop_bps = _float(ticket.get("stop_bps"))
    if opened_at is None:
        return _pending_or_missing(ticket=ticket, checked_at=checked_at, horizon_name=horizon_name, reason="opened_at is missing")
    if entry <= 0.0:
        return _pending_or_missing(ticket=ticket, checked_at=checked_at, horizon_name=horizon_name, reason="entry_mark is missing")
    horizon_end = opened_at + timedelta(minutes=horizon_minutes)
    mature = checked_at >= horizon_end
    end_at = horizon_end if mature else checked_at
    candles = tuple(
        candle
        for candle in _fetch_candles(asset=ticket.get("asset", ""), start=opened_at, end=end_at, url=url)
        if _datetime_from_ms(candle.get("t")) >= opened_at and _datetime_from_ms(candle.get("T")) <= end_at
    )
    if not candles:
        return _pending_or_missing(
            ticket=ticket,
            checked_at=checked_at,
            horizon_name=horizon_name,
            reason="waiting for first public 1m candle" if not mature else "public 1m candle path is missing",
        )
    path = _path_metrics(candles=candles, entry=entry, decision=ticket.get("decision", ""))
    stop_status = "stop_triggered" if path.max_adverse_bps <= -stop_bps else "stop_survived"
    checkpoint_status = "ready" if mature else "pending"
    outcome = _outcome(checkpoint_status=checkpoint_status, close_bps=path.close_return_bps, stop_status=stop_status)
    return SurvivingAlphaFillAuditOutcome(
        ticket_id=ticket.get("ticket_id", ""),
        opened_at=ticket.get("opened_at", ""),
        checked_at=checked_at.isoformat(timespec="seconds"),
        horizon=horizon_name,
        checkpoint_status=checkpoint_status,
        outcome=outcome,
        asset=ticket.get("asset", ""),
        decision=ticket.get("decision", ""),
        side=ticket.get("side", ""),
        entry_mark=ticket.get("entry_mark", ""),
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
            f"last={_time_from_ms(candles[-1]['T'])}; entry={ticket.get('entry_mark', '')}"
        ),
        next_step=_next_step(ticket=ticket, horizon_name=horizon_name, outcome=outcome, stop_status=stop_status),
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
        return "paper_fill_audit_stop_loss"
    if close_bps > 0.0:
        return "paper_fill_audit_win"
    return "paper_fill_audit_loss"


def _next_step(*, ticket: dict[str, str], horizon_name: str, outcome: str, stop_status: str) -> str:
    asset = ticket.get("asset", "")
    side = ticket.get("side", "")
    if outcome == "pending":
        return f"wait for {horizon_name} fill-audit checkpoint for {asset} {side}"
    if outcome == "paper_fill_audit_win":
        return f"compare {asset} {side} fill audit against prior repeat path before promotion"
    if stop_status == "stop_triggered":
        return f"do not promote {asset} {side}; fresh fill audit hit the stop"
    return f"do not promote {asset} {side}; fresh fill audit failed at {horizon_name}"


def _pending_or_missing(
    *,
    ticket: dict[str, str],
    checked_at: datetime,
    horizon_name: str,
    reason: str,
) -> SurvivingAlphaFillAuditOutcome:
    return SurvivingAlphaFillAuditOutcome(
        ticket_id=ticket.get("ticket_id", ""),
        opened_at=ticket.get("opened_at", ""),
        checked_at=checked_at.isoformat(timespec="seconds"),
        horizon=horizon_name,
        checkpoint_status="pending" if reason.startswith("waiting") else "missing",
        outcome="pending" if reason.startswith("waiting") else "missing_path",
        asset=ticket.get("asset", ""),
        decision=ticket.get("decision", ""),
        side=ticket.get("side", ""),
        entry_mark=ticket.get("entry_mark", ""),
        stop_bps=_float(ticket.get("stop_bps")),
        first_candle_at="",
        last_candle_at="",
        candle_count=0,
        max_favorable_bps=0.0,
        max_adverse_bps=0.0,
        close_return_bps=0.0,
        stop_status="not_checked",
        evidence=reason,
        next_step=f"wait for {horizon_name} fill-audit checkpoint for {ticket.get('asset', '')}",
    )


def _horizons(value: str) -> tuple[tuple[str, int], ...]:
    rows = []
    for token in value.split(","):
        horizon = token.strip()
        if horizon.endswith("m") and horizon[:-1].isdigit():
            rows.append((horizon, int(horizon[:-1])))
        elif horizon.endswith("h") and horizon[:-1].isdigit():
            rows.append((horizon, int(horizon[:-1]) * 60))
    return tuple(rows) or (("15m", 15), ("1h", 60))


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


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _time_from_ms(value: object) -> str:
    return _datetime_from_ms(value).isoformat(timespec="seconds")


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
        default=ROOT / "current_surviving_alpha_fill_audit_tickets.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_surviving_alpha_fill_audit_outcomes.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_surviving_alpha_fill_audit_outcomes.md",
    )
    args = parser.parse_args()

    rows = build_surviving_alpha_fill_audit_outcomes(tickets_path=args.tickets_path)
    write_surviving_alpha_fill_audit_outcomes_csv(rows, output_path=args.output_path)
    write_surviving_alpha_fill_audit_outcomes_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.ticket_id, row.horizon, row.checkpoint_status, row.outcome, f"{row.close_return_bps:.2f}")


if __name__ == "__main__":
    main()
