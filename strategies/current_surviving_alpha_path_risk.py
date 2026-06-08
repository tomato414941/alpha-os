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
class SurvivingAlphaPathRisk:
    work_id: str
    ticket_id: str
    asset: str
    decision: str
    status: str
    entry_mark: str
    first_candle_at: str
    last_candle_at: str
    candle_count: int
    max_favorable_bps: float
    max_adverse_bps: float
    close_return_bps: float
    second_net_after_cost_bps: float
    stop_50bps_status: str
    stop_100bps_status: str
    path_action: str
    evidence: str
    next_step: str


def build_surviving_alpha_path_risk(
    *,
    survival_path: Path = ROOT / "current_alpha_repeat_fill_survival.csv",
    second_tickets_path: Path = ROOT / "current_second_promoted_ticket_repeat_tickets.csv",
    url: str = HYPERLIQUID_INFO_URL,
) -> tuple[SurvivingAlphaPathRisk, ...]:
    second_tickets = {row.get("asset", ""): row for row in _read_rows(second_tickets_path)}
    rows = []
    for row in _read_rows(survival_path):
        if row.get("status") != "second_repeat_cost_survived":
            continue
        ticket = second_tickets.get(row.get("asset", ""), {})
        rows.append(_build_path_risk(row=row, ticket=ticket, url=url))
    return tuple(sorted(rows, key=lambda row: row.second_net_after_cost_bps, reverse=True))


def write_surviving_alpha_path_risk_csv(
    rows: tuple[SurvivingAlphaPathRisk, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "work_id",
                "ticket_id",
                "asset",
                "decision",
                "status",
                "entry_mark",
                "first_candle_at",
                "last_candle_at",
                "candle_count",
                "max_favorable_bps",
                "max_adverse_bps",
                "close_return_bps",
                "second_net_after_cost_bps",
                "stop_50bps_status",
                "stop_100bps_status",
                "path_action",
                "evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.work_id,
                    row.ticket_id,
                    row.asset,
                    row.decision,
                    row.status,
                    row.entry_mark,
                    row.first_candle_at,
                    row.last_candle_at,
                    row.candle_count,
                    f"{row.max_favorable_bps:.8f}",
                    f"{row.max_adverse_bps:.8f}",
                    f"{row.close_return_bps:.8f}",
                    f"{row.second_net_after_cost_bps:.8f}",
                    row.stop_50bps_status,
                    row.stop_100bps_status,
                    row.path_action,
                    row.evidence,
                    row.next_step,
                )
            )
    return output_path


def write_surviving_alpha_path_risk_md(
    rows: tuple[SurvivingAlphaPathRisk, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Surviving Alpha Path Risk\n\n")
        handle.write(
            "This checks second-repeat cost survivors against public 1m candle path risk. "
            "It is not a live fill report and does not prove executable PnL.\n\n"
        )
        handle.write(
            "| work | ticket | asset | action | net | close | favorable | adverse | stop50 | stop100 | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.work_id} | "
                f"{row.ticket_id} | "
                f"{row.asset} | "
                f"{row.path_action} | "
                f"{row.second_net_after_cost_bps:.4f} | "
                f"{row.close_return_bps:.4f} | "
                f"{row.max_favorable_bps:.4f} | "
                f"{row.max_adverse_bps:.4f} | "
                f"{row.stop_50bps_status} | "
                f"{row.stop_100bps_status} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _build_path_risk(*, row: dict[str, str], ticket: dict[str, str], url: str) -> SurvivingAlphaPathRisk:
    asset = row.get("asset", "")
    decision = row.get("decision", "")
    entry_mark = ticket.get("entry_mark", "")
    opened_at = _parse_time(ticket.get("opened_at", ""))
    end_at = _parse_time(row.get("latest_checked_at", "")) or datetime.now(UTC)
    if opened_at is None:
        return _missing_row(row=row, ticket=ticket, reason="ticket opened_at is missing")
    if end_at <= opened_at:
        end_at = opened_at + timedelta(hours=1)
    candles = _fetch_candles(asset=asset, start=opened_at, end=end_at, url=url)
    if not candles:
        return _missing_row(row=row, ticket=ticket, reason="public 1m candle path is missing")
    path = _path_metrics(candles=candles, entry=_float(entry_mark), decision=decision)
    stop_50 = _stop_status(path.max_adverse_bps, threshold_bps=50.0)
    stop_100 = _stop_status(path.max_adverse_bps, threshold_bps=100.0)
    action, next_step = _path_action(
        asset=asset,
        decision=decision,
        net_bps=_float(row.get("second_repeat_net_after_cost_bps")),
        stop_50=stop_50,
        stop_100=stop_100,
        close_bps=path.close_return_bps,
    )
    return SurvivingAlphaPathRisk(
        work_id=row.get("work_id", ""),
        ticket_id=ticket.get("ticket_id", ""),
        asset=asset,
        decision=decision,
        status=row.get("status", ""),
        entry_mark=entry_mark,
        first_candle_at=_time_from_ms(candles[0]["t"]),
        last_candle_at=_time_from_ms(candles[-1]["T"]),
        candle_count=len(candles),
        max_favorable_bps=path.max_favorable_bps,
        max_adverse_bps=path.max_adverse_bps,
        close_return_bps=path.close_return_bps,
        second_net_after_cost_bps=_float(row.get("second_repeat_net_after_cost_bps")),
        stop_50bps_status=stop_50,
        stop_100bps_status=stop_100,
        path_action=action,
        evidence=(
            f"candles={len(candles)}; first={_time_from_ms(candles[0]['t'])}; "
            f"last={_time_from_ms(candles[-1]['T'])}; entry={entry_mark}"
        ),
        next_step=next_step,
    )


@dataclass(frozen=True)
class PathMetrics:
    max_favorable_bps: float
    max_adverse_bps: float
    close_return_bps: float


def _path_metrics(*, candles: tuple[dict[str, str], ...], entry: float, decision: str) -> PathMetrics:
    if entry <= 0.0:
        return PathMetrics(0.0, 0.0, 0.0)
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


def _path_action(
    *,
    asset: str,
    decision: str,
    net_bps: float,
    stop_50: str,
    stop_100: str,
    close_bps: float,
) -> tuple[str, str]:
    if stop_50 == "stop_would_trigger":
        return (
            "stop_risk_blocks_promotion",
            f"do not promote {asset} {decision}; isolate a wider-stop or faster-exit regime before another repeat",
        )
    if stop_100 == "stop_would_trigger":
        return (
            "wide_stop_required",
            f"keep {asset} {decision} as a high-volatility candidate and retest with explicit wide-stop sizing",
        )
    if net_bps > 25.0 and close_bps > 0.0:
        return (
            "path_survived_paper_stop",
            f"open a tiny live-or-paper fill audit for {asset} {decision}; record actual entry, stop, and exit path",
        )
    return (
        "path_edge_unclear",
        f"repeat {asset} {decision} only if a fresh trigger appears with lower adverse excursion",
    )


def _stop_status(max_adverse_bps: float, *, threshold_bps: float) -> str:
    return "stop_would_trigger" if max_adverse_bps <= -threshold_bps else "stop_survived"


def _missing_row(*, row: dict[str, str], ticket: dict[str, str], reason: str) -> SurvivingAlphaPathRisk:
    return SurvivingAlphaPathRisk(
        work_id=row.get("work_id", ""),
        ticket_id=ticket.get("ticket_id", ""),
        asset=row.get("asset", ""),
        decision=row.get("decision", ""),
        status=row.get("status", ""),
        entry_mark=ticket.get("entry_mark", ""),
        first_candle_at="",
        last_candle_at="",
        candle_count=0,
        max_favorable_bps=0.0,
        max_adverse_bps=0.0,
        close_return_bps=0.0,
        second_net_after_cost_bps=_float(row.get("second_repeat_net_after_cost_bps")),
        stop_50bps_status="missing_path",
        stop_100bps_status="missing_path",
        path_action="path_missing",
        evidence=reason,
        next_step=f"collect 1m path data before judging {row.get('asset', '')} {row.get('decision', '')}",
    )


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _time_from_ms(value: object) -> str:
    return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC).isoformat(timespec="seconds")


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
    parser.add_argument("--survival-path", type=Path, default=ROOT / "current_alpha_repeat_fill_survival.csv")
    parser.add_argument(
        "--second-tickets-path",
        type=Path,
        default=ROOT / "current_second_promoted_ticket_repeat_tickets.csv",
    )
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_surviving_alpha_path_risk.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_surviving_alpha_path_risk.md")
    args = parser.parse_args()

    rows = build_surviving_alpha_path_risk(
        survival_path=args.survival_path,
        second_tickets_path=args.second_tickets_path,
    )
    write_surviving_alpha_path_risk_csv(rows, output_path=args.output_path)
    write_surviving_alpha_path_risk_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.asset, row.path_action, f"{row.max_adverse_bps:.2f}", f"{row.close_return_bps:.2f}")


if __name__ == "__main__":
    main()
