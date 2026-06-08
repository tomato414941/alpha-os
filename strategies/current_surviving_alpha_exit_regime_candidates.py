from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
HORIZONS_MINUTES = (5, 10, 15, 30, 60, 120, 240)


@dataclass(frozen=True)
class SurvivingAlphaExitRegimeCandidate:
    candidate_id: str
    asset: str
    decision: str
    horizon_minutes: int
    status: str
    close_return_bps: float
    max_favorable_bps: float
    max_adverse_bps: float
    stop_50bps_status: str
    stop_100bps_status: str
    priority: float
    required_record: str
    next_step: str


def build_surviving_alpha_exit_regime_candidates(
    *,
    path_risk_path: Path = ROOT / "current_surviving_alpha_path_risk.csv",
    second_tickets_path: Path = ROOT / "current_second_promoted_ticket_repeat_tickets.csv",
    url: str = HYPERLIQUID_INFO_URL,
) -> tuple[SurvivingAlphaExitRegimeCandidate, ...]:
    tickets = {row.get("ticket_id", ""): row for row in _read_rows(second_tickets_path)}
    rows = []
    for path_row in _read_rows(path_risk_path):
        if path_row.get("path_action") not in {"stop_risk_blocks_promotion", "wide_stop_required"}:
            continue
        ticket = tickets.get(path_row.get("ticket_id", ""), {})
        rows.extend(_candidate_rows(path_row=path_row, ticket=ticket, url=url))
    return tuple(sorted(rows, key=lambda row: row.priority, reverse=True))


def write_surviving_alpha_exit_regime_candidates_csv(
    rows: tuple[SurvivingAlphaExitRegimeCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "candidate_id",
                "asset",
                "decision",
                "horizon_minutes",
                "status",
                "close_return_bps",
                "max_favorable_bps",
                "max_adverse_bps",
                "stop_50bps_status",
                "stop_100bps_status",
                "priority",
                "required_record",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.candidate_id,
                    row.asset,
                    row.decision,
                    row.horizon_minutes,
                    row.status,
                    f"{row.close_return_bps:.8f}",
                    f"{row.max_favorable_bps:.8f}",
                    f"{row.max_adverse_bps:.8f}",
                    row.stop_50bps_status,
                    row.stop_100bps_status,
                    f"{row.priority:.8f}",
                    row.required_record,
                    row.next_step,
                )
            )
    return output_path


def write_surviving_alpha_exit_regime_candidates_md(
    rows: tuple[SurvivingAlphaExitRegimeCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Surviving Alpha Exit Regime Candidates\n\n")
        handle.write(
            "This searches stop/exit regimes for second-repeat survivors that had path-risk issues. "
            "It is a paper path review, not a live execution rule.\n\n"
        )
        handle.write(
            "| candidate | asset | decision | horizon | status | close | adverse | stop50 | stop100 | priority | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- | ---: | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.candidate_id} | "
                f"{row.asset} | "
                f"{row.decision} | "
                f"{row.horizon_minutes} | "
                f"{row.status} | "
                f"{row.close_return_bps:.4f} | "
                f"{row.max_adverse_bps:.4f} | "
                f"{row.stop_50bps_status} | "
                f"{row.stop_100bps_status} | "
                f"{row.priority:.4f} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _candidate_rows(
    *,
    path_row: dict[str, str],
    ticket: dict[str, str],
    url: str,
) -> tuple[SurvivingAlphaExitRegimeCandidate, ...]:
    asset = path_row.get("asset", "")
    decision = path_row.get("decision", "")
    entry = _float(path_row.get("entry_mark"))
    opened_at = _parse_time(ticket.get("opened_at", ""))
    checked_at = _parse_time(path_row.get("last_candle_at", ""))
    if entry <= 0.0 or opened_at is None or checked_at is None:
        return ()
    candles = _fetch_candles(asset=asset, start=opened_at, end=checked_at, url=url)
    rows = []
    for horizon in HORIZONS_MINUTES:
        horizon_end = opened_at + timedelta(minutes=horizon)
        subset = tuple(candle for candle in candles if _time_from_ms(candle.get("t")) < horizon_end)
        if not subset:
            continue
        close_bps, favorable_bps, adverse_bps = _path_bps(candles=subset, entry=entry, decision=decision)
        stop_50 = _stop_status(adverse_bps, threshold_bps=50.0)
        stop_100 = _stop_status(adverse_bps, threshold_bps=100.0)
        status = _status(close_bps=close_bps, stop_50=stop_50, stop_100=stop_100)
        rows.append(
            SurvivingAlphaExitRegimeCandidate(
                candidate_id=f"{asset.lower()}-{decision.replace('paper_', '')}-{horizon}m-exit",
                asset=asset,
                decision=decision,
                horizon_minutes=horizon,
                status=status,
                close_return_bps=close_bps,
                max_favorable_bps=favorable_bps,
                max_adverse_bps=adverse_bps,
                stop_50bps_status=stop_50,
                stop_100bps_status=stop_100,
                priority=_priority(close_bps=close_bps, favorable_bps=favorable_bps, adverse_bps=adverse_bps),
                required_record="fresh trigger, actual fill price, stop rule, exit timestamp, and post-exit drift",
                next_step=_next_step(asset=asset, decision=decision, horizon=horizon, status=status),
            )
        )
    return tuple(rows)


def _path_bps(
    *,
    candles: tuple[dict[str, str], ...],
    entry: float,
    decision: str,
) -> tuple[float, float, float]:
    highs = tuple(_float(candle.get("h")) for candle in candles)
    lows = tuple(_float(candle.get("l")) for candle in candles)
    close = _float(candles[-1].get("c"))
    if decision == "paper_short":
        close_bps = (entry / close - 1.0) * 10_000.0 if close > 0.0 else 0.0
        favorable_bps = (entry / min(lows) - 1.0) * 10_000.0 if lows and min(lows) > 0.0 else 0.0
        adverse_bps = (entry / max(highs) - 1.0) * 10_000.0 if highs and max(highs) > 0.0 else 0.0
    else:
        close_bps = (close / entry - 1.0) * 10_000.0 if close > 0.0 else 0.0
        favorable_bps = (max(highs) / entry - 1.0) * 10_000.0 if highs else 0.0
        adverse_bps = (min(lows) / entry - 1.0) * 10_000.0 if lows else 0.0
    return close_bps, max(favorable_bps, 0.0), min(adverse_bps, 0.0)


def _status(*, close_bps: float, stop_50: str, stop_100: str) -> str:
    if close_bps <= 0.0:
        return "exit_horizon_negative"
    if stop_50 == "stop_survived" and close_bps >= 25.0:
        return "tight_stop_exit_candidate"
    if stop_100 == "stop_survived" and close_bps >= 100.0:
        return "wide_stop_exit_candidate"
    if stop_100 == "stop_survived":
        return "wide_stop_low_edge_watch"
    return "stop_risk_blocks_exit_regime"


def _priority(*, close_bps: float, favorable_bps: float, adverse_bps: float) -> float:
    return max(close_bps, 0.0) + max(favorable_bps, 0.0) * 0.25 + min(adverse_bps, 0.0) * 0.5


def _next_step(*, asset: str, decision: str, horizon: int, status: str) -> str:
    if status == "tight_stop_exit_candidate":
        return f"paper-repeat {asset} {decision} with {horizon}m exit and 50bps stop on a fresh trigger"
    if status == "wide_stop_exit_candidate":
        return f"paper-repeat {asset} {decision} with {horizon}m exit and 100bps stop on a fresh trigger"
    if status == "wide_stop_low_edge_watch":
        return f"keep {asset} {decision} as context; edge is positive but too small for a new repeat"
    if status == "exit_horizon_negative":
        return f"do not use {horizon}m exit for {asset} {decision}; close return was negative"
    return f"do not use {horizon}m exit for {asset} {decision}; stop risk still blocks it"


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
    return "stop_would_trigger" if max_adverse_bps <= -threshold_bps else "stop_survived"


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _time_from_ms(value: object) -> datetime:
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
    parser.add_argument("--path-risk-path", type=Path, default=ROOT / "current_surviving_alpha_path_risk.csv")
    parser.add_argument(
        "--second-tickets-path",
        type=Path,
        default=ROOT / "current_second_promoted_ticket_repeat_tickets.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_surviving_alpha_exit_regime_candidates.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_surviving_alpha_exit_regime_candidates.md",
    )
    args = parser.parse_args()

    rows = build_surviving_alpha_exit_regime_candidates(
        path_risk_path=args.path_risk_path,
        second_tickets_path=args.second_tickets_path,
    )
    write_surviving_alpha_exit_regime_candidates_csv(rows, output_path=args.output_path)
    write_surviving_alpha_exit_regime_candidates_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.candidate_id, row.status, f"{row.close_return_bps:.2f}", f"{row.max_adverse_bps:.2f}")


if __name__ == "__main__":
    main()
