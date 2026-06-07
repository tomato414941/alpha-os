from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from math import log10
from pathlib import Path

import requests


CRYPTOBEACON_UNLOCKS_URL = "https://www.cryptobeacon.io/unlocks"


@dataclass(frozen=True)
class UnlockEvent:
    timestamp: str
    name: str
    symbol: str
    unlock_type: str
    unlock_value_usd: float
    percent_supply: float
    impact: str
    days_until: int | None
    action: str
    score: float


class TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)


def fetch_unlock_events(
    *,
    url: str = CRYPTOBEACON_UNLOCKS_URL,
    now: datetime | None = None,
) -> tuple[UnlockEvent, ...]:
    now = now or datetime.now(tz=UTC)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    parts = _html_text_parts(response.text)
    return _parse_cliff_unlock_events(parts, timestamp=now.isoformat())


def write_unlock_events(events: tuple[UnlockEvent, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "name",
                "symbol",
                "unlock_type",
                "unlock_value_usd",
                "percent_supply",
                "impact",
                "days_until",
                "action",
                "score",
            )
        )
        for event in events:
            writer.writerow(
                (
                    event.timestamp,
                    event.name,
                    event.symbol,
                    event.unlock_type,
                    f"{event.unlock_value_usd:.2f}",
                    f"{event.percent_supply:.4f}",
                    event.impact,
                    "" if event.days_until is None else event.days_until,
                    event.action,
                    f"{event.score:.8f}",
                )
            )
    return output_path


def write_markdown(events: tuple[UnlockEvent, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Token Unlock Snapshot\n\n")
        handle.write(
            "This extracts scheduled cliff unlocks from CryptoBeacon. It is supply-event context, not a trade instruction.\n\n"
        )
        handle.write(
            "| symbol | name | type | value USD | % supply | impact | in | action | score |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: |\n")
        for event in events[:30]:
            days = "" if event.days_until is None else str(event.days_until)
            handle.write(
                f"| {event.symbol} | {event.name} | {event.unlock_type} | "
                f"{event.unlock_value_usd:.2f} | {event.percent_supply:.4f} | "
                f"{event.impact} | {days} | {event.action} | {event.score:.6f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Large or high-supply unlocks can matter as supply pressure or attention catalysts, but the effect must be joined to tradable venue state and labeled against forward returns.\n"
        )
    return output_path


def _html_text_parts(html: str) -> tuple[str, ...]:
    parser = TextParser()
    parser.feed(html)
    return tuple(parser.parts)


def _parse_cliff_unlock_events(
    parts: tuple[str, ...],
    *,
    timestamp: str,
) -> tuple[UnlockEvent, ...]:
    try:
        start = parts.index("Token") + 8
        end = parts.index("Daily linear vesting")
    except ValueError:
        return ()

    events: list[UnlockEvent] = []
    index = start
    while index + 8 <= end:
        name = parts[index]
        symbol = parts[index + 1]
        category = parts[index + 2]
        unlock_type = parts[index + 3]
        value_text = parts[index + 4]
        supply_text = parts[index + 5]
        historical_impact = parts[index + 6]
        impact = parts[index + 7]
        days_text = parts[index + 8]
        if not _looks_like_event_row(
            category=category,
            unlock_type=unlock_type,
            value_text=value_text,
            supply_text=supply_text,
            historical_impact=historical_impact,
            impact=impact,
        ):
            index += 1
            continue
        unlock_value_usd = _money_to_float(value_text)
        percent_supply = _percent_to_float(supply_text)
        days_until = _days_until(days_text)
        events.append(
            UnlockEvent(
                timestamp=timestamp,
                name=name,
                symbol=symbol,
                unlock_type=unlock_type,
                unlock_value_usd=unlock_value_usd,
                percent_supply=percent_supply,
                impact=impact,
                days_until=days_until,
                action=_action_for_event(
                    unlock_value_usd=unlock_value_usd,
                    percent_supply=percent_supply,
                    days_until=days_until,
                    impact=impact,
                ),
                score=_score_event(
                    unlock_value_usd=unlock_value_usd,
                    percent_supply=percent_supply,
                    days_until=days_until,
                ),
            )
        )
        index += 9
    return tuple(sorted(events, key=lambda event: event.score, reverse=True))


def _looks_like_event_row(
    *,
    category: str,
    unlock_type: str,
    value_text: str,
    supply_text: str,
    historical_impact: str,
    impact: str,
) -> bool:
    return (
        category == "—"
        and unlock_type in {"CLIFF", "PERIODIC", "MONTHLY"}
        and value_text.startswith("$")
        and supply_text.endswith("%")
        and historical_impact == "—"
        and impact in {"CRITICAL", "HIGH", "MEDIUM"}
    )


def _money_to_float(value: str) -> float:
    cleaned = value.strip().replace("$", "").replace(",", "")
    multiplier = 1.0
    if cleaned.endswith("K"):
        multiplier = 1_000.0
        cleaned = cleaned[:-1]
    elif cleaned.endswith("M"):
        multiplier = 1_000_000.0
        cleaned = cleaned[:-1]
    elif cleaned.endswith("B"):
        multiplier = 1_000_000_000.0
        cleaned = cleaned[:-1]
    return float(cleaned) * multiplier


def _percent_to_float(value: str) -> float:
    return float(value.strip().replace("%", ""))


def _days_until(value: str) -> int | None:
    text = value.strip().lower()
    if text == "today":
        return 0
    if text.endswith("d") and text[:-1].isdigit():
        return int(text[:-1])
    return None


def _action_for_event(
    *,
    unlock_value_usd: float,
    percent_supply: float,
    days_until: int | None,
    impact: str,
) -> str:
    near = days_until is not None and days_until <= 30
    if near and (impact == "CRITICAL" or percent_supply >= 5.0):
        return "unlock_supply_shock_watch"
    if near and unlock_value_usd >= 10_000_000.0:
        return "large_unlock_watch"
    return "unlock_context"


def _score_event(
    *,
    unlock_value_usd: float,
    percent_supply: float,
    days_until: int | None,
) -> float:
    urgency = 1.0 if days_until is None else 1.0 / (1.0 + max(days_until, 0))
    value_score = log10(max(unlock_value_usd, 1.0))
    supply_score = max(percent_supply, 0.1) ** 0.5
    return value_score * supply_score * (1.0 + urgency)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=CRYPTOBEACON_UNLOCKS_URL)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_token_unlock_snapshot.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_token_unlock_snapshot.md",
    )
    args = parser.parse_args()

    events = fetch_unlock_events(url=args.url)
    write_unlock_events(events, output_path=args.output_path)
    write_markdown(events, output_path=args.markdown_output_path)
    for event in events[:10]:
        print(
            event.symbol,
            event.action,
            f"value={event.unlock_value_usd:.2f}",
            f"supply={event.percent_supply:.2f}",
            f"days={event.days_until}",
            f"score={event.score:.4f}",
        )


if __name__ == "__main__":
    main()
