from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ChainStablecoinMigrationForwardLabel:
    observed_at: str
    chain: str
    token_symbol: str
    migration_status: str
    side: str
    expected_direction: int
    week_change_usd: float
    week_change_pct: float
    start_price: float
    raw_return_1h: float | None
    raw_return_4h: float | None
    raw_return_12h: float | None
    directional_return_1h: float | None
    directional_return_4h: float | None
    directional_return_12h: float | None
    label_status: str
    next_step: str


def build_chain_stablecoin_migration_forward_labels(
    *,
    migration_path: Path = ROOT / "current_chain_stablecoin_migration.csv",
    top: int = 12,
) -> tuple[ChainStablecoinMigrationForwardLabel, ...]:
    rows = tuple(
        row
        for row in _read_rows(migration_path)
        if row.get("token_symbol") and row.get("status") != "chain_stablecoin_context"
    )[:top]
    candles_by_token = {
        token: _fetch_hyperliquid_candles(asset=token, start=_earliest_observed_at(rows))
        for token in sorted({row.get("token_symbol", "") for row in rows})
    }
    labels = tuple(
        _build_label(row=row, candles=candles_by_token.get(row.get("token_symbol", ""), ()))
        for row in rows
    )
    return tuple(sorted(labels, key=_sort_key, reverse=True))


def write_chain_stablecoin_migration_forward_labels_csv(
    rows: tuple[ChainStablecoinMigrationForwardLabel, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "observed_at",
                "chain",
                "token_symbol",
                "migration_status",
                "side",
                "expected_direction",
                "week_change_usd",
                "week_change_pct",
                "start_price",
                "raw_return_1h",
                "raw_return_4h",
                "raw_return_12h",
                "directional_return_1h",
                "directional_return_4h",
                "directional_return_12h",
                "label_status",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.observed_at,
                    row.chain,
                    row.token_symbol,
                    row.migration_status,
                    row.side,
                    row.expected_direction,
                    f"{row.week_change_usd:.8f}",
                    f"{row.week_change_pct:.8f}",
                    f"{row.start_price:.12f}",
                    _format_optional(row.raw_return_1h),
                    _format_optional(row.raw_return_4h),
                    _format_optional(row.raw_return_12h),
                    _format_optional(row.directional_return_1h),
                    _format_optional(row.directional_return_4h),
                    _format_optional(row.directional_return_12h),
                    row.label_status,
                    row.next_step,
                )
            )
    return output_path


def write_chain_stablecoin_migration_forward_labels_md(
    rows: tuple[ChainStablecoinMigrationForwardLabel, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_4h = tuple(row for row in rows if row.directional_return_4h is not None)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Chain Stablecoin Migration Forward Labels\n\n")
        handle.write(
            "This labels chain-level stablecoin migration against the mapped chain token. "
            "Positive directional return means the migration direction was right before costs and funding.\n\n"
        )
        handle.write(f"- total rows: `{len(rows)}`\n")
        handle.write(f"- labeled 4h rows: `{len(labeled_4h)}`\n\n")
        handle.write(
            "| chain | token | migration | dir | week change | week % | dir 1h | dir 4h | dir 12h | label status | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.chain} | {row.token_symbol} | {row.migration_status} | {row.expected_direction} | "
                f"{row.week_change_usd:.0f} | {row.week_change_pct:.6f} | "
                f"{_format_optional(row.directional_return_1h)} | "
                f"{_format_optional(row.directional_return_4h)} | "
                f"{_format_optional(row.directional_return_12h)} | "
                f"{row.label_status} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _build_label(
    *,
    row: dict[str, str],
    candles: tuple[dict[str, float], ...],
) -> ChainStablecoinMigrationForwardLabel:
    observed_at = _parse_datetime(row.get("timestamp", ""))
    expected_direction = _expected_direction(row)
    start_price = _close_at_or_after(candles, observed_at) or 0.0
    raw_return_1h = _forward_return(candles, observed_at, observed_at + timedelta(hours=1))
    raw_return_4h = _forward_return(candles, observed_at, observed_at + timedelta(hours=4))
    raw_return_12h = _forward_return(candles, observed_at, observed_at + timedelta(hours=12))
    label_status = _label_status(
        directional_return_4h=_directional_return(raw_return_4h, expected_direction),
        directional_return_12h=_directional_return(raw_return_12h, expected_direction),
    )
    token = row.get("token_symbol", "")
    return ChainStablecoinMigrationForwardLabel(
        observed_at=observed_at.isoformat(),
        chain=row.get("chain", ""),
        token_symbol=token,
        migration_status=row.get("status", ""),
        side=row.get("side", ""),
        expected_direction=expected_direction,
        week_change_usd=_float(row.get("week_change_usd")),
        week_change_pct=_float(row.get("week_change_pct")),
        start_price=start_price,
        raw_return_1h=raw_return_1h,
        raw_return_4h=raw_return_4h,
        raw_return_12h=raw_return_12h,
        directional_return_1h=_directional_return(raw_return_1h, expected_direction),
        directional_return_4h=_directional_return(raw_return_4h, expected_direction),
        directional_return_12h=_directional_return(raw_return_12h, expected_direction),
        label_status=label_status,
        next_step=_next_step(token=token, label_status=label_status),
    )


def _fetch_hyperliquid_candles(
    *,
    asset: str,
    start: datetime,
) -> tuple[dict[str, float], ...]:
    try:
        response = requests.post(
            HYPERLIQUID_INFO_URL,
            json={
                "type": "candleSnapshot",
                "req": {
                    "coin": asset,
                    "interval": "15m",
                    "startTime": int((start - timedelta(minutes=30)).timestamp() * 1000),
                    "endTime": int(datetime.now(UTC).timestamp() * 1000),
                },
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException:
        return ()
    return tuple(
        {
            "timestamp": float(row["t"]),
            "end_timestamp": float(row["T"]),
            "close": float(row["c"]),
        }
        for row in response.json()
    )


def _forward_return(
    candles: tuple[dict[str, float], ...],
    start: datetime,
    target: datetime,
) -> float | None:
    start_close = _close_at_or_after(candles, start)
    end_close = _close_at_or_after(candles, target)
    if start_close is None or end_close is None:
        return None
    return end_close / start_close - 1.0 if start_close > 0.0 else None


def _close_at_or_after(candles: tuple[dict[str, float], ...], target: datetime) -> float | None:
    target_ms = target.timestamp() * 1000
    for candle in candles:
        if candle["timestamp"] <= target_ms <= candle["end_timestamp"]:
            return candle["close"]
        if candle["timestamp"] >= target_ms:
            return candle["close"]
    return None


def _expected_direction(row: dict[str, str]) -> int:
    side = row.get("side", "")
    if side.startswith("long"):
        return 1
    if side.startswith("short"):
        return -1
    week_change = _float(row.get("week_change_usd"))
    if week_change > 0.0:
        return 1
    if week_change < 0.0:
        return -1
    return 0


def _directional_return(raw_return: float | None, direction: int) -> float | None:
    if raw_return is None or direction == 0:
        return None
    return raw_return * direction


def _label_status(
    *,
    directional_return_4h: float | None,
    directional_return_12h: float | None,
) -> str:
    if directional_return_4h is None:
        return "pending_4h"
    if directional_return_12h is None:
        return "labeled_4h_pending_12h"
    if directional_return_4h > 0.0 and directional_return_12h > 0.0:
        return "chain_migration_direction_supported"
    if directional_return_4h < 0.0 and directional_return_12h < 0.0:
        return "chain_migration_direction_contradicted"
    return "mixed_chain_migration_direction"


def _next_step(*, token: str, label_status: str) -> str:
    if label_status == "chain_migration_direction_supported":
        return f"repeat {token} chain-migration label and add venue, funding, and execution costs"
    if label_status == "chain_migration_direction_contradicted":
        return f"do not promote {token} chain-migration direction until a fresh independent snapshot confirms it"
    if label_status == "mixed_chain_migration_direction":
        return f"treat {token} chain migration as context and wait for a cleaner repeat label"
    if label_status == "labeled_4h_pending_12h":
        return f"wait for {token} 12h label before promotion"
    return f"wait for {token} 4h label"


def _earliest_observed_at(rows: tuple[dict[str, str], ...]) -> datetime:
    timestamps = tuple(_parse_datetime(row.get("timestamp", "")) for row in rows)
    return min(timestamps) if timestamps else datetime.now(UTC)


def _sort_key(row: ChainStablecoinMigrationForwardLabel) -> tuple[bool, float, float]:
    return (
        row.directional_return_4h is not None,
        row.directional_return_4h or -1.0,
        abs(row.week_change_usd),
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--migration-path",
        type=Path,
        default=ROOT / "current_chain_stablecoin_migration.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_chain_stablecoin_migration_forward_labels.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_chain_stablecoin_migration_forward_labels.md",
    )
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    rows = build_chain_stablecoin_migration_forward_labels(
        migration_path=args.migration_path,
        top=args.top,
    )
    write_chain_stablecoin_migration_forward_labels_csv(rows, output_path=args.output_path)
    write_chain_stablecoin_migration_forward_labels_md(rows, output_path=args.markdown_output_path)
    for row in rows:
        print(
            row.chain,
            row.token_symbol,
            row.label_status,
            f"dir4h={_format_optional(row.directional_return_4h)}",
            f"dir12h={_format_optional(row.directional_return_12h)}",
        )


if __name__ == "__main__":
    main()
