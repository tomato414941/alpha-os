from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import log, sqrt
from pathlib import Path

import requests


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class OptionsRealizedVolLabel:
    timestamp: str
    currency: str
    expiry: str
    days_to_expiry: float
    action: str
    atm_iv: float
    realized_vol_4h: float | None
    realized_vol_24h: float | None
    iv_premium_4h: float | None
    iv_premium_24h: float | None
    skew_iv: float | None
    term_iv_spread_to_next: float | None
    score: float
    reason: str


def build_options_realized_vol_labels(
    *,
    surface_path: Path = ROOT / "current_deribit_options_surface.csv",
) -> tuple[OptionsRealizedVolLabel, ...]:
    rows = _read_rows(surface_path)
    realized_by_currency = {
        currency: _realized_vols(currency)
        for currency in sorted({row["currency"] for row in rows})
    }
    labels = tuple(
        _build_label(row=row, realized=realized_by_currency.get(row["currency"], {}))
        for row in rows
    )
    return tuple(sorted(labels, key=lambda row: row.score, reverse=True))


def write_options_realized_vol_labels_csv(
    rows: tuple[OptionsRealizedVolLabel, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "currency",
                "expiry",
                "days_to_expiry",
                "action",
                "atm_iv",
                "realized_vol_4h",
                "realized_vol_24h",
                "iv_premium_4h",
                "iv_premium_24h",
                "skew_iv",
                "term_iv_spread_to_next",
                "score",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.currency,
                    row.expiry,
                    f"{row.days_to_expiry:.4f}",
                    row.action,
                    f"{row.atm_iv:.4f}",
                    _format_optional(row.realized_vol_4h),
                    _format_optional(row.realized_vol_24h),
                    _format_optional(row.iv_premium_4h),
                    _format_optional(row.iv_premium_24h),
                    _format_optional(row.skew_iv),
                    _format_optional(row.term_iv_spread_to_next),
                    f"{row.score:.6f}",
                    row.reason,
                )
            )
    return output_path


def write_options_realized_vol_labels_md(
    rows: tuple[OptionsRealizedVolLabel, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Deribit Options Realized Vol Labels\n\n")
        handle.write(
            "This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. "
            "It is a fast IV-vs-realized context label, not an options backtest.\n\n"
        )
        handle.write(
            "| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |\n"
        )
        handle.write(
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.currency} | "
                f"{row.expiry} | "
                f"{row.days_to_expiry:.2f} | "
                f"{row.action} | "
                f"{row.atm_iv:.2f} | "
                f"{'' if row.realized_vol_4h is None else f'{row.realized_vol_4h:.2f}'} | "
                f"{'' if row.realized_vol_24h is None else f'{row.realized_vol_24h:.2f}'} | "
                f"{'' if row.iv_premium_4h is None else f'{row.iv_premium_4h:.2f}'} | "
                f"{'' if row.iv_premium_24h is None else f'{row.iv_premium_24h:.2f}'} | "
                f"{'' if row.skew_iv is None else f'{row.skew_iv:.2f}'} | "
                f"{'' if row.term_iv_spread_to_next is None else f'{row.term_iv_spread_to_next:.2f}'} | "
                f"{row.score:.4f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Positive IV premium means listed ATM IV is above recent realized "
            "volatility. This can point to vol-selling or event-premium candidates, "
            "but it still needs realized-vol forecasts, hedge PnL, option spreads, "
            "margin, and tail-risk controls.\n"
        )
    return output_path


def _build_label(
    *,
    row: dict[str, str],
    realized: dict[str, float],
) -> OptionsRealizedVolLabel:
    atm_iv = float(row["atm_iv"])
    realized_4h = realized.get("4h")
    realized_24h = realized.get("24h")
    premium_4h = None if realized_4h is None else atm_iv - realized_4h
    premium_24h = None if realized_24h is None else atm_iv - realized_24h
    skew = _optional_float(row.get("skew_iv", ""))
    term = _optional_float(row.get("term_iv_spread_to_next", ""))
    action, reason = _classify(
        premium_24h=premium_24h,
        skew=skew,
        term=term,
    )
    return OptionsRealizedVolLabel(
        timestamp=row["timestamp"],
        currency=row["currency"],
        expiry=row["expiry"],
        days_to_expiry=float(row["days_to_expiry"]),
        action=action,
        atm_iv=atm_iv,
        realized_vol_4h=realized_4h,
        realized_vol_24h=realized_24h,
        iv_premium_4h=premium_4h,
        iv_premium_24h=premium_24h,
        skew_iv=skew,
        term_iv_spread_to_next=term,
        score=_score(premium_24h=premium_24h, skew=skew, term=term),
        reason=reason,
    )


def _realized_vols(currency: str) -> dict[str, float]:
    candles = _fetch_hyperliquid_candles(currency)
    return {
        "4h": _annualized_realized_vol(candles[-17:]),
        "24h": _annualized_realized_vol(candles[-97:]),
    }


def _fetch_hyperliquid_candles(currency: str) -> tuple[dict[str, float], ...]:
    end = datetime.now(UTC)
    start = end - timedelta(hours=30)
    response = requests.post(
        HYPERLIQUID_INFO_URL,
        json={
            "type": "candleSnapshot",
            "req": {
                "coin": currency,
                "interval": "15m",
                "startTime": int(start.timestamp() * 1000),
                "endTime": int(end.timestamp() * 1000),
            },
        },
        timeout=30,
    )
    response.raise_for_status()
    return tuple({"close": float(row["c"])} for row in response.json())


def _annualized_realized_vol(candles: tuple[dict[str, float], ...]) -> float | None:
    if len(candles) < 3:
        return None
    returns = tuple(
        log(candles[index]["close"] / candles[index - 1]["close"])
        for index in range(1, len(candles))
        if candles[index - 1]["close"] > 0.0
    )
    if len(returns) < 2:
        return None
    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)
    return sqrt(variance) * sqrt(365.0 * 24.0 * 4.0) * 100.0


def _classify(
    *,
    premium_24h: float | None,
    skew: float | None,
    term: float | None,
) -> tuple[str, str]:
    premium = premium_24h or 0.0
    skew_value = skew or 0.0
    term_value = term or 0.0
    if premium >= 20.0 and skew_value >= 5.0:
        return "rich_put_skew_vol_premium_watch", "ATM IV is rich to 24h realized and puts are richer than calls"
    if premium >= 20.0:
        return "rich_vol_premium_watch", "ATM IV is rich to 24h realized volatility"
    if premium <= -10.0:
        return "cheap_vol_watch", "ATM IV is below recent realized volatility"
    if abs(term_value) >= 5.0:
        return "term_structure_watch", "term structure is the dominant options-surface feature"
    return "realized_vol_context", "IV-realized context exists but no strong premium selected yet"


def _score(
    *,
    premium_24h: float | None,
    skew: float | None,
    term: float | None,
) -> float:
    return max(abs(premium_24h or 0.0) + abs(skew or 0.0) + abs(term or 0.0), 0.0)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--surface-path",
        type=Path,
        default=ROOT / "current_deribit_options_surface.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_deribit_options_realized_vol_labels.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_deribit_options_realized_vol_labels.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_options_realized_vol_labels(surface_path=args.surface_path)
    write_options_realized_vol_labels_csv(rows, output_path=args.output_path)
    write_options_realized_vol_labels_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.currency,
            row.expiry,
            row.action,
            f"iv={row.atm_iv:.2f}",
            f"rv24={'' if row.realized_vol_24h is None else f'{row.realized_vol_24h:.2f}'}",
            f"prem24={'' if row.iv_premium_24h is None else f'{row.iv_premium_24h:.2f}'}",
            f"score={row.score:.2f}",
        )


if __name__ == "__main__":
    main()
